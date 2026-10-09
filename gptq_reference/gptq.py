import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        if static_groups and groupsize <= 0:
            raise ValueError("static_groups requires a positive groupsize")

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        groups = None
        dynamic_group_scales = []
        dynamic_group_zeros = []
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = torch.arange(self.columns, device=self.dev)
            invperm = perm

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            # Dynamic GPTQ used to leave only the final group's
                            # qparams in self.quantizer.  Capture each group at
                            # the exact point it is selected so exporting the
                            # metadata does not change the quantized weights.
                            dynamic_group_scales.append(
                                self.quantizer.scale.detach().clone()
                            )
                            dynamic_group_zeros.append(
                                self.quantizer.zero.detach().clone()
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        # Preserve enough metadata to reconstruct Q in its final/original K
        # order.  Dynamic groups with act-order are groups in the permuted
        # quantization order, so g_idx maps every original K channel to the
        # scale/zero group that actually produced it.
        self.quantizer.gptq_group_size = int(groupsize)
        self.quantizer.gptq_static_groups = bool(static_groups)
        self.quantizer.gptq_act_order = bool(actorder)

        if groupsize > 0:
            expected_groups = (self.columns + groupsize - 1) // groupsize
            expected_shape = (self.rows, expected_groups)

            if static_groups:
                group_scales = torch.cat(
                    [group.scale.detach().clone() for group in groups], dim=1
                )
                group_zeros = torch.cat(
                    [group.zero.detach().clone() for group in groups], dim=1
                )
                # Static qparams are indexed by the original contiguous K
                # group, even when columns were visited in activation order.
                g_idx = torch.arange(self.columns, device=self.dev) // groupsize
                qparam_layout = "original_contiguous"
            else:
                if len(dynamic_group_scales) != expected_groups:
                    raise RuntimeError(
                        "collected an unexpected number of dynamic GPTQ groups: "
                        f"got {len(dynamic_group_scales)}, expected {expected_groups}"
                    )
                group_scales = torch.cat(dynamic_group_scales, dim=1)
                group_zeros = torch.cat(dynamic_group_zeros, dim=1)
                quant_order_groups = (
                    torch.arange(self.columns, device=self.dev) // groupsize
                )
                # invperm[original_k] is that channel's position in the
                # quantization order.  Indexing the quant-order group vector
                # by invperm therefore yields the group for original K order.
                g_idx = quant_order_groups[invperm]
                qparam_layout = "dynamic_act_order" if actorder else "dynamic"

            if tuple(group_scales.shape) != expected_shape:
                raise RuntimeError(
                    f"collected GPTQ scales have shape {tuple(group_scales.shape)}, "
                    f"expected {expected_shape}"
                )
            if tuple(group_zeros.shape) != expected_shape:
                raise RuntimeError(
                    f"collected GPTQ zeros have shape {tuple(group_zeros.shape)}, "
                    f"expected {expected_shape}"
                )
            self.quantizer.scale = group_scales
            self.quantizer.zero = group_zeros
        else:
            # Full-row quantization has one qparam group for every K channel.
            g_idx = torch.zeros(self.columns, dtype=torch.long, device=self.dev)
            qparam_layout = "full_row"

        self.quantizer.gptq_qparam_layout = qparam_layout
        self.quantizer.gptq_g_idx = g_idx.detach().to(
            device="cpu", dtype=torch.int32
        )
        self.quantizer.gptq_perm = perm.detach().to(
            device="cpu", dtype=torch.int32
        )
        self.quantizer.gptq_invperm = invperm.detach().to(
            device="cpu", dtype=torch.int32
        )

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
