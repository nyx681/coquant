import math
import os
import sys
import torch
import torch.nn as nn

_KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels"))
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)
import awq_inference_engine  # with CUDA kernels

# Set WQ0124_DEBUG_ROUNDTRIP=1 to print per-layer roundtrip checks in from_linear.
DEBUG_WQ0124_ROUNDTRIP = os.environ.get("WQ0124_DEBUG_ROUNDTRIP", "0") == "1"

_CUDA_NATIVE_MASK_PERMUTATIONS = {}


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    """Pack int weight on the same device as unpacked_qweight (CPU/CUDA)."""
    N, K = unpacked_qweight.shape[0], unpacked_qweight.shape[1]
    packed_kernel = unpacked_qweight.to(torch.int64).reshape(N, K // 32, 32)
    packed_kernel = (
        packed_kernel.reshape(N, K // 32, 4, 4, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(N, K // 32, 32)
    )
    packed_kernel = packed_kernel.reshape(N, K // 32, 4, 8)
    packed_kernel = (
        packed_kernel.reshape(N, K // 32, 4, 4, 2)
        .permute(0, 1, 2, 4, 3)
        .reshape(N, K)
    )
    packed_kernel = packed_kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    packed_kernel = packed_kernel.permute(0, 2, 1, 3).reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    packed_kernel = (
        packed_kernel[..., 0]
        | (packed_kernel[..., 1] << 4)
        | (packed_kernel[..., 2] << 8)
        | (packed_kernel[..., 3] << 12)
    )
    return packed_kernel.reshape(N // interleave, K).to(torch.int16).contiguous()


def unpack_intweight(packed_qweight, out_features, in_features, interleave=4, kstride=64):
    """Inverse of pack_intweight. Returns int32 [out_features, in_features] on same device."""
    N, K = out_features, in_features
    device = packed_qweight.device
    pk = packed_qweight.to(torch.int64).reshape(
        N // interleave, K // kstride, kstride
    )
    shifts = torch.tensor([0, 4, 8, 12], device=device, dtype=torch.int64)
    kernel = (pk.unsqueeze(-1) >> shifts) & 0xF
    kernel = kernel.reshape(N // interleave, K // kstride, interleave, kstride)
    kernel = kernel.permute(0, 2, 1, 3).reshape(N, K)
    kernel = (
        kernel.reshape(N, K // 32, 4, 2, 4)
        .permute(0, 1, 2, 4, 3)
        .reshape(N, K // 32, 4, 8)
    )
    kernel = (
        kernel.reshape(N, K // 32, 4, 4, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(N, K // 32, 32)
    )
    return kernel.reshape(N, K).to(torch.int32)

def _reorder_like_intweight(unpacked, interleave=4, kstride=64):
    """Apply the same layout permutation used before int4 packing."""
    N, K = unpacked.shape[0], unpacked.shape[1]
    kernel = unpacked.reshape(N, K // 32, 32)
    kernel = (
        kernel.reshape(N, K // 32, 4, 4, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(N, K // 32, 32)
    )
    kernel = kernel.reshape(N, K // 32, 4, 8)
    kernel = (
        kernel.reshape(N, K // 32, 4, 4, 2)
        .permute(0, 1, 2, 4, 3)
        .reshape(N, K)
    )
    kernel = kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    return kernel.permute(0, 2, 1, 3).reshape(
        N // interleave, K // kstride, kstride, interleave
    )


def pack_binary_mask_like_qweight(mask, interleave=4, kstride=64):
    """Pack a binary mask in qweight layout, 8 mask bits per byte."""
    N, K = mask.shape[0], mask.shape[1]
    assert K % 2 == 0
    kernel = _reorder_like_intweight(mask.to(torch.uint8), interleave, kstride)
    nibble_mask = (
        kernel[..., 0]
        | (kernel[..., 1] << 1)
        | (kernel[..., 2] << 2)
        | (kernel[..., 3] << 3)
    ).reshape(N // interleave, K)
    return (nibble_mask[:, 0::2] | (nibble_mask[:, 1::2] << 4)).to(
        torch.uint8
    ).contiguous()


def unpack_binary_mask_like_qweight(packed, out_features, in_features, interleave=4, kstride=64):
    """Restore a qweight-layout packed binary mask to bool [out_features, in_features]."""
    N, K = out_features, in_features
    packed = packed.reshape(N // interleave, K // 2)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble_mask = torch.empty(
        (N // interleave, K), dtype=torch.uint8, device=packed.device
    )
    nibble_mask[:, 0::2] = low
    nibble_mask[:, 1::2] = high
    packed_int16 = (
        (nibble_mask & 1).to(torch.int32)
        | (((nibble_mask >> 1) & 1).to(torch.int32) << 4)
        | (((nibble_mask >> 2) & 1).to(torch.int32) << 8)
        | (((nibble_mask >> 3) & 1).to(torch.int32) << 12)
    ).to(torch.int16)
    return unpack_intweight(
        packed_int16, out_features, in_features, interleave, kstride
    ).bool()


def _cuda_native_mask_permutation(device):
    """Map an [8 packed-N, 32 K-byte] source tile to a 16x16 b8 ldmatrix tile."""
    device = torch.device(device)
    key = (device.type, device.index)
    permutation = _CUDA_NATIVE_MASK_PERMUTATIONS.get(key)
    if permutation is None or permutation.device != device:
        destination_indices = []
        for source_row in range(8):
            for source_col in range(32):
                matrix_row = (
                    (source_col & 3) * 4
                    + ((source_col >> 2) & 1) * 2
                    + ((source_row & 3) >> 1)
                )
                matrix_col = (
                    (source_col >> 3)
                    + (source_row & 1) * 4
                    + (source_row >> 2) * 8
                )
                destination_indices.append(matrix_row * 16 + matrix_col)
        permutation = torch.tensor(
            destination_indices, dtype=torch.long, device=device
        )
        _CUDA_NATIVE_MASK_PERMUTATIONS[key] = permutation
    return permutation


def prepack_binary_mask_for_cuda(packed, out_features, in_features):
    """
    Convert qweight-layout packed mask to the CUDA-native b8 tile layout.

    Input is flat [N*K/8] in logical [N/4, K/2] layout. Output has the
    same dtype and numel, flattened from [K/64, N/32, 16, 16].
    """
    N, K = out_features, in_features
    if N % 32 != 0 or K % 64 != 0:
        raise ValueError("CUDA-native mask layout requires N%32==0 and K%64==0")
    packed = packed.reshape(N // 4, K // 2)
    n_tiles = N // 32
    k_tiles = K // 64
    permutation = _cuda_native_mask_permutation(packed.device)

    source_tiles = (
        packed.reshape(n_tiles, 8, k_tiles, 32)
        .permute(2, 0, 1, 3)
        .contiguous()
        .reshape(k_tiles, n_tiles, 256)
    )
    native_tiles = torch.empty_like(source_tiles)
    native_tiles[..., permutation] = source_tiles
    return native_tiles.reshape(-1).contiguous()


def unpack_cuda_native_mask_to_qweight_layout(packed, out_features, in_features):
    """Inverse of prepack_binary_mask_for_cuda; preserves dtype and numel."""
    N, K = out_features, in_features
    if N % 32 != 0 or K % 64 != 0:
        raise ValueError("CUDA-native mask layout requires N%32==0 and K%64==0")
    n_tiles = N // 32
    k_tiles = K // 64
    permutation = _cuda_native_mask_permutation(packed.device)
    source_tiles = packed.reshape(k_tiles, n_tiles, 256)[..., permutation]
    return (
        source_tiles.reshape(k_tiles, n_tiles, 8, 32)
        .permute(1, 2, 0, 3)
        .contiguous()
        .reshape(-1)
    )


def _int_quantize_zeros_per_path(
    scales,
    scaled_zeros,
    out_features,
    in_features,
    group_size,
    w_bit,
    group_ids=None,
):
    """
    Quantized int for weight=0 with each path's scale/zero (legacy column-wise rule).
    scales / scaled_zeros: stored layout [scale_cols, out_features] (transposed in buffers).
    Returns int32 [out_features, in_features].
    """
    if group_ids is None:
        group_ids = torch.arange(in_features, device=scales.device) // group_size
    else:
        group_ids = group_ids.to(device=scales.device, dtype=torch.long)
    qscales = scales.t().float()[:, group_ids]
    # scaled_zeros buffer holds -(scale * zero); quant uses (w + scale*zero) / scale
    scale_zeros = (-scaled_zeros.t().float())[:, group_ids]
    ints = torch.round(scale_zeros / qscales)
    return torch.clamp(ints.to(torch.int32), 0, 2**w_bit - 1)


def split_dual_packed_qweights(
    full_qweight,
    full_mask_packed,
    out_features,
    in_features,
    interleave=4,
    kstride=64,
    w_bit=4,
    group_size=128,
    awq_scales=None,
    awq_scaled_zeros=None,
    gptq_scales=None,
    gptq_scaled_zeros=None,
    gptq_g_idx=None,
):
    """Split full packed qweight into awq/gptq packed kernels using binary mask."""
    full_int = unpack_intweight(
        full_qweight, out_features, in_features, interleave, kstride
    )
    mask = unpack_binary_mask(
        full_mask_packed, out_features, in_features
    ).to(device=full_int.device)

    if (
        awq_scales is not None
        and awq_scaled_zeros is not None
        and gptq_scales is not None
        and gptq_scaled_zeros is not None
    ):
        awq_int_zero = _int_quantize_zeros_per_path(
            awq_scales, awq_scaled_zeros, out_features, in_features, group_size, w_bit
        )
        gptq_int_zero = _int_quantize_zeros_per_path(
            gptq_scales,
            gptq_scaled_zeros,
            out_features,
            in_features,
            group_size,
            w_bit,
            group_ids=gptq_g_idx,
        )
        awq_int = torch.where(mask, full_int, awq_int_zero)
        gptq_int = torch.where(~mask, full_int, gptq_int_zero)
    else:
        mask_i = mask.to(full_int.dtype)
        awq_int = (full_int * mask_i).to(torch.int32)
        gptq_int = (full_int * (1 - mask_i)).to(torch.int32)

    awq_packed = pack_intweight(awq_int.contiguous(), interleave, kstride)
    gptq_packed = pack_intweight(gptq_int.contiguous(), interleave, kstride)
    return awq_packed, gptq_packed


def _quantize_masked_w_columnwise(
    awq_w,
    gptq_w,
    awq_scale_zeros,
    gptq_scale_zeros,
    awq_qscales,
    gptq_qscales,
    in_features,
    group_size,
    gptq_g_idx=None,
):
    """Legacy per-column int quantize from masked awq_w / gptq_w (same as old WQLinear0124.from_linear)."""
    awq_cols, gptq_cols = [], []
    for idx in range(in_features):
        g = idx // group_size
        awq_cols.append(
            torch.round((awq_w[:, idx] + awq_scale_zeros[:, g]) / awq_qscales[:, g])
            .to(torch.int)[:, None]
        )
        gptq_g = g if gptq_g_idx is None else int(gptq_g_idx[idx].item())
        gptq_cols.append(
            torch.round(
                (gptq_w[:, idx] + gptq_scale_zeros[:, gptq_g])
                / gptq_qscales[:, gptq_g]
            )
            .to(torch.int)[:, None]
        )
    awq_int = torch.cat(awq_cols, dim=1).to(torch.int32)
    gptq_int = torch.cat(gptq_cols, dim=1).to(torch.int32)
    return awq_int, gptq_int


def _debug_verify_full_to_dual_qweights(
    awq_linear,
    full_intweight,
    full_mask_dev,
    awq_w,
    gptq_w,
    awq_scale_zeros,
    gptq_scale_zeros,
    awq_qscales,
    gptq_qscales,
    group_size,
    w_bit,
    gptq_g_idx=None,
):
    """
    After full_qweight + full_mask_packed are written, try to restore awq/gptq packed
    weights and compare against direct references.
    """
    il, ks = awq_linear.interleave, 64
    N, K = awq_linear.out_features, awq_linear.in_features
    max_int = 2**w_bit - 1

    row_mask_packed = pack_binary_mask(full_mask_dev)
    awq_rest, gptq_rest = split_dual_packed_qweights(
        awq_linear.full_qweight,
        row_mask_packed,
        N,
        K,
        interleave=il,
        kstride=ks,
        w_bit=w_bit,
        group_size=group_size,
        awq_scales=awq_linear.awq_scales,
        awq_scaled_zeros=awq_linear.awq_scaled_zeros,
        gptq_scales=awq_linear.gptq_scales,
        gptq_scaled_zeros=awq_linear.gptq_scaled_zeros,
        gptq_g_idx=gptq_g_idx,
    )

    # Reference: legacy column-wise quantize on masked_w1 / masked_w2
    awq_ref_b, gptq_ref_b = _quantize_masked_w_columnwise(
        awq_w,
        gptq_w,
        awq_scale_zeros,
        gptq_scale_zeros,
        awq_qscales,
        gptq_qscales,
        K,
        group_size,
        gptq_g_idx=gptq_g_idx,
    )
    awq_ref_b = pack_intweight(torch.clamp(awq_ref_b, 0, max_int).contiguous(), il, ks)
    gptq_ref_b = pack_intweight(torch.clamp(gptq_ref_b, 0, max_int).contiguous(), il, ks)

    awq_ok = torch.equal(awq_rest, awq_ref_b)
    gptq_ok = torch.equal(gptq_rest, gptq_ref_b)

    if DEBUG_WQ0124_ROUNDTRIP or not (awq_ok and gptq_ok):
        tag = f"{N}x{K}"
        print(
            f"[WQ0124 roundtrip {tag}] "
            f"restore==legacy( masked_w1/w2 + scales ): awq={awq_ok} gptq={gptq_ok}"
        )
        if not awq_ok or not gptq_ok:
            ua, ub = unpack_intweight(awq_rest, N, K), unpack_intweight(awq_ref_b, N, K)
            ug, vg = unpack_intweight(gptq_rest, N, K), unpack_intweight(gptq_ref_b, N, K)
            print(
                f"  int mismatches: awq={(ua != ub).sum().item()} "
                f"gptq={(ug != vg).sum().item()}"
            )

    return awq_ok and gptq_ok


def pack_binary_mask(mask):
    """Pack a 0/1 mask into uint8 (8 bits per byte). mask: [out_features, in_features]."""
    mask_flat = mask.reshape(-1).to(torch.uint8)
    numel = mask_flat.numel()
    pad_len = (8 - numel % 8) % 8
    if pad_len:
        mask_flat = torch.cat(
            [
                mask_flat,
                torch.zeros(pad_len, dtype=torch.uint8, device=mask_flat.device),
            ]
        )
    shifts = torch.arange(8, device=mask_flat.device, dtype=torch.uint8)
    return (mask_flat.view(-1, 8) << shifts).sum(dim=1).to(torch.uint8)


def unpack_binary_mask(packed, out_features, in_features):
    """Restore a packed binary mask to bool tensor [out_features, in_features]."""
    numel = out_features * in_features
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)[:numel]
    return bits.reshape(out_features, in_features).bool()


def quantize_intweight_with_mask(
    weight,
    mask,
    awq_scale_zeros,
    gptq_scale_zeros,
    awq_qscales,
    gptq_qscales,
    group_size,
    w_bit,
    gptq_g_idx=None,
):
    """
    Per-element int quantize: use awq scale/zero where mask==1, else gptq.
    weight, mask: [out_features, in_features]
    awq_* / gptq_*: [out_features, num_groups]
    """
    mask = mask.reshape(weight.shape).bool()
    group_ids = torch.arange(weight.shape[1], device=weight.device) // group_size
    if gptq_g_idx is None:
        gptq_group_ids = group_ids
    else:
        gptq_group_ids = torch.as_tensor(
            gptq_g_idx, device=weight.device, dtype=torch.long
        ).reshape(-1)
        if gptq_group_ids.numel() != weight.shape[1]:
            raise ValueError(
                f"GPTQ g_idx has {gptq_group_ids.numel()} elements, "
                f"expected {weight.shape[1]}"
            )
    awq_sz = awq_scale_zeros[:, group_ids]
    gptq_sz = gptq_scale_zeros[:, gptq_group_ids]
    awq_sc = awq_qscales[:, group_ids]
    gptq_sc = gptq_qscales[:, gptq_group_ids]
    awq_int = torch.round((weight + awq_sz) / awq_sc)
    gptq_int = torch.round((weight + gptq_sz) / gptq_sc)
    full_int = torch.where(mask, awq_int, gptq_int)
    return torch.clamp(full_int.to(torch.int32), min=0, max=2**w_bit - 1)


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit

        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        awq_linear.qweight = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )

        zeros = zeros.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        if inputs.numel() / inputs.shape[-1] < 8:
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_inference_engine.gemm_forward_cuda_new(
                inputs, self.qweight, self.scales, self.scaled_zeros
            )  # - 8.0 * self.scales)
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )

class WQLinear0124_old(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit

        assert out_features % (self.interleave) == 0

        self.register_buffer(
            "awq_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "current_awq_x_s",
            torch.zeros(
                (
                    1, in_features
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None
        
        # 创建CUDA streams用于并行执行（如果设备是CUDA）
        # 注意：只在CUDA设备上创建，避免每次forward都创建新stream的开销
        if False:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        else:
            self.stream1 = None
            self.stream2 = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, awq_w=None, awq_scales=None, awq_zeros=None, gptq_w=None, gptq_scales=None, gptq_zeros=None, current_awq_x_s=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        current_device = linear.weight.device
        gptq_w = gptq_w.to(device=current_device, dtype=awq_w.dtype)
        gptq_scales = gptq_scales.to(device=current_device, dtype=awq_scales.dtype)
        gptq_zeros = gptq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        awq_w = awq_w.to(device=current_device, dtype=awq_w.dtype)
        awq_scales = awq_scales.to(device=current_device, dtype=awq_scales.dtype)
        awq_zeros = awq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        assert awq_scales is not None and awq_zeros is not None and gptq_scales is not None and gptq_zeros is not None
        awq_scale_zeros = awq_zeros * awq_scales
        gptq_scale_zeros = gptq_zeros * gptq_scales
        
        # awq_linear.test_awq_weight = awq_w.clone()
        pack_num = 32 // awq_linear.w_bit
        awq_qscales = torch.zeros(
            (
                awq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=awq_scales.device,
        )
        awq_qscales[:, : awq_scales.shape[1]] = awq_scales

        gptq_qscales = torch.zeros(
            (
                gptq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=gptq_scales.device,
        )
        gptq_qscales[:, : gptq_scales.shape[1]] = gptq_scales
        awq_linear.current_awq_x_s = current_awq_x_s.view(1, -1)
        # awq_linear.scales = scales.clone().half()
        awq_linear.awq_scales = awq_qscales.transpose(1, 0).contiguous()
        awq_linear.gptq_scales = gptq_qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        awq_intweight = []
        gptq_intweight = []
        for idx in range(awq_linear.in_features):
            awq_intweight.append(
                torch.round(
                    (awq_w[:, idx] + awq_scale_zeros[:, idx // group_size])
                    / awq_qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
            gptq_intweight.append(
                torch.round(
                    (gptq_w[:, idx] + gptq_scale_zeros[:, idx // group_size])
                    / gptq_qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        awq_intweight = torch.cat(awq_intweight, dim=1)
        gptq_intweight = torch.cat(gptq_intweight, dim=1)
        # intweight = intweight.t().contiguous()
        awq_intweight = awq_intweight.to(dtype=torch.int32)
        gptq_intweight = gptq_intweight.to(dtype=torch.int32)
        awq_linear.awq_qweight = pack_intweight(
            awq_intweight.contiguous(), interleave=4, kstride=64
        )
        awq_linear.gptq_qweight = pack_intweight(
            gptq_intweight.contiguous(), interleave=4, kstride=64
        )

        awq_zeros = awq_zeros.to(dtype=torch.int32)
        gptq_zeros = gptq_zeros.to(dtype=torch.int32)
        awq_scaled_zeros = torch.zeros_like(awq_qscales)
        gptq_scaled_zeros = torch.zeros_like(gptq_qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        # 注意：使用 awq_scales.shape[1] 而不是 awq_qscales.shape[1]，因为 awq_zeros 的形状与 awq_scales 匹配
        awq_scaled_zeros[:, : awq_scales.shape[1]] = -(
            awq_qscales[:, : awq_scales.shape[1]] * (awq_zeros.to(torch.float32))
        ).to(torch.float16)
        gptq_scaled_zeros[:, : gptq_scales.shape[1]] = -(
            gptq_qscales[:, : gptq_scales.shape[1]] * (gptq_zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.awq_scaled_zeros = awq_scaled_zeros.transpose(1, 0).contiguous()
        awq_linear.gptq_scaled_zeros = gptq_scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        
        # concatenated_qweight = torch.cat([self.awq_qweight, self.gptq_qweight], dim=-1)
        # concatenated_scales = torch.cat([self.awq_scales, self.gptq_scales], dim=0)
        # concatenated_scaled_zeros = torch.cat([self.awq_scaled_zeros, self.gptq_scaled_zeros], dim=0)
        # concatenated_input = torch.cat([inputs/self.current_awq_x_s, inputs], dim=-1)
        print('--------------------------------')
        print("infeature:", self.in_features)
        print("outfeatures:", self.out_features)
        print("group_size:", self.group_size)
        print(f"inputs.shape: {inputs.shape}")
        #print(f"concatenated_input.shape: {concatenated_input.shape}")
        print(f"self.current_awq_x_s.shape: {self.current_awq_x_s.shape}")
        print(f"self.awq_qweight.shape: {self.awq_qweight.shape}")
        print(f"self.gptq_qweight.shape: {self.gptq_qweight.shape}")
        #print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
        print(f"self.awq_scales.shape: {self.awq_scales.shape}")
        print(f"self.gptq_scales.shape: {self.gptq_scales.shape}")
        #print(f"concatenated_scales.shape: {concatenated_scales.shape}")
        
        print(f"self.awq_scaled_zeros.shape: {self.awq_scaled_zeros.shape}")
        print(f"self.gptq_scaled_zeros.shape: {self.gptq_scaled_zeros.shape}")
        print('--------------------------------')
        #print(f"concatenated_scaled_zeros.shape: {concatenated_scaled_zeros.shape}")

        #assert False
        if inputs.numel() / inputs.shape[-1] < 8:
            # 按第一个维度拼接 awq_qweight 和 gptq_qweight
            # print(f"concatenated_input.shape: {concatenated_input.shape}")
            # concatenated_out = awq_inference_engine.gemv_forward_cuda_new(
            #     concatenated_input,
            #     concatenated_qweight,
            #     concatenated_scales,
            #     concatenated_scaled_zeros,
            #     concatenated_input.numel() // concatenated_input.shape[-1],
            #     self.out_features,
            #     self.in_features*2,
            #     self.group_size,
            # )
            # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # print(f"self.out_features: {self.out_features}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
            awq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs/self.current_awq_x_s,#.div_(self.current_awq_x_s),
                self.awq_qweight,
                self.awq_scales,
                self.awq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            gptq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.gptq_qweight,
                self.gptq_scales,
                self.gptq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            out = awq_out + gptq_out
        else:
            # 使用CUDA Streams并行执行两个gemm操作
            # 注意：PyTorch的CUDA操作默认就是异步的，串行代码也可能自动并行
            # 但如果需要显式并行，可以使用预创建的stream
            if inputs.is_cuda and self.stream1 is not None and self.stream2 is not None:
                # 使用预创建的stream（避免每次forward都创建新stream的开销）
                # 预先计算AWQ的输入（避免在stream中计算）
                #print("here")
                awq_inputs = inputs / self.current_awq_x_s
                
                # # 在stream1中执行AWQ gemm
                with torch.cuda.stream(self.stream1):
                    awq_out = awq_inference_engine.gemm_forward_cuda_new(
                        awq_inputs,
                        self.awq_qweight,
                        self.awq_scales,
                        self.awq_scaled_zeros
                    )
                
                # 在stream2中执行GPTQ gemm（与stream1并行执行）
                with torch.cuda.stream(self.stream2):
                    gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                        inputs,
                        self.gptq_qweight,
                        self.gptq_scales,
                        self.gptq_scaled_zeros
                    )
                
                # 不需要显式synchronize，相加操作会自动等待两个stream完成
                # 这是PyTorch的默认行为，比显式synchronize更高效
                out = awq_out + gptq_out

                # concatenated_out = awq_inference_engine.gemm_forward_cuda_new(
                #         concatenated_input,
                #         concatenated_qweight,
                #         concatenated_scales,
                #         concatenated_scaled_zeros
                #     )

                # out = concatenated_out

                # assert (out==concatenated_out).all()

            else:
                # CPU模式或没有stream：串行执行
                # 注意：即使看起来是串行的，PyTorch的CUDA操作默认是异步的
                # GPU可能会自动并行执行（如果kernel支持且资源允许）
                #print("here")
                awq_out = awq_inference_engine.gemm_forward_cuda_new(
                    inputs/self.current_awq_x_s,
                    self.awq_qweight,
                    self.awq_scales,
                    self.awq_scaled_zeros
                )
                gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                    inputs,
                    self.gptq_qweight,
                    self.gptq_scales,
                    self.gptq_scaled_zeros
                )
                out = awq_out + gptq_out
            # # # print(f"inputs.shape: {inputs.shape}")
            # # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # # print(f"self.out_features: {self.out_features}")
            # # # print(f"awq_out.shape: {awq_out.shape}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )



class WQLinear0124_backup(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit

        assert out_features % (self.interleave) == 0

        self.register_buffer(
            "awq_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "current_awq_x_s",
            torch.zeros(
                (
                    1, in_features
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None
        
        # 创建CUDA streams用于并行执行（如果设备是CUDA）
        # 注意：只在CUDA设备上创建，避免每次forward都创建新stream的开销
        if False:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        else:
            self.stream1 = None
            self.stream2 = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, awq_w=None, awq_scales=None, awq_zeros=None, gptq_w=None, gptq_scales=None, gptq_zeros=None, current_awq_x_s=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        current_device = linear.weight.device
        gptq_w = gptq_w.to(device=current_device, dtype=awq_w.dtype)
        gptq_scales = gptq_scales.to(device=current_device, dtype=awq_scales.dtype)
        gptq_zeros = gptq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        awq_w = awq_w.to(device=current_device, dtype=awq_w.dtype)
        awq_scales = awq_scales.to(device=current_device, dtype=awq_scales.dtype)
        awq_zeros = awq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        assert awq_scales is not None and awq_zeros is not None and gptq_scales is not None and gptq_zeros is not None
        awq_scale_zeros = awq_zeros * awq_scales
        gptq_scale_zeros = gptq_zeros * gptq_scales
        
        # awq_linear.test_awq_weight = awq_w.clone()
        pack_num = 32 // awq_linear.w_bit
        awq_qscales = torch.zeros(
            (
                awq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=awq_scales.device,
        )
        awq_qscales[:, : awq_scales.shape[1]] = awq_scales

        gptq_qscales = torch.zeros(
            (
                gptq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=gptq_scales.device,
        )
        gptq_qscales[:, : gptq_scales.shape[1]] = gptq_scales
        awq_linear.current_awq_x_s = current_awq_x_s.view(1, -1)
        # awq_linear.scales = scales.clone().half()
        awq_linear.awq_scales = awq_qscales.transpose(1, 0).contiguous()
        awq_linear.gptq_scales = gptq_qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        awq_intweight = []
        gptq_intweight = []
        for idx in range(awq_linear.in_features):
            awq_intweight.append(
                torch.round(
                    (awq_w[:, idx] + awq_scale_zeros[:, idx // group_size])
                    / awq_qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
            gptq_intweight.append(
                torch.round(
                    (gptq_w[:, idx] + gptq_scale_zeros[:, idx // group_size])
                    / gptq_qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        awq_intweight = torch.cat(awq_intweight, dim=1)
        gptq_intweight = torch.cat(gptq_intweight, dim=1)
        # intweight = intweight.t().contiguous()
        awq_intweight = awq_intweight.to(dtype=torch.int32)
        gptq_intweight = gptq_intweight.to(dtype=torch.int32)
        awq_linear.awq_qweight = pack_intweight(
            awq_intweight.contiguous(), interleave=4, kstride=64
        )
        awq_linear.gptq_qweight = pack_intweight(
            gptq_intweight.contiguous(), interleave=4, kstride=64
        )

        awq_zeros = awq_zeros.to(dtype=torch.int32)
        gptq_zeros = gptq_zeros.to(dtype=torch.int32)
        awq_scaled_zeros = torch.zeros_like(awq_qscales)
        gptq_scaled_zeros = torch.zeros_like(gptq_qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        # 注意：使用 awq_scales.shape[1] 而不是 awq_qscales.shape[1]，因为 awq_zeros 的形状与 awq_scales 匹配
        awq_scaled_zeros[:, : awq_scales.shape[1]] = -(
            awq_qscales[:, : awq_scales.shape[1]] * (awq_zeros.to(torch.float32))
        ).to(torch.float16)
        gptq_scaled_zeros[:, : gptq_scales.shape[1]] = -(
            gptq_qscales[:, : gptq_scales.shape[1]] * (gptq_zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.awq_scaled_zeros = awq_scaled_zeros.transpose(1, 0).contiguous()
        awq_linear.gptq_scaled_zeros = gptq_scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        
        # concatenated_qweight = torch.cat([self.awq_qweight, self.gptq_qweight], dim=-1)
        # concatenated_scales = torch.cat([self.awq_scales, self.gptq_scales], dim=0)
        # concatenated_scaled_zeros = torch.cat([self.awq_scaled_zeros, self.gptq_scaled_zeros], dim=0)
        # concatenated_input = torch.cat([inputs/self.current_awq_x_s, inputs], dim=-1)
        # print('--------------------------------')
        # print("infeature:", self.in_features)
        # print("outfeatures:", self.out_features)
        # print("group_size:", self.group_size)
        # print(f"inputs.shape: {inputs.shape}")
        # #print(f"concatenated_input.shape: {concatenated_input.shape}")
        # print(f"self.current_awq_x_s.shape: {self.current_awq_x_s.shape}")
        # print(f"self.awq_qweight.shape: {self.awq_qweight.shape}")
        # print(f"self.gptq_qweight.shape: {self.gptq_qweight.shape}")
        # #print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
        # print(f"self.awq_scales.shape: {self.awq_scales.shape}")
        # print(f"self.gptq_scales.shape: {self.gptq_scales.shape}")
        # #print(f"concatenated_scales.shape: {concatenated_scales.shape}")
        
        # print(f"self.awq_scaled_zeros.shape: {self.awq_scaled_zeros.shape}")
        # print(f"self.gptq_scaled_zeros.shape: {self.gptq_scaled_zeros.shape}")
        # print('--------------------------------')
        #print(f"concatenated_scaled_zeros.shape: {concatenated_scaled_zeros.shape}")

        #assert False
        if inputs.numel() / inputs.shape[-1] < 8:
            # 按第一个维度拼接 awq_qweight 和 gptq_qweight
            # print(f"concatenated_input.shape: {concatenated_input.shape}")
            # concatenated_out = awq_inference_engine.gemv_forward_cuda_new(
            #     concatenated_input,
            #     concatenated_qweight,
            #     concatenated_scales,
            #     concatenated_scaled_zeros,
            #     concatenated_input.numel() // concatenated_input.shape[-1],
            #     self.out_features,
            #     self.in_features*2,
            #     self.group_size,
            # )
            # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # print(f"self.out_features: {self.out_features}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
            awq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs/self.current_awq_x_s,#.div_(self.current_awq_x_s),
                self.awq_qweight,
                self.awq_scales,
                self.awq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            gptq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.gptq_qweight,
                self.gptq_scales,
                self.gptq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            out = awq_out + gptq_out
        else:
            # 使用CUDA Streams并行执行两个gemm操作
            # 注意：PyTorch的CUDA操作默认就是异步的，串行代码也可能自动并行
            # 但如果需要显式并行，可以使用预创建的stream
            if False:#inputs.is_cuda and self.stream1 is not None and self.stream2 is not None:
                # 使用预创建的stream（避免每次forward都创建新stream的开销）
                # 预先计算AWQ的输入（避免在stream中计算）
                #print("here")
                awq_inputs = inputs / self.current_awq_x_s
                
                # # 在stream1中执行AWQ gemm
                with torch.cuda.stream(self.stream1):
                    awq_out = awq_inference_engine.gemm_forward_cuda_new(
                        awq_inputs,
                        self.awq_qweight,
                        self.awq_scales,
                        self.awq_scaled_zeros
                    )
                
                # 在stream2中执行GPTQ gemm（与stream1并行执行）
                with torch.cuda.stream(self.stream2):
                    gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                        inputs,
                        self.gptq_qweight,
                        self.gptq_scales,
                        self.gptq_scaled_zeros
                    )
                
                # 不需要显式synchronize，相加操作会自动等待两个stream完成
                # 这是PyTorch的默认行为，比显式synchronize更高效
                out = awq_out + gptq_out

                # concatenated_out = awq_inference_engine.gemm_forward_cuda_new(
                #         concatenated_input,
                #         concatenated_qweight,
                #         concatenated_scales,
                #         concatenated_scaled_zeros
                #     )

                # out = concatenated_out

                # assert (out==concatenated_out).all()

            else:
                # CPU模式或没有stream：串行执行
                # 注意：即使看起来是串行的，PyTorch的CUDA操作默认是异步的
                # GPU可能会自动并行执行（如果kernel支持且资源允许）
                #print("here")
                awq_out = awq_inference_engine.gemm_forward_cuda_new(
                    inputs/self.current_awq_x_s,
                    self.awq_qweight,
                    self.awq_scales,
                    self.awq_scaled_zeros
                )
                gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                    inputs,
                    self.gptq_qweight,
                    self.gptq_scales,
                    self.gptq_scaled_zeros
                )
                out = awq_out + gptq_out

                # out = awq_inference_engine.gemm_forward_cuda_new(
                #     inputs/self.current_awq_x_s + inputs,
                #     self.awq_qweight+self.gptq_qweight,
                #     self.awq_scales+self.gptq_scales,
                #     self.awq_scaled_zeros+self.gptq_scaled_zeros
                # )
            # # # print(f"inputs.shape: {inputs.shape}")
            # # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # # print(f"self.out_features: {self.out_features}")
            # # # print(f"awq_out.shape: {awq_out.shape}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


class WQLinear0124_trial(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit

        assert out_features % (self.interleave) == 0
        

        self.register_buffer(
            "awq_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "masks",
            torch.zeros(
                (
                    out_features ,
                    in_features 
                ),
                dtype=torch.uint8,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "current_awq_x_s",
            torch.zeros(
                (
                    1, in_features
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None
        
        # 创建CUDA streams用于并行执行（如果设备是CUDA）
        # 注意：只在CUDA设备上创建，避免每次forward都创建新stream的开销
        if False:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        else:
            self.stream1 = None
            self.stream2 = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, awq_w=None, awq_scales=None, awq_zeros=None, masks=None, gptq_scales=None, gptq_zeros=None, current_awq_x_s=None, 
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        current_device = linear.weight.device
        masks = masks.to(device=current_device, dtype=awq_w.dtype)
        gptq_scales = gptq_scales.to(device=current_device, dtype=awq_scales.dtype)
        gptq_zeros = gptq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        awq_w = awq_w.to(device=current_device, dtype=awq_w.dtype)
        awq_scales = awq_scales.to(device=current_device, dtype=awq_scales.dtype)
        awq_zeros = awq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        assert awq_scales is not None and awq_zeros is not None and gptq_scales is not None and gptq_zeros is not None
        awq_scale_zeros = awq_zeros * awq_scales
        gptq_scale_zeros = gptq_zeros * gptq_scales
        
        # awq_linear.test_awq_weight = awq_w.clone()
        pack_num = 32 // awq_linear.w_bit
        awq_qscales = torch.zeros(
            (
                awq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=awq_scales.device,
        )
        awq_qscales[:, : awq_scales.shape[1]] = awq_scales

        gptq_qscales = torch.zeros(
            (
                gptq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=gptq_scales.device,
        )
        gptq_qscales[:, : gptq_scales.shape[1]] = gptq_scales
        awq_linear.current_awq_x_s = current_awq_x_s.view(1, -1)
        # awq_linear.scales = scales.clone().half()
        awq_linear.awq_scales = awq_qscales.transpose(1, 0).contiguous()
        awq_linear.gptq_scales = gptq_qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        awq_intweight = []
        #gptq_intweight = []
        #int_masks = []

        for idx in range(awq_linear.in_features):
            awq_intweight.append(
                torch.round(
                    (awq_w[:, idx] + awq_scale_zeros[:, idx // group_size])
                    / awq_qscales[:, idx // group_size]
                )*(1-masks[:, idx]) 
            )

            tmp_intweight = masks[:, idx]*torch.round(
                    (awq_w[:, idx] + gptq_scale_zeros[:, idx // group_size])
                    / gptq_qscales[:, idx // group_size]
                )
            awq_intweight[-1] += tmp_intweight
            awq_intweight[-1] = awq_intweight[-1].to(torch.int)[:, None]
            #int_masks.append(masks[:, idx].to(torch.int)[:, None])
            # gptq_intweight.append(
            #     torch.round(
            #         (gptq_w[:, idx] + gptq_scale_zeros[:, idx // group_size])
            #         / gptq_qscales[:, idx // group_size]
            #     ).to(torch.int)[:, None]
            # )
        awq_intweight = torch.cat(awq_intweight, dim=1)
        #int_masks = torch.cat(int_masks, dim=1)
        #gptq_intweight = torch.cat(gptq_intweight, dim=1)
        # intweight = intweight.t().contiguous()
        awq_intweight = awq_intweight.to(dtype=torch.int32)
        #int_masks = int_masks.to(dtype=torch.int32)
        #gptq_intweight = gptq_intweight.to(dtype=torch.int32)
        #masks = masks.to(shape=awq_intweight.shape, dtype=torch.int32)
        #awq_intweight = awq_intweight * (1-masks) + gptq_intweight * masks
        awq_linear.awq_qweight = pack_intweight(
            awq_intweight.contiguous(), interleave=4, kstride=64
        )
        # awq_linear.masks = pack_intweight(
        #     int_masks.contiguous(), interleave=4, kstride=64
        # ).to(dtype=torch.uint8)
        awq_linear.masks = masks.to(dtype=torch.uint8)
        # awq_linear.gptq_qweight = pack_intweight(
        #     gptq_intweight.contiguous(), interleave=4, kstride=64
        # )

        awq_zeros = awq_zeros.to(dtype=torch.int32)
        gptq_zeros = gptq_zeros.to(dtype=torch.int32)
        awq_scaled_zeros = torch.zeros_like(awq_qscales)
        gptq_scaled_zeros = torch.zeros_like(gptq_qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        # 注意：使用 awq_scales.shape[1] 而不是 awq_qscales.shape[1]，因为 awq_zeros 的形状与 awq_scales 匹配
        awq_scaled_zeros[:, : awq_scales.shape[1]] = -(
            awq_qscales[:, : awq_scales.shape[1]] * (awq_zeros.to(torch.float32))
        ).to(torch.float16)
        gptq_scaled_zeros[:, : gptq_scales.shape[1]] = -(
            gptq_qscales[:, : gptq_scales.shape[1]] * (gptq_zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.awq_scaled_zeros = awq_scaled_zeros.transpose(1, 0).contiguous()
        awq_linear.gptq_scaled_zeros = gptq_scaled_zeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        
        # concatenated_qweight = torch.cat([self.awq_qweight, self.gptq_qweight], dim=-1)
        # concatenated_scales = torch.cat([self.awq_scales, self.gptq_scales], dim=0)
        # concatenated_scaled_zeros = torch.cat([self.awq_scaled_zeros, self.gptq_scaled_zeros], dim=0)
        # concatenated_input = torch.cat([inputs/self.current_awq_x_s, inputs], dim=-1)
        # print('--------------------------------')
        # print("infeature:", self.in_features)
        # print("outfeatures:", self.out_features)
        # print("group_size:", self.group_size)
        # print(f"inputs.shape: {inputs.shape}")
        # #print(f"concatenated_input.shape: {concatenated_input.shape}")
        # print(f"self.current_awq_x_s.shape: {self.current_awq_x_s.shape}")
        # print(f"self.awq_qweight.shape: {self.awq_qweight.shape}")
        # print(f"self.gptq_qweight.shape: {self.gptq_qweight.shape}")
        # #print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
        # print(f"self.awq_scales.shape: {self.awq_scales.shape}")
        # print(f"self.gptq_scales.shape: {self.gptq_scales.shape}")
        # #print(f"concatenated_scales.shape: {concatenated_scales.shape}")
        
        # print(f"self.awq_scaled_zeros.shape: {self.awq_scaled_zeros.shape}")
        # print(f"self.gptq_scaled_zeros.shape: {self.gptq_scaled_zeros.shape}")
        # print('--------------------------------')
        #print(f"concatenated_scaled_zeros.shape: {concatenated_scaled_zeros.shape}")

        #assert False
        if False: #inputs.numel() / inputs.shape[-1] < 8:
            # 按第一个维度拼接 awq_qweight 和 gptq_qweight
            # print(f"concatenated_input.shape: {concatenated_input.shape}")
            # concatenated_out = awq_inference_engine.gemv_forward_cuda_new(
            #     concatenated_input,
            #     concatenated_qweight,
            #     concatenated_scales,
            #     concatenated_scaled_zeros,
            #     concatenated_input.numel() // concatenated_input.shape[-1],
            #     self.out_features,
            #     self.in_features*2,
            #     self.group_size,
            # )
            # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # print(f"self.out_features: {self.out_features}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
            awq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs/self.current_awq_x_s,#.div_(self.current_awq_x_s),
                self.awq_qweight,
                self.awq_scales,
                self.awq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            gptq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.gptq_qweight,
                self.gptq_scales,
                self.gptq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            out = awq_out + gptq_out
        else:
            # 使用CUDA Streams并行执行两个gemm操作
            # 注意：PyTorch的CUDA操作默认就是异步的，串行代码也可能自动并行
            # 但如果需要显式并行，可以使用预创建的stream
            if False:#inputs.is_cuda and self.stream1 is not None and self.stream2 is not None:
                # 使用预创建的stream（避免每次forward都创建新stream的开销）
                # 预先计算AWQ的输入（避免在stream中计算）
                #print("here")
                awq_inputs = inputs / self.current_awq_x_s
                
                # # 在stream1中执行AWQ gemm
                with torch.cuda.stream(self.stream1):
                    awq_out = awq_inference_engine.gemm_forward_cuda_new(
                                awq_inputs,
                                self.awq_qweight,
                                self.awq_scales,
                                self.awq_scaled_zeros
                    )
                
                # 在stream2中执行GPTQ gemm（与stream1并行执行）
                with torch.cuda.stream(self.stream2):
                    gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                                inputs,
                                self.gptq_qweight,
                                self.gptq_scales,
                                self.gptq_scaled_zeros
                    )
                
                # 不需要显式synchronize，相加操作会自动等待两个stream完成
                # 这是PyTorch的默认行为，比显式synchronize更高效
                out = awq_out + gptq_out

            # concatenated_out = awq_inference_engine.gemm_forward_cuda_new(
                #         concatenated_input,
                #         concatenated_qweight,
                #         concatenated_scales,
                #         concatenated_scaled_zeros
                #     )

                # out = concatenated_out

                # assert (out==concatenated_out).all()

            else:
                # CPU模式或没有stream：串行执行
                # 注意：即使看起来是串行的，PyTorch的CUDA操作默认是异步的
                # GPU可能会自动并行执行（如果kernel支持且资源允许）
                #print("here")
                # mask 需要是 uint8 类型（Byte），形状和 awq_qweight 一样
                #masks = torch.zeros(self.awq_qweight.shape, dtype=torch.uint8, device=self.awq_qweight.device)
                awq_out = awq_inference_engine.gemm_forward_cuda_new_new(
                    inputs,
                    self.awq_qweight,
                    self.awq_scales,
                    self.awq_scaled_zeros,
                    self.gptq_scales,
                    self.gptq_scaled_zeros,
                    self.masks,
                    self.current_awq_x_s
                )
                # gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                #     inputs,
                #     self.gptq_qweight,
                #     self.gptq_scales,
                #     self.gptq_scaled_zeros
                # )
                out = awq_out #+ gptq_out

                # out = awq_inference_engine.gemm_forward_cuda_new(
                #     inputs/self.current_awq_x_s + inputs,
                #     self.awq_qweight+self.gptq_qweight,
                #     self.awq_scales+self.gptq_scales,
                #     self.awq_scaled_zeros+self.gptq_scaled_zeros
            # )
            # # # print(f"inputs.shape: {inputs.shape}")
            # # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # # print(f"self.out_features: {self.out_features}")
            # # # print(f"awq_out.shape: {awq_out.shape}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


class _WQ0124SharedInputScaleGroup:
    """Reuse one scaled activation across projections with an identical x-scale."""

    def __init__(self, scale):
        self.scale = scale
        self._input_key = None
        self._scaled_inputs = None

    def acquire(self, inputs, release):
        # Q/K/V and Gate/Up receive the same Tensor object in the HuggingFace
        # Llama forward.  _version protects against an in-place update between
        # projections without adding a CUDA synchronization to the hot path.
        input_key = (id(inputs), inputs._version)
        if self._input_key != input_key or self._scaled_inputs is None:
            self._scaled_inputs = inputs / self.scale
            self._input_key = input_key

        scaled_inputs = self._scaled_inputs
        if release:
            # The final projection keeps a local reference until its GEMM has
            # been enqueued.  Clearing here prevents the temporary from living
            # through attention/MLP and therefore keeps the peak unchanged.
            self._scaled_inputs = None
            self._input_key = None
        return scaled_inputs


def bind_wqlinear0124_shared_input_scale(modules):
    """Bind modules only when their resident input-scale tensors are identical."""
    if len(modules) < 2 or not all(isinstance(m, WQLinear0124) for m in modules):
        return False
    if any(module.prepare_input_domain_mode() for module in modules):
        # Same-domain GPTQ/RTN uses the original input directly and therefore
        # has no scaled activation temporary to share.
        return False

    reference = modules[0].current_awq_x_s
    if not all(
        m.current_awq_x_s.shape == reference.shape
        and m.current_awq_x_s.dtype == reference.dtype
        and m.current_awq_x_s.device == reference.device
        and torch.equal(m.current_awq_x_s, reference)
        for m in modules[1:]
    ):
        return False

    group = _WQ0124SharedInputScaleGroup(reference)
    for index, module in enumerate(modules):
        module._shared_input_scale_group = group
        module._shared_input_scale_release = index == len(modules) - 1
    return True


class WQLinear0124(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        self._mask_numel = out_features * in_features
        # Checkpoints keep the historical qweight-aligned mask layout. At
        # runtime it is replaced once with the equal-size CUDA-native b8 layout.
        self._mask_is_cuda_native = False
        # The checkpoint keeps four historical parameter buffers.  The fused
        # runtime replaces them with one [group, N, 4] buffer whose record is
        # {AWQ scale, AWQ zero, GPTQ scale, GPTQ zero}.  Each source is one
        # packed half2, so an indexed GPTQ lookup reads 4 rather than 8 bytes.
        # Total bytes stay equal to the four historical buffers.
        self._quant_params_are_cuda_paired = False
        # Set after model loading for Llama QKV and Gate/Up projections whose
        # current_awq_x_s buffers are verified bit-identical.
        self._shared_input_scale_group = None
        self._shared_input_scale_release = False
        # Set once after checkpoint loading.  An all-ones resident input scale
        # marks GPTQ/RTN, whose two candidates are both in the original domain.
        self._same_input_domain = None
        # True only for GPTQ dynamic-groups/act-order checkpoints whose
        # original K columns do not map to qparams by ``k // group_size``.
        self._uses_dynamic_gptq_groups = None
        # g_idx validation used to run GPU reductions from every linear
        # forward.  Runtime preparation is immutable after checkpoint load,
        # so cache it and keep dynamic group ids in their smallest form.
        self._gptq_g_idx_runtime_prepared = False
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit

        assert out_features % (self.interleave) == 0

        # self.register_buffer(
        #     "test_awq_qweight",
        #     torch.zeros(
        #         (
        #             out_features ,
        #             in_features
        #         ),
        #         dtype=torch.float16,
        #         device=dev,
        #     ),
        # )
        # self.register_buffer(
        #     "test_gptq_qweight",
        #     torch.zeros(
        #         (
        #             out_features ,
        #             in_features
        #         ),
        #         dtype=torch.float16,
        #         device=dev,
        #     ),
        # )
        self.register_buffer(
            "awq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "awq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "full_qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_scaled_zeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "cuda_param_quads",
            None,
            persistent=False,
        )
        self.register_buffer(
            "current_awq_x_s",
            torch.zeros(
                (
                    1, in_features
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "gptq_g_idx",
            torch.arange(in_features, dtype=torch.int32, device=dev)
            // self.group_size,
        )
        # 1-bit packed path mask in qweight layout: 8 path bits per uint8.
        mask_packed_size = (self._mask_numel + 7) // 8
        self.register_buffer(
            "full_mask_packed",
            torch.zeros(mask_packed_size, dtype=torch.uint8, device=dev),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None
        
        # 创建CUDA streams用于并行执行（如果设备是CUDA）
        # 注意：只在CUDA设备上创建，避免每次forward都创建新stream的开销
        if False:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        else:
            self.stream1 = None
            self.stream2 = None

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        group_size,
        init_only=False,
        awq_w=None,
        awq_scales=None,
        awq_zeros=None,
        gptq_w=None,
        gptq_scales=None,
        gptq_zeros=None,
        current_awq_x_s=None,
        full_w=None,
        full_mask=None,
        gptq_g_idx=None,
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        current_device = linear.weight.device
        gptq_w = gptq_w.to(device=current_device, dtype=awq_w.dtype)
        gptq_scales = gptq_scales.to(device=current_device, dtype=awq_scales.dtype)
        gptq_zeros = gptq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        awq_w = awq_w.to(device=current_device, dtype=awq_w.dtype)
        awq_scales = awq_scales.to(device=current_device, dtype=awq_scales.dtype)
        awq_zeros = awq_zeros.to(device=current_device, dtype=awq_zeros.dtype)
        if full_w is not None:
            full_w = full_w.to(device=current_device, dtype=awq_w.dtype)
        if gptq_g_idx is None:
            gptq_g_idx = torch.arange(
                linear.in_features, device=current_device, dtype=torch.int32
            ) // awq_linear.group_size
        else:
            gptq_g_idx = torch.as_tensor(
                gptq_g_idx, device=current_device, dtype=torch.int32
            ).reshape(-1)
            if gptq_g_idx.numel() != linear.in_features:
                raise ValueError(
                    f"GPTQ g_idx has {gptq_g_idx.numel()} elements, "
                    f"expected {linear.in_features}"
                )
            num_groups = linear.in_features // awq_linear.group_size
            if (gptq_g_idx < 0).any() or (gptq_g_idx >= num_groups).any():
                raise ValueError("GPTQ g_idx contains an invalid group id")
        awq_linear.gptq_g_idx = gptq_g_idx.contiguous()
        static_g_idx = torch.arange(
            linear.in_features, device=current_device, dtype=torch.int32
        ) // awq_linear.group_size
        awq_linear._uses_dynamic_gptq_groups = not torch.equal(
            awq_linear.gptq_g_idx, static_g_idx
        )
        assert awq_scales is not None and awq_zeros is not None and gptq_scales is not None and gptq_zeros is not None
        awq_scale_zeros = awq_zeros * awq_scales
        gptq_scale_zeros = gptq_zeros * gptq_scales
        # awq_linear.test_awq_qweight = awq_w.clone()
        # awq_linear.test_gptq_qweight = gptq_w.clone()
        # awq_linear.test_awq_weight = awq_w.clone()
        pack_num = 32 // awq_linear.w_bit
        awq_qscales = torch.zeros(
            (
                awq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=awq_scales.device,
        )
        awq_qscales[:, : awq_scales.shape[1]] = awq_scales

        gptq_qscales = torch.zeros(
            (
                gptq_scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=gptq_scales.device,
        )
        gptq_qscales[:, : gptq_scales.shape[1]] = gptq_scales
        awq_linear.current_awq_x_s = current_awq_x_s.view(1, -1)
        # awq_linear.scales = scales.clone().half()
        awq_linear.awq_scales = awq_qscales.transpose(1, 0).contiguous()
        awq_linear.gptq_scales = gptq_qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        awq_zeros = awq_zeros.to(dtype=torch.int32)
        gptq_zeros = gptq_zeros.to(dtype=torch.int32)
        awq_scaled_zeros = torch.zeros_like(awq_qscales)
        gptq_scaled_zeros = torch.zeros_like(gptq_qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        # 注意：使用 awq_scales.shape[1] 而不是 awq_qscales.shape[1]，因为 awq_zeros 的形状与 awq_scales 匹配
        awq_scaled_zeros[:, : awq_scales.shape[1]] = -(
            awq_qscales[:, : awq_scales.shape[1]] * (awq_zeros.to(torch.float32))
        ).to(torch.float16)
        gptq_scaled_zeros[:, : gptq_scales.shape[1]] = -(
            gptq_qscales[:, : gptq_scales.shape[1]] * (gptq_zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.awq_scaled_zeros = awq_scaled_zeros.transpose(1, 0).contiguous()
        awq_linear.gptq_scaled_zeros = gptq_scaled_zeros.transpose(1, 0).contiguous()

        if full_w is not None and full_mask is not None:
            assert full_w.shape == (linear.out_features, linear.in_features), (
                f"full_w shape {full_w.shape} != "
                f"({linear.out_features}, {linear.in_features})"
            )
            assert full_mask.shape == (linear.out_features, linear.in_features), (
                f"full_mask shape {full_mask.shape} != "
                f"({linear.out_features}, {linear.in_features})"
            )
            full_mask_dev = full_mask.to(device=current_device)
            awq_linear.full_mask_packed = pack_binary_mask_like_qweight(
                full_mask_dev,
                interleave=awq_linear.interleave,
                kstride=64,
            ).reshape(-1)
            awq_linear._mask_is_cuda_native = False
            full_intweight = quantize_intweight_with_mask(
                full_w,
                full_mask_dev,
                awq_scale_zeros,
                gptq_scale_zeros,
                awq_qscales,
                gptq_qscales,
                group_size,
                w_bit,
                gptq_g_idx=awq_linear.gptq_g_idx,
            )
            awq_linear.full_qweight = pack_intweight(
                full_intweight.contiguous(), interleave=4, kstride=64
            )
            awq_linear._dual_packed_cache = None

            # Debug: full_qweight + full_mask_packed -> awq/gptq packed roundtrip
            _debug_verify_full_to_dual_qweights(
                awq_linear,
                full_intweight,
                full_mask_dev,
                awq_w,
                gptq_w,
                awq_scale_zeros,
                gptq_scale_zeros,
                awq_qscales,
                gptq_qscales,
                group_size,
                w_bit,
                gptq_g_idx=awq_linear.gptq_g_idx,
            )

        return awq_linear

    @torch.no_grad()
    def prepare_cuda_native_mask(self):
        """Convert the resident mask once; total persistent mask bytes stay unchanged."""
        if not self._mask_is_cuda_native:
            self.full_mask_packed = prepack_binary_mask_for_cuda(
                self.full_mask_packed, self.out_features, self.in_features
            )
            self._mask_is_cuda_native = True
            self._dual_packed_cache = None
        return self.full_mask_packed

    @torch.no_grad()
    def prepare_input_domain_mode(self):
        """Resolve the checkpoint's input domain once, outside the hot path."""
        if self._same_input_domain is None:
            self._same_input_domain = bool(
                torch.all(self.current_awq_x_s == 1).item()
            )
        return self._same_input_domain

    @torch.no_grad()
    def prepare_gptq_group_index(self):
        """Validate g_idx once and compact dynamic ids for the CUDA hot path."""
        index = self.gptq_g_idx
        target_device = self.full_qweight.device
        if (
            self._gptq_g_idx_runtime_prepared
            and index.device == target_device
            and index.numel() == self.in_features
        ):
            return index

        if index.device.type == "meta" or index.numel() != self.in_features:
            index = torch.arange(
                self.in_features, device=target_device, dtype=torch.int32
            ) // self.group_size
        elif index.device != target_device:
            index = index.to(device=target_device)

        # Checkpoints remain int32.  A previously prepared module may already
        # contain uint8, so normalize only for the one-time validation below.
        index_i32 = index.to(dtype=torch.int32).contiguous()

        num_groups = self.in_features // self.group_size
        if (
            (index_i32 < 0).any()
            or (index_i32 >= num_groups).any()
        ):
            raise ValueError("Loaded GPTQ g_idx contains an invalid group id")
        static_g_idx = torch.arange(
            self.in_features,
            device=index_i32.device,
            dtype=torch.int32,
        ) // self.group_size
        self._uses_dynamic_gptq_groups = not torch.equal(
            index_i32, static_g_idx
        )
        if self._uses_dynamic_gptq_groups:
            if num_groups > 256:
                raise ValueError(
                    "Fused dynamic-group CUDA supports at most 256 groups"
                )
            # Llama-7B has at most 86 G128 groups.  uint8 cuts this resident
            # lookup and its global-memory traffic to one quarter of int32.
            index = index_i32.to(dtype=torch.uint8)
        else:
            index = index_i32
        self.gptq_g_idx = index.contiguous()
        self._gptq_g_idx_runtime_prepared = True
        return self.gptq_g_idx

    @torch.no_grad()
    def prepare_cuda_paired_quant_params(self):
        """Interleave scale/zero pairs into one CUDA record without extra bytes."""
        if not self._quant_params_are_cuda_paired:
            self.cuda_param_quads = torch.stack(
                (
                    self.awq_scales,
                    self.awq_scaled_zeros,
                    self.gptq_scales,
                    self.gptq_scaled_zeros,
                ),
                dim=-1,
            ).contiguous()
            empty = self.cuda_param_quads.new_empty(0)
            self.awq_scales = empty
            self.gptq_scales = empty

            # Release the four old storages.  Their state-dict keys remain
            # registered and are reconstructed by _save_to_state_dict().
            self.awq_scaled_zeros = empty
            self.gptq_scaled_zeros = empty
            self._quant_params_are_cuda_paired = True
            self._dual_packed_cache = None
        return self.cuda_param_quads

    @torch.no_grad()
    def restore_legacy_quant_params(self):
        """Restore four contiguous buffers if a paired module switches to legacy mode."""
        if self._quant_params_are_cuda_paired:
            self.awq_scales = self.cuda_param_quads[..., 0].contiguous()
            self.gptq_scales = self.cuda_param_quads[..., 1].contiguous()
            self.awq_scaled_zeros = self.cuda_param_quads[..., 2].contiguous()
            self.gptq_scaled_zeros = self.cuda_param_quads[..., 3].contiguous()
            self.cuda_param_quads = self.cuda_param_quads.new_empty(0)
            self._quant_params_are_cuda_paired = False
            self._dual_packed_cache = None

    def _legacy_quant_param_views(self):
        if self._quant_params_are_cuda_paired:
            return (
                self.cuda_param_quads[..., 0],
                self.cuda_param_quads[..., 1],
                self.cuda_param_quads[..., 2],
                self.cuda_param_quads[..., 3],
            )
        return (
            self.awq_scales,
            self.awq_scaled_zeros,
            self.gptq_scales,
            self.gptq_scaled_zeros,
        )

    def _mask_in_qweight_layout(self):
        if self._mask_is_cuda_native:
            return unpack_cuda_native_mask_to_qweight_layout(
                self.full_mask_packed, self.out_features, self.in_features
            )
        return self.full_mask_packed

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Always serialize backward-compatible mask and quant-parameter layouts."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.gptq_g_idx.dtype != torch.int32:
            legacy_g_idx = self.gptq_g_idx.to(dtype=torch.int32)
            destination[prefix + "gptq_g_idx"] = (
                legacy_g_idx if keep_vars else legacy_g_idx.detach()
            )
        if self._mask_is_cuda_native:
            legacy_mask = self._mask_in_qweight_layout()
            destination[prefix + "full_mask_packed"] = (
                legacy_mask if keep_vars else legacy_mask.detach()
            )
        if self._quant_params_are_cuda_paired:
            names = (
                "awq_scales",
                "awq_scaled_zeros",
                "gptq_scales",
                "gptq_scaled_zeros",
            )
            for name, value in zip(names, self._legacy_quant_param_views()):
                destination[prefix + name] = value if keep_vars else value.detach()

    def get_full_mask(self):
        """Restore the resident packed mask to bool [out_features, in_features]."""
        return unpack_binary_mask_like_qweight(
            self._mask_in_qweight_layout(),
            self.out_features,
            self.in_features,
            self.interleave,
            64,
        )

    def _get_dual_packed_qweights(self):
        """Rebuild awq/gptq packed qweights from full_qweight + full_mask_packed."""
        cache = getattr(self, "_dual_packed_cache", None)
        if cache is not None:
            return cache
        row_mask_packed = pack_binary_mask(self.get_full_mask())
        awq_scales, awq_zeros, gptq_scales, gptq_zeros = (
            self._legacy_quant_param_views()
        )
        awq_qweight, gptq_qweight = split_dual_packed_qweights(
            self.full_qweight,
            row_mask_packed,
            self.out_features,
            self.in_features,
            interleave=self.interleave,
            kstride=64,
            w_bit=self.w_bit,
            group_size=self.group_size,
            awq_scales=awq_scales,
            awq_scaled_zeros=awq_zeros,
            gptq_scales=gptq_scales,
            gptq_scaled_zeros=gptq_zeros,
            gptq_g_idx=self.gptq_g_idx,
        )
        self._dual_packed_cache = (awq_qweight, gptq_qweight)
        return self._dual_packed_cache

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        same_input_domain = self.prepare_input_domain_mode()
        if same_input_domain:
            scaled_inputs = inputs
        elif self._shared_input_scale_group is None:
            scaled_inputs = inputs / self.current_awq_x_s
        else:
            scaled_inputs = self._shared_input_scale_group.acquire(
                inputs, self._shared_input_scale_release
            )
        # zch0721's AWQ and GPTQ candidates use different input-scaling
        # domains, so the historical same-input dual-GEMM fallback is not an
        # equivalent implementation.  Keep the exact fused path mandatory.
        use_legacy_split = False
        if use_legacy_split:
            self.restore_legacy_quant_params()
            awq_qweight, gptq_qweight = self._get_dual_packed_qweights()
        
        # concatenated_qweight = torch.cat([self.awq_qweight, self.gptq_qweight], dim=-1)
        # concatenated_scales = torch.cat([self.awq_scales, self.gptq_scales], dim=0)
        # concatenated_scaled_zeros = torch.cat([self.awq_scaled_zeros, self.gptq_scaled_zeros], dim=0)
        # concatenated_input = torch.cat([inputs/self.current_awq_x_s, inputs], dim=-1)
        # print('--------------------------------')
        # print("infeature:", self.in_features)
        # print("outfeatures:", self.out_features)
        # print("group_size:", self.group_size)
        # print(f"inputs.shape: {inputs.shape}")
        # #print(f"concatenated_input.shape: {concatenated_input.shape}")
        # print(f"self.current_awq_x_s.shape: {self.current_awq_x_s.shape}")
        # print(f"self.awq_qweight.shape: {self.awq_qweight.shape}")
        # print(f"self.gptq_qweight.shape: {self.gptq_qweight.shape}")
        # #print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
        # print(f"self.awq_scales.shape: {self.awq_scales.shape}")
        # print(f"self.gptq_scales.shape: {self.gptq_scales.shape}")
        # #print(f"concatenated_scales.shape: {concatenated_scales.shape}")
        
        # print(f"self.awq_scaled_zeros.shape: {self.awq_scaled_zeros.shape}")
        # print(f"self.gptq_scaled_zeros.shape: {self.gptq_scaled_zeros.shape}")
        # print('--------------------------------')
        #print(f"concatenated_scaled_zeros.shape: {concatenated_scaled_zeros.shape}")

        #assert False
        if False:#inputs.numel() / inputs.shape[-1] < 8:
            # 按第一个维度拼接 awq_qweight 和 gptq_qweight
            # print(f"concatenated_input.shape: {concatenated_input.shape}")
            # concatenated_out = awq_inference_engine.gemv_forward_cuda_new(
            #     concatenated_input,
            #     concatenated_qweight,
            #     concatenated_scales,
            #     concatenated_scaled_zeros,
            #     concatenated_input.numel() // concatenated_input.shape[-1],
            #     self.out_features,
            #     self.in_features*2,
            #     self.group_size,
            # )
            # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # print(f"self.out_features: {self.out_features}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
            awq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs/self.current_awq_x_s,#.div_(self.current_awq_x_s),
                awq_qweight,
                self.awq_scales,
                self.awq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            gptq_out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                gptq_qweight,
                self.gptq_scales,
                self.gptq_scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )

            out = awq_out + gptq_out
        else:
            # 使用CUDA Streams并行执行两个gemm操作
            # 注意：PyTorch的CUDA操作默认就是异步的，串行代码也可能自动并行
            # 但如果需要显式并行，可以使用预创建的stream
            if False:#inputs.is_cuda and self.stream1 is not None and self.stream2 is not None:
                # 使用预创建的stream（避免每次forward都创建新stream的开销）
                # 预先计算AWQ的输入（避免在stream中计算）
                #print("here")
                awq_inputs = inputs / self.current_awq_x_s
                
                # # 在stream1中执行AWQ gemm
                with torch.cuda.stream(self.stream1):
                    awq_out = awq_inference_engine.gemm_forward_cuda_new(
                        awq_inputs,
                        awq_qweight,
                        self.awq_scales,
                        self.awq_scaled_zeros
                    )
                
                # 在stream2中执行GPTQ gemm（与stream1并行执行）
                with torch.cuda.stream(self.stream2):
                    gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                        inputs,
                        gptq_qweight,
                        self.gptq_scales,
                        self.gptq_scaled_zeros
                    )
                
                # 不需要显式synchronize，相加操作会自动等待两个stream完成
                # 这是PyTorch的默认行为，比显式synchronize更高效
                out = awq_out + gptq_out

                # concatenated_out = awq_inference_engine.gemm_forward_cuda_new(
                #         concatenated_input,
                #         concatenated_qweight,
                #         concatenated_scales,
                #         concatenated_scaled_zeros
                #     )

                # out = concatenated_out

                # assert (out==concatenated_out).all()

            else:
                # CPU模式或没有stream：串行执行
                # 注意：即使看起来是串行的，PyTorch的CUDA操作默认是异步的
                # GPU可能会自动并行执行（如果kernel支持且资源允许）
                #print("here")
                # scaled_inputs = inputs/self.current_awq_x_s
                # awq_out = awq_inference_engine.gemm_forward_cuda_new(
                #     scaled_inputs,
                #     awq_qweight,
                #     self.awq_scales,
                #     self.awq_scaled_zeros
                # )
                # gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                #     scaled_inputs,
                #     gptq_qweight,
                #     self.gptq_scales,
                #     self.gptq_scaled_zeros
                # )
                # out = awq_out + gptq_out
                # Matrix multiplication: inputs @ test_awq_qweight.T
                # inputs shape: [batch_size, in_features]
                # test_awq_qweight shape: [out_features, in_features]
                # result shape: [batch_size, out_features]
                # test_out = torch.matmul(inputs/self.current_awq_x_s, self.test_awq_qweight.t())
                # test_out2 = torch.matmul(inputs/self.current_awq_x_s, self.test_gptq_qweight.t())
                
                # out = test_out + test_out2

                if use_legacy_split:
                    awq_out = awq_inference_engine.gemm_forward_cuda_new(
                        scaled_inputs,
                        awq_qweight,
                        self.awq_scales,
                        self.awq_scaled_zeros,
                    )
                    gptq_out = awq_inference_engine.gemm_forward_cuda_new(
                        inputs,
                        gptq_qweight,
                        self.gptq_scales,
                        self.gptq_scaled_zeros,
                    )
                    out = awq_out + gptq_out
                else:
                    native_mask = self.prepare_cuda_native_mask()
                    param_quads = self.prepare_cuda_paired_quant_params()
                    self.prepare_gptq_group_index()
                    if same_input_domain and self._uses_dynamic_gptq_groups:
                        out = awq_inference_engine.gemm_forward_cuda_masked_same_domain_gidx(
                            inputs,
                            self.full_qweight,
                            native_mask,
                            param_quads,
                            self.gptq_g_idx,
                            self.group_size,
                        )
                    elif same_input_domain:
                        out = awq_inference_engine.gemm_forward_cuda_masked_same_domain(
                            inputs,
                            self.full_qweight,
                            native_mask,
                            param_quads,
                            self.group_size,
                        )
                    elif self._uses_dynamic_gptq_groups:
                        out = awq_inference_engine.gemm_forward_cuda_masked_gidx(
                            scaled_inputs,
                            self.full_qweight,
                            native_mask,
                            param_quads,
                            self.current_awq_x_s,
                            self.gptq_g_idx,
                            self.group_size,
                        )
                    else:
                        out = awq_inference_engine.gemm_forward_cuda_masked(
                            scaled_inputs,
                            self.full_qweight,
                            native_mask,
                            param_quads,
                            self.current_awq_x_s,
                            self.group_size,
                        )

                # out = awq_inference_engine.gemm_forward_cuda_new(
                #     inputs/self.current_awq_x_s + inputs,
                #     awq_qweight+gptq_qweight,
                #     self.awq_scales+self.gptq_scales,
                #     self.awq_scaled_zeros+self.gptq_scaled_zeros
                # )
            # # # print(f"inputs.shape: {inputs.shape}")
            # # # print(f"awq_qweight.shape: {self.awq_qweight.shape}")
            # # # print(f"gptq_qweight.shape: {self.gptq_qweight.shape}")
            # # # print(f"concatenated_qweight.shape: {concatenated_qweight.shape}")
            # # # print(f"concatenated_out.shape: {concatenated_out.shape}")
            # # # print(f"self.out_features: {self.out_features}")
            # # # print(f"awq_out.shape: {awq_out.shape}")
            # out = concatenated_out[:, :, :self.out_features]/self.current_awq_x_s + concatenated_out[:, :, self.out_features:]/self.current_awq_x_s
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
