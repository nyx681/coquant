import gc
import torch
import torch.nn as nn
import numpy as np

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from .qmodule import ScaledActivation
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name

__all__ = ["auto_scale_block", "apply_scale"]

np.random.seed(42)


def _is_llama_like_decoder_layer(module):
    attn = getattr(module, "self_attn", None)
    mlp = getattr(module, "mlp", None)
    return (
        attn is not None
        and mlp is not None
        and all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))
        and hasattr(module, "input_layernorm")
        and hasattr(module, "post_attention_layernorm")
    )


def _is_norm_layer(module):
    class_name = module.__class__.__name__.lower()
    return hasattr(module, "weight") and ("norm" in class_name)

@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block722(module, module_kwargs, w_bit, q_config, input_feat):
    from .quantizer import pseudo_quantize_tensor722

    # firstly, get the weight quantize function
    if w_bit is not None:

        def w_quantize_func722(p,th_t,search_flag=False):
            return pseudo_quantize_tensor722(
                p,
                p,
                p,
                n_bit=w_bit,
                th_t=th_t,
                **q_config,
                search_flag=search_flag
            ).detach()

    else:

        def w_quantize_func(p):
            return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale722(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        best_th_t = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        search_flag = True
        scales = x_max.pow(0).clamp(min=1e-4).view(-1)
        best_scales = scales
        best_th_t = 0
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            # tmp_loss_list = []
            for th_t in [1]:
            # for th_t in range(n_grid+1):
            #     th_t = th_t * 1 / n_grid #0,.0.05,...,1.0
                for fc in linears2scale:
                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = w_quantize_func722(fc.weight.data,th_t,search_flag=search_flag) / (scales.view(1, -1))
                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = ((org_out - out).float().pow(2).mean().item())  # float prevents overflow
                history.append(loss)
                #tmp_loss_list.append(loss)
                block.load_state_dict(org_sd)
            
                # tmp_loss_list = np.array(tmp_loss_list)
                # weights = np.exp(-tmp_loss_list)
                # probs = weights/weights.sum()
                # th_t = np.argmin(tmp_loss_list)
                # loss = tmp_loss_list[th_t]
            
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales
                    best_th_t = th_t
                    print(f"Use 722 best_th_t: {best_th_t}   ",f"best_ratio: {best_ratio}   ",f"best_error: {best_error}")  
                    
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        #best_scales = best_scales.view(-1
        #best_scales = torch.ones_like(best_scales.view(-1))

        assert torch.isnan(best_scales).sum() == 0, best_scales

        
        return best_scales.detach(), best_th_t

    def _auto_get_scale722(prev_op, layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales, th_t = _search_module_scale722(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        th_t = th_t
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
            th_t
        )

    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, LlamaDecoderLayer) or _is_llama_like_decoder_layer(module):
        # attention input
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale722(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        else:
        # fc1scales_list.append(
            scales_list.append(
                (
                get_op_name(module, module.self_attn.v_proj),
                tuple([get_op_name(module, m) for m in [module.self_attn.o_proj]]),
                None,
                0
            )
            )
            
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale722(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # attn out
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale722(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale722(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale722(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale722(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale722(module, scales_list, input_feat_dict=None, verbose=False):
    th_dict = {}
    i = 0
    for prev_op_name, layer_names, scales, th_t in scales_list:
        for name in layer_names:
            th_dict[i] = th_t 
            i += 1
        #i += 1


        # continue

        if scales is None:
            continue
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        # 输出scales的统计信息，便于分析（仅在verbose时打印）
        if verbose:
            with torch.no_grad():
                stats_device = scales.device
                stats_tensor = scales.float().flatten()
                scale_min = stats_tensor.min().item()
                scale_max = stats_tensor.max().item()
                scale_mean = stats_tensor.mean().item()
                scale_median = stats_tensor.median().item()
                print(
                    f"[apply_scale] {layer_names} | "
                    f"shape={tuple(scales.shape)} device={stats_device} | "
                    f"min={scale_min:.4f} max={scale_max:.4f} "
                    f"mean={scale_mean:.4f} median={scale_median:.4f}"
                )

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or _is_norm_layer(prev_op):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
    if verbose:
        print(f"th_dict: {len(th_dict.keys())}", '    i=',i)
        print(f"th_dict: {th_dict}")

    return th_dict