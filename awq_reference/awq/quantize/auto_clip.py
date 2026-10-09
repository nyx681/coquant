import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor722
import gc

__all__ = ["auto_clip_block"]


# weight quantization
@torch.no_grad()
def auto_clip_layer722(
    w, input_feat, n_bit, q_config, th_t, n_grid=20, max_shrink=0.5, n_sample_token=512, search_flag=False
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor722(cur_w, th_t=th_t, n_bit=n_bit, **q_config, search_flag=False)#search_flag)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block722(module, w_bit, q_config, input_feat, scales_list, search_flag=False):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    # 假设 scales_list 里每个元素是 {"name": ..., "th_t": ...}
    scales_dict = {}
    for _, name_tuple,_, th_t in scales_list:
        for name in name_tuple:
            scales_dict[name] = th_t
    print(f"scales_dict_len: {len(scales_dict.keys())}", '    scales_dict:',scales_dict)
    clip_list = []
    
    print("available named_linears:", list(named_linears.keys()))
    print("available scales_dict keys:", list(scales_dict.keys()))
    """
    available named_linears: ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    available scales_dict keys: ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']"""

    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
        # if any([_ in name for _ in ["q_", "k_", "o_","query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        print("apply clip using th_t")
                
        max_val = auto_clip_layer722(named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config, th_t=scales_dict[name], search_flag=search_flag)
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    print("-----Apply Clip-----")
    from ..utils.module import get_op_by_name

    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
