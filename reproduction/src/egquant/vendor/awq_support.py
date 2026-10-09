import torch
import torch.nn as nn
from ..baselines import awq_reference

def get_op_by_name(module, op_name):
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f'Cannot find op {op_name} in module {module}')

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f'Cannot find op {op} in module {module}')

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

class ScaledActivation(nn.Module):

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

def pseudo_quantize_tensor722(o_w, current_awq_x_s=None, gptq_w=None, n_bit=4, th_t=1, zero_point=True, q_group_size=128, search_flag=False):
    if not zero_point or q_group_size != 128:
        raise ValueError('Main-table AWQ uses asymmetric G128')
    if current_awq_x_s is None or current_awq_x_s.numel() != o_w.shape[-1]:
        current_awq_x_s = torch.ones(o_w.shape[-1], device=o_w.device, dtype=o_w.dtype)
    return awq_reference(o_w, current_awq_x_s, n_bit)
