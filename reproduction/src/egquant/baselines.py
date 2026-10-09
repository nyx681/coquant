import re
import torch

def awq_scales(cache, module_map):
    if not isinstance(cache.get('scale'), list):
        raise ValueError('Expected an AWQ scale-search cache')
    suffixes = {v.rsplit('.', 1)[-1]: k for k, v in module_map.items()}
    result = {}
    for item in cache['scale']:
        _, names, scale = item[:3]
        if scale is None:
            continue
        for name in names:
            match = re.search('(?:^|\\.)layers\\.(\\d+)\\.', name)
            short = suffixes.get(name.rsplit('.', 1)[-1])
            if match and short:
                key = (int(match.group(1)), short)
                value = torch.as_tensor(scale).float().cpu()
                if key in result and (not torch.equal(result[key], value)):
                    raise ValueError('Conflicting AWQ scale records')
                result[key] = value
    return result

def awq_reference(weight, channel_scale, bits):
    scale_x = channel_scale.view(1, -1).to(device=weight.device, dtype=weight.dtype)
    if scale_x.shape[1] != weight.shape[1] or not torch.isfinite(scale_x).all() or (scale_x <= 0).any():
        raise ValueError('Invalid AWQ channel scales')
    q = (weight * scale_x).reshape(-1, 128)
    maximum, minimum = (q.amax(dim=1, keepdim=True), q.amin(dim=1, keepdim=True))
    max_int = 2 ** bits - 1
    scales = (maximum - minimum).clamp(min=1e-05) / max_int
    zeros = (-torch.round(minimum / scales)).clamp_(0, max_int)
    y = (torch.clamp(torch.round(q / scales) + zeros, 0, max_int) - zeros) * scales
    return y.reshape(weight.shape) / scale_x

def gptq_reference(record, original, bits):
    if not isinstance(record, dict) or not torch.is_tensor(record.get('weight')):
        raise ValueError('Expected a GPTQ export with dequantized weight')
    if int(record.get('maxq', -1)) != 2 ** bits - 1:
        raise ValueError('GPTQ maxq does not match the requested bit width')
    scale = record.get('scale')
    if scale is None or scale.shape != (original.shape[0], original.shape[1] // 128):
        raise ValueError('GPTQ scale shape does not match G128')
    if int(record.get('group_size', 128)) != 128:
        raise ValueError('GPTQ grouping does not match G128')
    weight = record['weight'].detach().cpu().contiguous()
    if weight.shape != original.shape or not torch.isfinite(weight).all():
        raise ValueError('Invalid GPTQ weight export')
    return weight
