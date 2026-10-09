import torch

def validate_record(record, shape):
    rows, cols = shape
    if cols % 128 or record.get('group_size', 128) != 128 or int(record.get('maxq', -1)) != 7:
        raise ValueError('Requires W3-G128 with full 128-element groups')
    expected = torch.arange(cols) // 128
    if 'g_idx' in record:
        if not torch.equal(record['g_idx'].cpu().long(), expected):
            raise ValueError('Non-contiguous GPTQ groups: explicit remapping is required')
    elif 'static_groups' in record and not record['static_groups']:
        raise ValueError('Missing verified static group mapping')
    # Legacy records without mapping metadata are accepted only after the full
    # contiguous-G128 reconstruction check below; shape alone is insufficient.
    scale, zero, native = (record[k].cpu() for k in ('scale', 'zero', 'weight'))
    if scale.shape != (rows, cols//128) or zero.shape != scale.shape or native.shape != shape:
        raise ValueError('GPTQ parameter shapes do not match original weights')
    if not all(torch.isfinite(t).all() for t in (scale, zero, native)) or (scale <= 0).any():
        raise ValueError('Invalid GPTQ parameter values')
    max_error = 0.
    for start in range(0, rows, 64):
        w = native[start:start+64].float().reshape(-1, cols//128, 128)
        s, z = scale[start:start+64, :, None], zero[start:start+64, :, None]
        codes = torch.round(w/s + z).clamp(0, 7)
        reconstructed = (s*(codes-z)).to(native.dtype).float()
        # Allow one dtype ULP for historical export arithmetic rounding.
        tolerance = torch.finfo(native.dtype).eps * torch.maximum(w.abs(), reconstructed.abs()) + torch.finfo(native.dtype).tiny
        error = (w-reconstructed).abs()
        if (error > tolerance).any():
            raise ValueError('GPTQ scales/zeros do not reconstruct native weights in original column order')
        max_error = max(max_error, error.max().item())
    return max_error

