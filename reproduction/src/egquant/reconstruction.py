import numpy as np
import torch

@torch.no_grad()
def reconstruct_into(destination, original, centers, scales, bits, rows_per_batch=4):
    rows, cols = original.shape
    groups = (cols + 2047) // 2048
    if cols % 128 or centers.shape != (rows, groups, 2 ** bits) or scales.shape != (rows, groups, 16):
        raise ValueError('Codebook/scales shape mismatch')
    if centers.dtype != np.float32 or scales.dtype != np.float32:
        raise ValueError('Original FP32 parameters are required to freeze assignments')
    c16, t16 = (centers.astype(np.float16), scales.astype(np.float16))
    if not np.isfinite(c16).all() or not np.isfinite(t16).all():
        raise ValueError('FP16 parameter overflow')
    device = destination.device
    for start in range(0, rows, rows_per_batch):
        x = original[start:start + rows_per_batch].to(device=device, dtype=torch.float64)
        x = torch.nn.functional.pad(x, (0, groups * 2048 - cols)).reshape(-1, groups, 16, 128)
        c = torch.as_tensor(centers[start:start + rows_per_batch], device=device, dtype=torch.float64)
        t = torch.as_tensor(scales[start:start + rows_per_batch], device=device, dtype=torch.float64)
        levels = t[..., None] * c[:, :, None, :]
        indices = (x[..., None] - levels[:, :, :, None, :]).abs().argmin(-1)
        c = torch.as_tensor(c16[start:start + rows_per_batch], device=device, dtype=torch.float64)
        t = torch.as_tensor(t16[start:start + rows_per_batch], device=device, dtype=torch.float64)
        new_levels = t[..., None] * c[:, :, None, :]
        y = torch.gather(new_levels, 3, indices).reshape(-1, groups * 2048)[:, :cols]
        destination[start:start + rows_per_batch].copy_(y.to(destination.dtype))
    return dict(weights=rows * cols, parameter_bits=16 * (c16.size + rows * (cols // 128)))
