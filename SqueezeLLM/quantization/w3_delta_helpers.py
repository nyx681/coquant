"""W3-G128 center fitting and reconstruction, isolated from active W4 jobs."""
import numpy as np
import torch
from nuq import kmeans_fit
from group_center_selection import assign

def quantize_row(task):
    w, importance = task
    centers = np.empty((len(w)//128, 8), np.float32)
    costs = np.zeros(2, np.float64)
    for g in range(len(centers)):
        sl = slice(g*128,(g+1)*128)
        centers[g], _ = kmeans_fit((w[sl,None], importance[sl], 8))
        _, weighted, ordinary = assign(w[sl], importance[sl], centers[g])
        costs += weighted, ordinary
    return centers, costs

def fill_weight(weight, original, centers):
    """Reconstruct nearest-original labels in float64, matching assign()."""
    if centers.shape != (original.shape[0], original.shape[1]//128, 8):
        raise ValueError('Center shape mismatch')
    with torch.no_grad():
        for start in range(0, len(original), 32):
            stop = min(start+32, len(original))
            x = original[start:stop].reshape(stop-start, -1, 128).astype(np.float64)
            c = centers[start:stop]
            labels = np.abs(x[..., None]-c.astype(np.float64)[:, :, None, :]).argmin(-1)
            q = np.take_along_axis(c, labels, axis=2).reshape(stop-start, -1)
            weight[start:stop].copy_(torch.from_numpy(q))


