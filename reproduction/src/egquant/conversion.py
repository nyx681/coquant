import numpy as np
from .kernels import conversion3, conversion4

def importance(original, reference):
    w = np.asarray(original, np.float32)
    q = np.asarray(reference, np.float32)
    if w.shape != q.shape or not np.isfinite(w).all() or (not np.isfinite(q).all()):
        raise ValueError('Invalid original/reference weights')
    eps = np.finfo(np.float32).eps
    return (0.5 + 0.5 * (np.abs(w) + eps) / (np.abs(w) + np.abs(w - q) + eps)).astype(np.float32)

def quantize_row(task):
    w, s, bits = task
    w, s = (np.asarray(w), np.asarray(s))
    if w.ndim != 1 or w.shape != s.shape or len(w) % 128:
        raise ValueError('G128 alignment/shape mismatch')
    fit = {3: conversion3, 4: conversion4}[bits].fit_group
    books, scales = ([], [])
    cost = np.zeros(2, np.float64)
    for start in range(0, len(w), 2048):
        n = min(2048, len(w) - start)
        x = w[start:start + n].astype(np.float64)
        a = s[start:start + n].astype(np.float64)
        c, t = fit(np.pad(x, (0, 2048 - n)), np.pad(a, (0, 2048 - n)))
        levels = t[:n // 128].astype(np.float64)[:, None] * c.astype(np.float64)[None, :]
        z = np.abs(x.reshape(-1, 128)[:, :, None] - levels[:, None, :]).argmin(-1)
        y = np.take_along_axis(levels, z, axis=1).ravel()
        err = (x - y) ** 2
        cost += [np.sum(a * err), np.sum(err)]
        books.append(c)
        scales.append(t)
    return (np.stack(books), np.stack(scales), cost)
