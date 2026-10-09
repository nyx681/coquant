"""Exact weighted 1D selection from a finite candidate codebook."""

import numpy as np


def assign(values, weights, centers):
    distance = (np.asarray(values, dtype=np.float64)[:, None]
                - np.asarray(centers, dtype=np.float64)[None, :]) ** 2
    labels = distance.argmin(axis=1).astype(np.uint8)
    error = distance[np.arange(len(labels)), labels]
    return labels, float(np.dot(weights, error)), float(error.sum())


def select_centers(values, weights, candidates, k=16):
    """Globally minimize weighted nearest-center SSE, keeping candidate values.

    Float64 costs, deterministic ties. Duplicate/unused slots are padded when
    there are fewer than k distinct candidates. Weights must be nonnegative.
    """
    x = np.asarray(values, dtype=np.float64)
    s = np.asarray(weights, dtype=np.float64)
    u = np.unique(np.asarray(candidates, dtype=np.float64))
    if (x.ndim != 1 or s.shape != x.shape or not len(x) or not len(u)
            or not 1 <= k <= 256 or np.any(s < 0)
            or not all(np.isfinite(v).all() for v in (x, s, u))):
        raise ValueError("Invalid values, sample weights, candidates or k")
    m = len(u)
    if m <= k:
        return np.pad(u, (0, k - m), mode="edge")
    cost = s[None, :] * (u[:, None] - x[None, :]) ** 2
    left = (cost * (x[None, :] < u[:, None])).sum(axis=1)
    right = (cost * (x[None, :] >= u[:, None])).sum(axis=1)
    edges = np.full((m, m), np.inf)
    for a in range(m - 1):
        mask = (x[None, :] >= u[a]) & (x[None, :] < u[a+1:, None])
        edges[a, a+1:] = (np.minimum(cost[a], cost[a+1:]) * mask).sum(axis=1)
    dp = left
    parents = np.full((k + 1, m), -1, dtype=np.int32)
    for t in range(2, k + 1):
        transition = dp[:, None] + edges
        parents[t] = transition.argmin(axis=0)
        dp = transition.min(axis=0)
    b = int(np.argmin(dp + right))
    chosen = [b]
    for t in range(k, 1, -1):
        b = int(parents[t, b])
        chosen.append(b)
    return u[chosen[::-1]]
