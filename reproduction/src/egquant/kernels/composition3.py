import numpy as np
METRICS = ('awq_sse', 'gptq_sse', 'union16_sse', 'selected_source_scale_sse', 'projected_initial_sse', 'final_fp32_sse', 'ordinary_sse', 'groups', 'groups_worse_than_best_source', 'retained_awq_centers', 'refinement_iterations')

def deletion(cost, keep=8, audit=False):
    active = list(range(cost.shape[1]))
    while len(active) > keep:
        d = cost[:, active]
        nearest = d.argmin(axis=1)
        pair = np.partition(d, 1, axis=1)[:, :2]
        increment = np.bincount(nearest, weights=pair[:, 1] - pair[:, 0], minlength=len(active))
        if audit:
            base = d.min(axis=1).sum()
            brute = np.array([np.delete(d, k, axis=1).min(axis=1).sum() - base for k in range(len(active))])
            np.testing.assert_allclose(increment, brute, rtol=1e-08, atol=1e-10)
        del active[int(increment.argmin())]
    return np.asarray(active, dtype=np.int16)

def refine(x, f, c, t, z, max_iter=30):

    def loss():
        return float(np.sum(f * (x - t[:, None] * c[z]) ** 2))
    history = [loss()]
    for _ in range(max_iter):
        num = np.bincount(z.ravel(), weights=(f * t[:, None] * x).ravel(), minlength=8)
        den = np.bincount(z.ravel(), weights=(f * t[:, None] ** 2).ravel(), minlength=8)
        np.divide(num, den, out=c, where=den > 0)
        v = c[z]
        num = np.sum(f * v * x, axis=1)
        den = np.sum(f * v * v, axis=1)
        proposed = t.copy()
        np.divide(num, den, out=proposed, where=den > 0)
        t = np.maximum(proposed, 1e-20)
        z = np.abs(x[:, :, None] - t[:, None, None] * c[None, None, :]).argmin(-1)
        new_loss = loss()
        roundoff = 64 * np.finfo(np.float64).eps * max(float(np.sum(f * x * x)), 1e-30)
        if new_loss > history[-1] + roundoff + 1e-09 * history[-1]:
            raise ValueError('Refinement objective increased')
        history.append(new_loss)
        if history[-2] - new_loss <= 1e-06 * max(history[-2], 1e-30):
            break
    norm = np.max(np.abs(c))
    if norm > 0:
        c = c / norm
        t = t * norm
    return (c.astype(np.float32), t.astype(np.float32), history)

def fit_group(x, fisher, ca, ta, cg, tg, audit=False):
    x = np.asarray(x, np.float64).reshape(-1, 128)
    fisher = np.asarray(fisher, np.float64).reshape(-1, 128)
    ca, cg = [np.asarray(v, np.float64).reshape(8) for v in (ca, cg)]
    ta, tg = [np.asarray(v, np.float64).reshape(16) for v in (ta, tg)]
    ta, tg = (ta[:len(x)], tg[:len(x)])
    if not all((np.isfinite(v).all() for v in (x, fisher, ca, ta, cg, tg))):
        raise ValueError('Nonfinite inputs')
    if (fisher < 0).any() or (ta <= 0).any() or (tg <= 0).any():
        raise ValueError('Invalid Fisher/scales')
    f = fisher / fisher.max() if fisher.max() > 0 else np.ones_like(fisher)
    levels = np.concatenate((ta[:, None] * ca, tg[:, None] * cg), axis=1)
    error = (x[:, :, None] - levels[:, None, :]) ** 2
    cost = (f[:, :, None] * error).reshape(-1, 16)
    selected = deletion(cost, audit=audit)
    chosen_levels = levels[:, selected]
    z = np.abs(x[:, :, None] - chosen_levels[:, None, :]).argmin(-1)
    selected_error = (x - np.take_along_axis(chosen_levels, z, axis=1)) ** 2
    t = np.sqrt(ta * tg)
    c = np.concatenate((ca, cg))[selected].copy()

    z = np.abs(x[:, :, None] - t[:, None, None] * c[None, None, :]).argmin(-1)
    c, t, history = refine(x, f, c, t, z)
    ylevels = t.astype(np.float64)[:, None] * c.astype(np.float64)
    labels = np.abs(x[:, :, None] - ylevels[:, None, :]).argmin(-1)
    final_error = (x - np.take_along_axis(ylevels, labels, axis=1)) ** 2
    awq = float(np.sum(fisher * error[:, :, :8].min(-1)))
    gptq = float(np.sum(fisher * error[:, :, 8:].min(-1)))
    final = float(np.sum(fisher * final_error))
    initial = history[0] * fisher.max() if fisher.max() > 0 else 0.0
    stats = np.array([awq, gptq, np.sum(fisher * error.min(-1)), np.sum(fisher * selected_error), initial, final, final_error.sum(), 1, final > min(awq, gptq) * (1 + 1e-08), np.sum(selected < 8), len(history) - 1], np.float64)
    if not np.isfinite(c).all() or not np.isfinite(t).all():
        raise ValueError('Invalid fitted codebook')
    if audit:
        assert stats[3] >= stats[2] - 1e-10 * max(stats[2], 1.0)
    return (c, t, selected, stats)

def quantize_row(task):
    w, f, ca, ta, cg, tg = task
    if len(w) % 128 or w.shape != f.shape:
        raise ValueError('G2048 divisibility/shape mismatch')
    books, scales, selected = ([], [], [])
    stats = np.zeros(len(METRICS), np.float64)
    for g, start in enumerate(range(0, len(w), 2048)):
        c, t, ids, s = fit_group(w[start:start + 2048], f[start:start + 2048], ca[g], ta[g], cg[g], tg[g])
        books.append(c)
        scales.append(np.pad(t, (0, 16 - len(t)), constant_values=1))
        selected.append(ids)
        stats += s
    return (np.stack(books), np.stack(scales), np.stack(selected), stats)
