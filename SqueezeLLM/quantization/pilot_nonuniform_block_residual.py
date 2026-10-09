"""LLaMA-2 partial-block diagnostic for two nonuniform fusion extensions.

Reuse AWQ G128 codebooks; rebuild only 64 rows/module of GPTQ-delta using the
original sklearn configuration. No test data, no PPL, no full-model claim.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import copy
import gc
import hashlib
import json
import pickle
import re
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

from llama import get_model
from chunk_gptq_params import LLAMA_MODULES
from fuse_group_centers import validate_record
from group_center_selection import assign, select_centers
from nuq import (kmeans_fit, build_inverse_channel_relative_weights,
                 normalize_score_row, load_awq_channel_scales)
from run_cross_model_dual_nonuniform_g128 import TARGETS

AWQ_ROOT = Path('/mnt/dfq/llama2_w4_g128_center_fusion/retry_20260916/full_static_g128')
NAMES = ('awq_delta', 'gptq_delta', 'fusion')


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unexpected captured argument: {type(value)}')


def calibration(tokenizer):
    articles, current = [], []
    for text in load_dataset('/home/public/dataset/wikitext-2-raw-v1', split='train')['text']:
        if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and current:
            articles.append(''.join(current))
            current = []
        current.append(text)
    if current:
        articles.append(''.join(current))
    excluded = set()
    for path in sorted(Path('/mnt/dfq').glob('*pilot*/calibration.json')):
        excluded.update(v['sha256'] for v in json.loads(path.read_text()))
    blocks, manifest = [], []
    for i in np.random.default_rng(20260921).permutation(len(articles)):
        digest = hashlib.sha256(articles[i].encode()).hexdigest()
        if digest in excluded:
            continue
        ids = tokenizer(articles[i], return_tensors='pt').input_ids
        if ids.shape[1] < 512:
            continue
        split = 'fit' if len(blocks) < 8 else 'select' if len(blocks) < 12 else 'guard'
        blocks.append((split, ids[:, :512]))
        manifest.append(dict(article=int(i), sha256=digest, split=split, tokens=512))
        excluded.add(digest)
        if len(blocks) == 16:
            return blocks, manifest
    raise ValueError('Not enough fresh train articles')


def endpoint_row(task):
    w, sg, shared, ac = task
    centers, labels = [], []
    for g, a in enumerate(ac):
        sl = slice(g * 128, (g + 1) * 128)
        x = w[sl]
        b, _ = kmeans_fit((x[:, None], sg[sl], 16))
        b = b.reshape(-1)
        c = select_centers(x, shared[sl], np.r_[a, b]).astype(np.float32)
        books = [a, b, c]
        assigned = [assign(x, shared[sl], book) for book in books]
        if assigned[2][1] > min(v[1] for v in assigned[:2]) + 1e-8:
            raise AssertionError('Union selection increased shared SSE')
        centers.append(books)
        labels.append([v[0] for v in assigned])
    return (np.asarray(centers, dtype=np.float32).transpose(1, 0, 2),
            np.asarray(labels, dtype=np.uint8).transpose(1, 0, 2))


def reconstruct(centers, labels):
    return centers.gather(2, labels.long()).flatten(1).half().float()


@torch.no_grad()
def lloyd(target, weights, initial, steps=30):
    """Batched weighted 1-D Lloyd, keeping unused center slots unchanged."""
    c = initial.clone()
    weights = weights.expand_as(target)
    for _ in range(steps):
        idx = (target[..., None] - c[:, :, None, :]).square().argmin(-1)
        mass = torch.zeros_like(c).scatter_add_(2, idx, weights)
        sums = torch.zeros_like(c).scatter_add_(2, idx, target * weights)
        c = torch.where(mass > 0, sums / mass.clamp_min(1e-20), c)
    idx = (target[..., None] - c[:, :, None, :]).square().argmin(-1)
    if not torch.isfinite(c).all():
        raise ValueError('Nonfinite generated centers')
    return c, idx


@torch.no_grad()
def residual_generate(w, book, xfit, xselect):
    """Fit conditional residual targets; no guard access during selection.

    R = X(Q-W)^T. The unconstrained update for one group is
    dQ_g^T = -(X_g^T X_g + damping I)^-1 X_g^T R.
    All targets use the same initial R (Jacobi); the full output loss is NOT
    guaranteed to improve, so include the unchanged book in selection.
    """
    c, idx = (v.to(w.device) for v in book)
    q = reconstruct(c, idx)
    groups = w.shape[1] // 128
    xg = xfit.reshape(-1, groups, 128).permute(1, 0, 2)
    gram = xg.transpose(1, 2) @ xg
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clone().clamp_min(1e-12)
    ridge = 0.01 * diagonal.mean(-1)
    gram.diagonal(dim1=-2, dim2=-1).add_(ridge[:, None])
    residual = xfit @ (q - w).T
    rhs = xg.transpose(1, 2) @ residual
    delta = torch.linalg.solve(gram, rhs).permute(2, 0, 1)
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite conditional target')
    def score(weight, x):
        return float((x @ (weight - w).T).square().mean())
    best = (score(q, xselect), (c.clone(), idx.clone()), 0.0)
    trace = [dict(alpha=0.0, select_mse=best[0], fit_mse=score(q, xfit))]
    for alpha in (0.25, 0.5, 1.0):
        target = q.reshape_as(delta) - alpha * delta
        newc, newidx = lloyd(target, diagonal[None, :, :], c)
        newq = reconstruct(newc, newidx)
        select = score(newq, xselect)
        trace.append(dict(alpha=alpha, select_mse=select, fit_mse=score(newq, xfit)))
        if select < best[0]:
            best = (select, (newc.clone(), newidx.clone()), alpha)
    return tuple(v.cpu() for v in best[1]), dict(selected_alpha=best[2], trace=trace)


class LUTLinear(nn.Module):
    """Quantize only the tested rows; remaining rows stay FP16 in this pilot."""
    def __init__(self, original, book):
        super().__init__()
        self.original = original
        for parameter in original.parameters():
            parameter.requires_grad_(False)
        c, idx = book
        self.register_buffer('centers', c.to(original.weight.device).float())
        self.register_buffer('labels', idx.to(original.weight.device).long())
        self.register_buffer('step', self.centers.std(-1, keepdim=True).clamp_min(1e-5))
        self.offset = nn.Parameter(torch.zeros_like(self.centers))

    def codebook(self):
        return self.centers + self.step * self.offset

    def forward(self, x):
        q = self.codebook().gather(2, self.labels).flatten(1).to(self.original.weight.dtype)
        weight = torch.cat((q, self.original.weight[q.shape[0]:]), dim=0)
        return F.linear(x, weight, self.original.bias)


def make_student(teacher, books):
    block = copy.deepcopy(teacher)
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    quantized = {}
    for short, suffix in LLAMA_MODULES.items():
        parent_name, attribute = suffix.rsplit('.', 1)
        parent = block.get_submodule(parent_name)
        replacement = LUTLinear(getattr(parent, attribute), books[short])
        setattr(parent, attribute, replacement)
        quantized[short] = replacement
    return block, quantized


def forward_sample(block, sample):
    args, kwargs, target = tree_to(sample, 'cuda')
    return block(*args, **kwargs)[0].float(), target.float()


@torch.no_grad()
def evaluate(block, samples):
    losses = []
    for sample in samples:
        output, target = forward_sample(block, sample)
        losses.append(float((output - target).square().mean()))
    value = float(np.mean(losses))
    if not np.isfinite(value):
        raise ValueError('Nonfinite block loss')
    return value


def optimize_block(teacher, books, samples, steps=24):
    block, quantized = make_student(teacher, books)
    parameters = [module.offset for module in quantized.values()]
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    scale = max(evaluate(block, samples['fit']), 1e-12)
    best_loss = evaluate(block, samples['select'])
    best_step = 0
    best_offsets = [p.detach().clone() for p in parameters]
    trace = [dict(step=0, select_mse=best_loss)]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, target = forward_sample(block, samples['fit'][(step - 1) % len(samples['fit'])])
        loss = (output - target).square().mean() / scale
        loss = loss + 1e-4 * sum(p.square().mean() for p in parameters)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite block optimization loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for p in parameters:
                p.clamp_(-0.5, 0.5)
        if step in (8, steps):
            value = evaluate(block, samples['select'])
            trace.append(dict(step=step, select_mse=value, gradient_norm=float(norm)))
            if value < best_loss:
                best_loss, best_step = value, step
                best_offsets = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p, value in zip(parameters, best_offsets):
            p.copy_(value)
        guard = evaluate(block, samples['guard'])
        result_books = {name: (m.codebook().cpu(), m.labels.cpu().to(torch.uint8)) for name, m in quantized.items()}
    del block, quantized, optimizer, parameters
    return result_books, dict(selected_step=best_step, select_mse=best_loss,
                              guard_block_mse=guard, trace=trace)


def execute(args, root, config):
    cfg = TARGETS['llama2']
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    blocks, manifest = calibration(AutoTokenizer.from_pretrained(cfg['model'], use_fast=False))
    write_json(root / 'calibration.json', manifest)
    model = get_model(cfg['model']).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = model.model.layers[0]
    samples = {s: [] for s in ('fit', 'select', 'guard')}
    inputs = {name: {s: [] for s in samples} for name in LLAMA_MODULES}
    active = ['fit']
    def block_hook(module, inputs_, kwargs, output):
        samples[active[0]].append(tree_to((inputs_, kwargs, output[0]), 'cpu'))
    handles = [layer.register_forward_hook(block_hook, with_kwargs=True)]
    for name, suffix in LLAMA_MODULES.items():
        def linear_hook(module, values, name=name):
            x = values[0].detach().reshape(-1, values[0].shape[-1])
            inputs[name][active[0]].append(x[::4].float().cpu())
        handles.append(layer.get_submodule(suffix).register_forward_pre_hook(linear_hook))
    with torch.no_grad():
        for split, ids in blocks:
            active[0] = split
            model(ids.cuda(), use_cache=False)
    for handle in handles:
        handle.remove()
    teacher = copy.deepcopy(layer)
    del layer, model, handles
    gc.collect()
    torch.cuda.empty_cache()
    # Round-trip the captured block inputs to catch version-specific kwargs bugs.
    if evaluate(teacher, samples['select']) > 1e-12:
        raise AssertionError('Captured block forward does not match original')
    print('Captured independent fit/select/guard block inputs', flush=True)
    original = torch.load(Path(cfg['original']) / 'layer_0.pt', map_location='cpu')
    awq = torch.load(Path(cfg['awq']) / 'layer_0.pt', map_location='cpu')
    params = torch.load(cfg['params'], map_location='cpu', mmap=True)
    scales = load_awq_channel_scales(cfg['scale'])
    with (AWQ_ROOT / 'awq/lut/l0.pkl').open('rb') as stream:
        cached = pickle.load(stream)
    books = {variant: {} for variant in NAMES}
    with get_context('spawn').Pool(args.workers) as pool:
        for name, suffix in LLAMA_MODULES.items():
            start = time.monotonic()
            w = original[name].float().numpy()
            if not torch.equal(teacher.get_submodule(suffix).weight.detach().cpu(), original[name].half()):
                raise ValueError('Original weights differ from model')
            record = params[f'model.layers.0.{suffix}']
            validate_record(record, w.shape)
            da = build_inverse_channel_relative_weights(w, np.abs(w - awq[name].float().numpy()), gamma=1, min_weight=.1)
            dg = build_inverse_channel_relative_weights(w, np.abs(w - record['weight'].float().numpy()), gamma=1, min_weight=.1)
            prior = normalize_score_row(scales[0, name].numpy()) ** (3 if name == 'o' else 4)
            ma, mg = 1 + normalize_score_row(da), 1 + normalize_score_row(dg)
            sg, shared = prior * mg, prior * np.sqrt(ma * mg)
            ac = cached[name]['centers'][:args.rows]
            tasks = ((w[r], sg, shared, ac[r]) for r in range(args.rows))
            results = list(pool.imap(endpoint_row, tasks))
            cs = np.stack([v[0] for v in results])
            ids = np.stack([v[1] for v in results])
            for i, variant in enumerate(NAMES):
                books[variant][name] = (torch.from_numpy(cs[:, i].copy()), torch.from_numpy(ids[:, i].copy()))
            torch.save({v: books[v][name] for v in NAMES}, root / f'endpoint_{name}.pt')
            print('Prepared nonuniform endpoint books', name, 'seconds', round(time.monotonic()-start, 1), flush=True)
    del params, awq, cached, results, record
    config['status'] = 'residual_generation'
    write_json(root / 'run.json', config)
    residual_books, residual_metrics = {v: {} for v in NAMES}, {v: {} for v in NAMES}
    # The same residual-generation optimizer is given to every initialization.
    for name in LLAMA_MODULES:
        w = original[name][:args.rows].float().cuda()
        x = {s: torch.cat(v).cuda() for s, v in inputs[name].items()}
        for variant in NAMES:
            book, metric = residual_generate(w, books[variant][name], x['fit'], x['select'])
            q = reconstruct(*(t.cuda() for t in book))
            metric['guard_linear_mse'] = float((x['guard'] @ (q-w).T).square().mean())
            residual_books[variant][name] = book
            residual_metrics[variant][name] = metric
        write_json(root / 'residual_linear_results.json', residual_metrics)
        print('Generated residual-directed centers', name, flush=True)
    torch.save(residual_books, root / 'residual_books.pt')
    del x, w, q, inputs, original
    gc.collect()
    results = {}
    for family, starts in [('original', books), ('residual', residual_books)]:
        for variant in NAMES:
            key = family + '_' + variant
            config.update(status='block_optimization', active_variant=key)
            write_json(root / 'run.json', config)
            baseline, _ = make_student(teacher, starts[variant])
            initial = dict(select_mse=evaluate(baseline, samples['select']),
                           guard_block_mse=evaluate(baseline, samples['guard']))
            del baseline
            updated, metric = optimize_block(teacher, starts[variant], samples)
            results[key] = dict(initial=initial, optimized=metric)
            torch.save(updated, root / f'{key}_optimized_books.pt')
            write_json(root / 'block_results.json', results)
            print('Block result', key, json.dumps(results[key]), flush=True)
    config.update(status='complete', result_type='partial-block output MSE, not PPL')
    config.pop('active_variant', None)
    write_json(root / 'run.json', config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 4096 or args.workers < 1:
        parser.error('Invalid rows or workers')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(model=TARGETS['llama2']['model'], layer=0, rows=args.rows,
                  bit=4, group_size=128, centers=16, AWQ_scale=True,
                  endpoint_definition='cached AWQ-delta; original sklearn GPTQ-delta; exact shared-weight union selection',
                  awq_reused=str(AWQ_ROOT / 'awq/lut/l0.pkl'),
                  limitation='only leading rows quantized; other weights FP16; first block only; no PPL',
                  data='fresh article-disjoint WikiText train; fit 8/select 4/guard 4',
                  test_access=False, residual_ridge=0.01, residual_alpha=[0, .25, .5, 1],
                  residual_kmeans_steps=30, block_steps=24, block_lr=.02,
                  block_checkpoints=[0, 8, 24], offset_limit=.5,
                  status='preparing', workers=args.workers)
    write_json(root / 'run.json', config)
    try:
        execute(args, root, config)
    except BaseException as error:
        config.update(status='failed', error=repr(error))
        write_json(root / 'run.json', config)
        (root / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
