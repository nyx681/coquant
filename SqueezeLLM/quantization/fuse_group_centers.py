"""G128 AWQ-delta nonuniform + native GPTQ uniform candidate fusion.

Outputs compact group LUTs for float simulation, not packed CUDA kernels.
Existing output directories are refused. No test-set evaluation is triggered.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
import pickle
import shutil
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from nuq import (build_inverse_channel_relative_weights, normalize_score_row,
                 load_awq_channel_scales, kmeans_fit)
from chunk_gptq_params import LLAMA_MODULES
from group_center_selection import assign, select_centers

VARIANTS = ('awq', 'gptq_nearest', 'selected', 'refined')


def process_row(task):
    x, sample_weight, scales, zeros, native = task
    groups = len(scales)
    books = {v: np.empty((groups, 16), np.float32) for v in VARIANTS}
    labels = {v: np.empty((groups, 128), np.uint8) for v in VARIANTS}
    # Per variant: weighted SSE and unweighted SSE; plus native GPTQ.
    metrics = np.zeros((5, 2), np.float64)
    for g in range(groups):
        w = x[g*128:(g+1)*128]
        s = sample_weight[g*128:(g+1)*128]
        a, _ = kmeans_fit((w.reshape(-1, 1), s, 16))
        uniform = (scales[g] * (np.arange(16, dtype=np.float32) - zeros[g])).astype(np.float32)
        chosen = select_centers(w, s, np.concatenate((a, uniform))).astype(np.float32)
        refined = chosen.copy()
        _, best, _ = assign(w, s, refined)
        for _ in range(50):
            index, _, _ = assign(w, s, refined)
            updated = refined.copy()
            for c in range(16):
                mask = index == c
                if s[mask].sum() > 0:
                    updated[c] = np.average(w[mask].astype(np.float64), weights=s[mask])
            _, objective, _ = assign(w, s, updated)
            if objective >= best:
                break
            refined, best = updated, objective
        for v_idx, (v, centers) in enumerate(zip(VARIANTS, (a, uniform, chosen, refined))):
            index, weighted, ordinary = assign(w, s, centers)
            books[v][g], labels[v][g] = centers, index
            metrics[v_idx] += weighted, ordinary
        # Check each group, not just a module aggregate.
        objectives = [assign(w, s, c)[1] for c in (a, uniform, chosen, refined)]
        tol = 1e-10 + 1e-9 * max(objectives)
        if objectives[2] > min(objectives[:2]) + tol or objectives[3] > objectives[2] + tol:
            raise AssertionError('Selection/refinement increased the common objective')
    error = (x.astype(np.float64) - native.astype(np.float64)) ** 2
    metrics[4] = np.dot(sample_weight, error), error.sum()
    return books, labels, metrics


def validate_record(record, shape):
    rows, cols = shape
    if cols % 128 or record.get('group_size', 128) != 128 or int(record.get('maxq', -1)) != 15:
        raise ValueError('Requires W4-G128 with full 128-element groups')
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
        codes = torch.round(w/s + z).clamp(0, 15)
        reconstructed = (s*(codes-z)).to(native.dtype).float()
        # Allow one dtype ULP for historical export arithmetic rounding.
        tolerance = torch.finfo(native.dtype).eps * torch.maximum(w.abs(), reconstructed.abs()) + torch.finfo(native.dtype).tiny
        error = (w-reconstructed).abs()
        if (error > tolerance).any():
            raise ValueError('GPTQ scales/zeros do not reconstruct native weights in original column order')
        max_error = max(max_error, error.max().item())
    return max_error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original_chunks', required=True)
    p.add_argument('--awq_chunks', required=True)
    p.add_argument('--gptq_params', required=True)
    p.add_argument('--awq_scale_cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--layers', default='all', help='all or comma-separated layer IDs')
    p.add_argument('--modules', default=','.join(LLAMA_MODULES))
    p.add_argument('--row_limit', type=int, default=0, help='Diagnostic partial rows; not a full-model LUT')
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args()
    if args.workers < 1 or args.row_limit < 0:
        p.error('workers must be positive; row_limit must be nonnegative')
    modules = args.modules.split(',')
    if len(set(modules)) != len(modules) or set(modules)-set(LLAMA_MODULES):
        p.error('Invalid or duplicate modules')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), format='squeezellm_group_lut_v1', group_size=128, bit=4,
                  weight_source='awq_scale_inverse_awq_delta', gamma=1., min_weight=.1,
                  delta_lambda=1., power={'o': 3, 'rest': 4}, scale_normalization='module_median',
                  kmeans={'random_state': 0, 'n_init': 'auto', 'max_iter': 50},
                  exploratory=True, status='running')
    (root/'run.json').write_text(json.dumps(config, indent=2))
    for v in VARIANTS:
        (root/v/'lut').mkdir(parents=True)
        (root/v/'kmeans_config.json').write_text(json.dumps(dict(config, variant=v), indent=2))
    torch.set_num_threads(1)
    params = torch.load(args.gptq_params, map_location='cpu', mmap=True)
    awq_scales = load_awq_channel_scales(args.awq_scale_cache)
    layers = (sorted(int(x.stem.split('_')[-1]) for x in Path(args.original_chunks).glob('layer_*.pt'))
              if args.layers == 'all' else list(map(int, args.layers.split(','))))
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('No layers or duplicate layer IDs')
    # Four uncompressed variants: FP32 centers + uint8 labels = 6 bytes/weight.
    estimated_bytes = 0
    for layer in layers:
        for module in modules:
            record = params[f'model.layers.{layer}.{LLAMA_MODULES[module]}']
            rows, cols = record['weight'].shape
            estimated_bytes += (min(args.row_limit, rows) if args.row_limit else rows)*cols*6
    if shutil.disk_usage(root).free < estimated_bytes + 2*1024**3:
        raise OSError(f'Insufficient disk space: need about {estimated_bytes / 1024**3:.1f} GiB plus 2 GiB reserve')
    summary = {}
    with threadpool_limits(limits=1), get_context('spawn').Pool(args.workers) as pool:
        for layer in layers:
            original = torch.load(Path(args.original_chunks)/f'layer_{layer}.pt', map_location='cpu')
            awq = torch.load(Path(args.awq_chunks)/f'layer_{layer}.pt', map_location='cpu')
            layer_luts = {v: {} for v in VARIANTS}
            for module in modules:
                w = original[module].float().numpy()
                aw = awq[module].float().numpy()
                if w.shape != aw.shape or not np.isfinite(w).all() or not np.isfinite(aw).all():
                    raise ValueError('Invalid original/AWQ reference weights')
                record = params[f'model.layers.{layer}.{LLAMA_MODULES[module]}']
                reconstruction_error = validate_record(record, w.shape)
                scale = awq_scales[layer, module].numpy()
                if scale.shape != (w.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError('Invalid AWQ scale')
                d = build_inverse_channel_relative_weights(w, np.abs(w-aw), gamma=1., min_weight=.1)
                s = normalize_score_row(scale)**(3 if module == 'o' else 4) * (1+normalize_score_row(d))
                if not np.isfinite(s).all() or np.any(s <= 0):
                    raise ValueError('Invalid AWQ-delta sample weights')
                rows = min(args.row_limit, len(w)) if args.row_limit else len(w)
                groups = w.shape[1]//128
                for v in VARIANTS:
                    layer_luts[v][module] = dict(format='squeezellm_group_lut_v1', group_size=128,
                        centers=np.empty((rows, groups, 16), np.float32),
                        labels=np.empty((rows, groups, 128), np.uint8))
                scales, zeros, native = (record[k].float().numpy() for k in ('scale','zero','weight'))
                tasks = ((w[r], s, scales[r], zeros[r], native[r]) for r in range(rows))
                total = np.zeros((5,2), np.float64)
                for r, (books, labels, metrics) in enumerate(tqdm(pool.imap(process_row, tasks, chunksize=1), total=rows, desc=f'l{layer}/{module}')):
                    for v in VARIANTS:
                        layer_luts[v][module]['centers'][r] = books[v]
                        layer_luts[v][module]['labels'][r] = labels[v]
                    total += metrics
                summary[f'{layer}.{module}'] = dict(rows=rows, groups=rows*groups,
                    gptq_reconstruction_max_abs_error=reconstruction_error,
                    metrics={v: dict(weighted_sse=float(total[i,0]), mse=float(total[i,1]/(rows*w.shape[1])))
                             for i,v in enumerate((*VARIANTS, 'gptq_native'))})
                (root/'metrics.json').write_text(json.dumps(summary, indent=2))
                print(json.dumps({f'{layer}.{module}': summary[f'{layer}.{module}']}), flush=True)
            for v in VARIANTS:
                target = root/v/'lut'/f'l{layer}.pkl'
                with open(str(target)+'.incomplete', 'wb') as f:
                    pickle.dump(layer_luts[v], f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(str(target)+'.incomplete', target)
    config['status'] = 'complete'
    config['partial'] = bool(args.row_limit or args.layers != 'all' or set(modules) != set(LLAMA_MODULES))
    (root/'run.json').write_text(json.dumps(config, indent=2))
    for v in VARIANTS:
        (root/v/'kmeans_config.json').write_text(json.dumps(dict(config, variant=v), indent=2))
    print(f'Completed: {root}', flush=True)


if __name__ == '__main__':
    main()
