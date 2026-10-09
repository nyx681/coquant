"""Calibration diagnostic: choose complete GPTQ/RTN groups by output error.

No test-set access. A routed group retains one endpoint's 16 centers and codes.
This pilot saves measurements, not a deployable model or a PPL result.
"""
import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from pilot_activation_index_fusion import (
    BASE, LAYERS, MODEL, OPT_MODULES, AutoTokenizer, get_model,
    load_dataset, optimize, reconstruct, rtn,
)
from chunk_gptq_params import LLAMA_MODULES
from fuse_group_centers import validate_record
from squeezellm.model_parse import get_layers

TARGETS = {
    'opt': dict(model=MODEL, model_type='opt', params=str(BASE / 'static_gptq_params.pt')),
    'llama2': dict(model='/mnt/dfq/pretrained_model/llama-2-7b', model_type='llama',
                   params='/home/zch/project/Quantification/llama2_static_g128_retry_20260916/llama-2-7b-w4-g128-static-gptq_params.pt'),
}


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
    for path in sorted(Path('/mnt/dfq').glob('opt67b_activation_*_pilot_20260917/calibration.json')):
        excluded.update(item['sha256'] for item in json.loads(path.read_text()))
    blocks, manifest = [], []
    for i in np.random.default_rng(20260920).permutation(len(articles)):
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
    raise ValueError('Not enough fresh calibration articles')


@torch.no_grad()
def route_groups(w, qa, qb, xfit, xselect, xguard, group_size=128):
    """Two-start binary coordinate descent; choose checkpoint on select only."""
    rows, cols = w.shape
    assert cols % group_size == 0 and qa.shape == qb.shape == w.shape
    assert all(torch.isfinite(t).all() for t in (w, qa, qb, xfit, xselect, xguard))
    groups = cols // group_size
    delta = (qb - qa).reshape(rows, groups, group_size)
    effects = torch.einsum('tgj,rgj->gtr', xfit.reshape(-1, groups, group_size), delta)
    def score(q, x):
        return float((x @ (q - w).T).square().mean())
    best, traces = None, []
    for start, initial in enumerate((qa, qb)):
        q = initial.clone()
        mask = torch.full((rows, groups), bool(start), device=w.device)
        residual = xfit @ (q - w).T
        for sweep in range(3):
            fit, select = score(q, xfit), score(q, xselect)
            traces.append(dict(start=start, sweep=sweep, fit_mse=fit, select_mse=select))
            if best is None or select < best[0]:
                best = (select, q.clone(), mask.clone(), start, sweep)
            if sweep == 2:
                break
            before = fit
            for g in range(groups):
                # Changing one whole group changes the full output residual.
                change = effects[g] * (1 - 2 * mask[:, g].float())[None, :]
                improvement = 2 * (residual * change).sum(0) + change.square().sum(0)
                flip = improvement < 0
                residual += change * flip[None, :]
                mask[:, g] ^= flip
                lo, hi = g * group_size, (g + 1) * group_size
                q[:, lo:hi] = torch.where(mask[:, g, None], qb[:, lo:hi], qa[:, lo:hi])
            # Recompute to bound drift in the incremental residual.
            residual = xfit @ (q - w).T
            if score(q, xfit) > before * (1 + 1e-4) + 1e-8:
                raise AssertionError('Group routing increased fit error')
    select, q, mask, start, sweep = best
    return dict(select_output_mse=select, guard_output_mse=score(q, xguard),
                selected_start=start, selected_sweep=sweep,
                rtn_group_fraction=float(mask.float().mean()), selection_trace=traces,
                gptq_guard_output_mse=score(qa, xguard), rtn_guard_output_mse=score(qb, xguard))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', choices=tuple(TARGETS), default='opt')
    args = parser.parse_args()
    target = TARGETS[args.target]
    module_map = OPT_MODULES if target['model_type'] == 'opt' else LLAMA_MODULES
    prefix = 'model.decoder.layers' if target['model_type'] == 'opt' else 'model.layers'
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = dict(model=target['model'], params=target['params'], target=args.target,
                  layers=LAYERS, output_rows=64, group_size=128,
                  centers=16, method='complete group routing; output residual coordinate descent',
                  sweeps=2, starts=['native_gptq', 'native_rtn'], test_access=False,
                  control='native endpoints and one-sweep code-optimized endpoints',
                  limitation='routing and code controls have different search spaces and budgets',
                  status='running')
    def save():
        (root / 'config.json').write_text(json.dumps(config, indent=2))
    save()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        blocks, manifest = calibration(AutoTokenizer.from_pretrained(target['model'], use_fast=False))
        (root / 'calibration.json').write_text(json.dumps(manifest, indent=2))
        model = get_model(target['model']).eval().cuda()
        model_layers = get_layers(model, target['model_type'])
        captured, original, handles, active = {}, {}, [], ['fit']
        for layer in LAYERS:
            modules = dict(model_layers[layer].named_modules())
            for short, suffix in module_map.items():
                key = f'{layer}.{short}'
                original[key] = modules[suffix].weight[:64].detach().float().cpu().clone()
                captured[key] = {s: [] for s in ('fit', 'select', 'guard')}
                def hook(module, inputs, key=key):
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                    captured[key][active[0]].append(x[::8].float().cpu())
                handles.append(modules[suffix].register_forward_pre_hook(hook))
        with torch.no_grad():
            for split, ids in blocks:
                active[0] = split
                model(ids.cuda(), use_cache=False)
        for handle in handles:
            handle.remove()
        del model, model_layers, modules, handles
        gc.collect()
        torch.cuda.empty_cache()
        params = torch.load(target['params'], map_location='cpu', mmap=True)
        results = {}
        for layer in LAYERS:
            for short, suffix in module_map.items():
                key = f'{layer}.{short}'
                w = original[key].cuda()
                x = {s: torch.cat(v).cuda() for s, v in captured[key].items()}
                record = params[f'{prefix}.{layer}.{suffix}']
                validate_record(record, record['weight'].shape)
                if record['weight'].shape[1] != w.shape[1]:
                    raise ValueError(f'GPTQ input width mismatch: {key}')
                scale, zero = record['scale'][:64].numpy(), record['zero'][:64].numpy()
                centers = scale[:, :, None] * (np.arange(16, dtype=np.float32) - zero[:, :, None])
                native = record['weight'][:64].float().numpy().reshape(64, -1, 128)
                labels = np.clip(np.round(native / scale[:, :, None] + zero[:, :, None]), 0, 15).astype(np.uint8)
                rc, rl = rtn(w.cpu().numpy().reshape(64, -1, 128))
                qa = torch.as_tensor(native.reshape(w.shape), device='cuda')
                qb = reconstruct(torch.as_tensor(rc, device='cuda'), torch.as_tensor(rl, device='cuda').long()).half().float()
                result = route_groups(w, qa, qb, x['fit'], x['select'], x['guard'])
                for name, c, codes in [('gptq', centers, labels), ('rtn', rc, rl)]:
                    result[name + '_code_control'] = optimize(c, codes, w, x['fit'], x['select'], x['guard'])
                results[key] = result
                (root / 'results.json').write_text(json.dumps(results, indent=2))
                print(key, json.dumps(result), flush=True)
                del w, x, qa, qb
        config.update(status='complete', modules=len(results),
                      wins_vs_native=sum(v['guard_output_mse'] < min(v['gptq_guard_output_mse'], v['rtn_guard_output_mse']) for v in results.values()),
                      wins_vs_code_controls=sum(v['guard_output_mse'] < min(v['gptq_code_control']['guard_output_mse'], v['rtn_code_control']['guard_output_mse']) for v in results.values()))
        save()
        print(json.dumps(config), flush=True)
    except Exception as error:
        config.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
