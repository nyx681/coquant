"""Reuse S=N(d) endpoints; exact unit/mean-absolute-activation G128 fusion.

The two objectives use the same candidate books, original weights, DP solver,
reconstruction and PPL protocol as the completed LLaMA-1 experiments. Endpoint
PPLs are not dependencies: read each atomically completed module as available.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import fcntl
import gc
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from chunk_gptq_params import LLAMA_MODULES, OPT_MODULES
from group_center_selection import assign, select_centers
from llama import get_model, awq_compatible_ppl_eval
from run_cross_model_dual_nonuniform_g128 import TARGETS
from run_llama1_activation_fusion import (
    atomic_torch_save, calibration_tokens, digest_arrays, fill_weight,
    tensor_mean_abs_accumulate, write_json,
)
from squeezellm.datautils import get_awq_ppl_tokens
from squeezellm.model_parse import get_layers

VARIANTS = ('unit', 'activation')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


@contextmanager
def gpu_slot(lock_path, update):
    # Serialize this suite's GPU phases, without locking CPU center selection.
    with Path(lock_path).open('a') as handle:
        update(status='waiting_for_gpu')
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(10)
        try:
            while torch.cuda.mem_get_info()[0] < 40 * 1024**3:
                time.sleep(10)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect_activation(a, cfg, modules, root, update):
    path = root / 'activation_mean_abs.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu')
    else:
        with gpu_slot(a.gpu_lock, update):
            update(status='collecting_activation')
            tokenizer = AutoTokenizer.from_pretrained(cfg['model'], use_fast=False)
            tokens, manifest = calibration_tokens(tokenizer)
            write_json(root / 'calibration.json', manifest)
            model = get_model(cfg['model']).eval().cuda()
            layers = get_layers(model, cfg['model_type'])
            sums, counts, handles = {}, {}, []
            for l, layer in enumerate(layers):
                original = torch.load(Path(cfg['original']) / f'layer_{l}.pt', map_location='cpu')
                for name, suffix in modules.items():
                    module = layer.get_submodule(suffix)
                    if not torch.equal(module.weight.detach().cpu(), original[name].to(module.weight.dtype)):
                        raise ValueError(f'Original model/chunk mismatch: {l}.{name}')
                    key = f'{l}.{name}'
                    sums[key] = torch.zeros(module.in_features, dtype=torch.float64, device='cuda')
                    counts[key] = 0

                    def hook(module, inputs, key=key):
                        counts[key] = tensor_mean_abs_accumulate(sums[key], counts[key], inputs[0])

                    handles.append(module.register_forward_pre_hook(hook))
            with torch.no_grad():
                for start in tqdm(range(0, len(tokens), 4), desc='FP calibration mean abs', mininterval=10):
                    model.model(tokens[start:start + 4].cuda(), use_cache=False)
            for handle in handles:
                handle.remove()
            alpha = {key: (total / counts[key]).float().cpu() for key, total in sums.items()}
            cache = dict(model=cfg['model'], statistic='mean_abs_input_fp_original',
                         alpha=alpha, counts=counts, calibration=manifest)
            atomic_torch_save(cache, path)
            del model, layers, layer, module, sums, handles, tokens, original
            gc.collect()
            torch.cuda.empty_cache()
    expected = {f'{l}.{name}' for l in range(32) for name in modules}
    if (cache.get('model') != cfg['model'] or cache.get('statistic') != 'mean_abs_input_fp_original'
            or set(cache['alpha']) != expected or set(cache['counts']) != expected):
        raise ValueError('Activation cache model/statistic/modules mismatch')
    if any(n != cache['calibration']['tokens'] for n in cache['counts'].values()):
        raise ValueError('Activation token count mismatch')
    alpha = {key: v.float().numpy() for key, v in cache['alpha'].items()}
    if any(not np.isfinite(v).all() or (v < 0).any() for v in alpha.values()):
        raise ValueError('Invalid activation weights')
    write_json(root / 'activation_summary.json', {
        key: dict(channels=len(v), minimum=float(v.min()), maximum=float(v.max()),
                  mean=float(v.mean()), zero_channels=int((v == 0).sum()), tokens=cache['counts'][key])
        for key, v in alpha.items()
    })
    update(activation_cache=str(path), activation_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
           calibration_tokens=cache['calibration']['tokens'])
    return alpha


def select_row(task):
    x, ca, cg, alpha, variants = task
    books = np.empty((len(variants), *ca.shape), np.float32)
    costs = np.zeros((len(variants), 3), np.float64)
    for g in range(len(ca)):
        sl = slice(g * 128, (g + 1) * 128)
        values, candidates = x[sl], np.r_[ca[g], cg[g]]
        for index, variant in enumerate(variants):
            weights = np.ones(128, np.float64) if variant == 'unit' else alpha[sl]
            c = select_centers(values, weights, candidates).astype(np.float32)
            if not np.isin(c, candidates).all():
                raise AssertionError('Center outside candidate union')
            loss = np.array([assign(values, weights, v)[1] for v in (ca[g], cg[g], c)])
            if loss[2] > min(loss[:2]) + 1e-10 + 1e-9 * max(loss):
                raise AssertionError('Selected SSE exceeds endpoint under common objective')
            books[index, g] = c
            costs[index] += loss
    return books, costs


def validate_endpoints(a, cfg):
    for variant in ('awq_delta', 'gptq_delta'):
        meta = read_json(Path(a.endpoint_root) / variant / 'run.json')
        for key, expected in dict(model=cfg['model'], bit=4, group_size=128, AWQ_scale=False,
                                  importance_offset=0, variant=variant).items():
            if meta.get(key) != expected:
                raise ValueError(f'Endpoint provenance mismatch: {variant}/{key}')


def wait_book(a, variant, key, update):
    source = Path(a.endpoint_root) / variant
    path = source / 'centers' / f'{key}.npz'
    while not path.exists():
        status = read_json(source / 'run.json').get('status')
        if status in ('failed', 'complete', 'paused_by_user'):
            raise RuntimeError(f'Missing source {path}; producer status={status}')
        jobs = read_json(Path(a.producer_jobs))
        matches = [job for job in jobs.values() if Path(job.get('output', '')) == source]
        if len(matches) != 1 or not Path(f'/proc/{matches[0]["pid"]}').exists():
            raise RuntimeError(f'Candidate producer unavailable: {source}')
        update(status='waiting_for_candidate', waiting_for=str(path))
        time.sleep(15)
    with np.load(path, allow_pickle=False) as z:
        return z['centers'], dict(path=str(path), fingerprint=str(z['fingerprint']))


def execute(a, root, run, update):
    cfg = TARGETS[a.target]
    modules = OPT_MODULES if cfg['model_type'] == 'opt' else LLAMA_MODULES
    torch.set_num_threads(1)
    validate_endpoints(a, cfg)
    alpha = collect_activation(a, cfg, modules, root, update)
    for variant in VARIANTS:
        (root / variant / 'centers').mkdir(parents=True, exist_ok=True)
    metrics = read_json(root / 'metrics.json')
    workers = a.workers
    pool = get_context('spawn').Pool(workers)
    try:
        for l in range(32):
            original = torch.load(Path(cfg['original']) / f'layer_{l}.pt', map_location='cpu')
            for name in modules:
                if a.producer_run and read_json(Path(a.producer_run)).get('status') == 'complete' and workers != a.boost_workers:
                    pool.close()
                    pool.join()
                    workers = a.boost_workers
                    pool = get_context('spawn').Pool(workers)
                key = f'{l}.{name}'
                start = time.monotonic()
                ca, pa = wait_book(a, 'awq_delta', key, update)
                cg, pg = wait_book(a, 'gptq_delta', key, update)
                w = original[name].float().numpy()
                shape = (len(w), w.shape[1] // 128, 16)
                if w.shape[1] % 128 or ca.shape != shape or cg.shape != shape or alpha[key].shape != (w.shape[1],):
                    raise ValueError(f'Invalid weight/candidate/activation shape: {key}')
                if not all(np.isfinite(v).all() for v in (w, ca, cg)):
                    raise ValueError('Nonfinite source')
                fingerprints = dict(unit=digest_arrays(w, ca, cg), activation=digest_arrays(w, ca, cg, alpha[key]))
                missing, costs, reused = [], {}, []
                for variant in VARIANTS:
                    path = root / variant / 'centers' / f'{key}.npz'
                    if path.exists():
                        with np.load(path, allow_pickle=False) as z:
                            if str(z['fingerprint']) != fingerprints[variant] or z['centers'].shape != shape:
                                raise ValueError('Checkpoint mismatch')
                            costs[variant] = z['cost'].tolist()
                        reused.append(variant)
                    else:
                        missing.append(variant)
                update(status='selecting_centers', current_module=key, completed_modules=len(metrics), workers=workers)
                if missing:
                    books = np.empty((len(missing), *shape), np.float32)
                    loss = np.zeros((len(missing), 3), np.float64)
                    tasks = ((w[i], ca[i], cg[i], alpha[key], missing) for i in range(len(w)))
                    for i, (row_books, row_cost) in enumerate(tqdm(
                            pool.imap(select_row, tasks, chunksize=4), total=len(w), desc=key, mininterval=30)):
                        books[:, i] = row_books
                        loss += row_cost
                    for index, variant in enumerate(missing):
                        path = root / variant / 'centers' / f'{key}.npz'
                        tmp = Path(str(path) + '.tmp')
                        with tmp.open('wb') as handle:
                            np.savez(handle, centers=books[index], cost=loss[index],
                                     fingerprint=np.asarray(fingerprints[variant]))
                        tmp.replace(path)
                        costs[variant] = loss[index].tolist()
                    del books
                metrics[key] = dict(seconds=time.monotonic() - start, groups=w.size // 128,
                                    costs=costs, cost_order=['awq_endpoint_nearest', 'gptq_endpoint_nearest', 'selected'],
                                    sources=dict(awq=pa, gptq=pg), fingerprints=fingerprints, reused=reused)
                write_json(root / 'metrics.json', metrics)
                update(completed_modules=len(metrics), last_module=key)
                print('Completed', a.target, key, round(metrics[key]['seconds'], 1), flush=True)
    finally:
        pool.close()
        pool.join()
    del original, w, ca, cg
    gc.collect()
    for variant in VARIANTS:
        out = root / variant
        if read_json(out / 'run.json').get('status') == 'complete' and (out / 'ppl.json').exists():
            continue
        state = dict(model=cfg['model'], variant=variant, bit=4, group_size=128,
                     source=str(root / 'run.json'), status='reconstructing', started_at=time.strftime('%F %T'))
        write_json(out / 'run.json', state)
        update(status='reconstructing', current_variant=variant)
        model = get_model(cfg['model']).eval()
        for l, layer in enumerate(get_layers(model, cfg['model_type'])):
            original = torch.load(Path(cfg['original']) / f'layer_{l}.pt', map_location='cpu')
            for name, suffix in modules.items():
                module = layer.get_submodule(suffix)
                if not torch.equal(module.weight.detach().cpu(), original[name].to(module.weight.dtype)):
                    raise ValueError(f'Original model/chunk mismatch during evaluation: {l}.{name}')
                with np.load(out / 'centers' / f'{l}.{name}.npz', allow_pickle=False) as z:
                    fill_weight(module.weight, original[name].float().numpy(), z['centers'])
        del original, layer, module
        gc.collect()
        with gpu_slot(a.gpu_lock, update):
            update(status='evaluating', current_variant=variant)
            state['status'] = 'evaluating'
            write_json(out / 'run.json', state)
            torch.set_num_threads(2)
            tokens, dataset_path = get_awq_ppl_tokens('wikitext', cfg['model'])
            with torch.no_grad():
                ppl = awq_compatible_ppl_eval(model, tokens, torch.device('cuda:0'),
                                               'wikitext', block_size=2048, batch_size=8)
            result = dict(model=cfg['model'], variant=variant, ppl=ppl, bit=4, group_size=128,
                          dataset='wikitext2_test', dataset_path=dataset_path, tokens=tokens.numel(),
                          block_size=2048, batch_size=8, endpoint_root=a.endpoint_root,
                          candidate_AWQ_scale=False, candidate_importance_offset=0,
                          fusion_AWQ_scale=False, fusion_delta=False, fusion_activation=variant == 'activation',
                          objective='sum_j coefficient_j * min_c (W_ij-c)^2',
                          coefficient='1' if variant == 'unit' else 'mean_t(abs(X_tj)), original FP model',
                          center_refinement=False, run=str(root / 'run.json'))
            write_json(out / 'ppl.json', result)
            state.update(status='complete', finished_at=time.strftime('%F %T'))
            write_json(out / 'run.json', state)
            del model, tokens
            gc.collect()
            torch.cuda.empty_cache()
        print('PPL', a.target, variant, ppl, flush=True)
    update(status='complete', finished_at=time.strftime('%F %T'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--endpoint-root', required=True)
    parser.add_argument('--producer-jobs', required=True)
    parser.add_argument('--producer-run')
    parser.add_argument('--gpu-lock', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--boost-workers', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    if min(a.workers, a.boost_workers) < 1:
        parser.error('Workers must be positive')
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=a.resume)
    cfg = TARGETS[a.target]
    files = [Path(__file__), Path(__file__).with_name('group_center_selection.py'),
             Path(__file__).with_name('run_llama1_activation_fusion.py'),
             Path(__file__).parent.parent / 'llama.py', Path(__file__).parent.parent / 'squeezellm/datautils.py']
    spec = dict(target=a.target, model=cfg['model'], endpoint_root=a.endpoint_root, bit=4, group_size=128,
                candidate_AWQ_scale=False, candidate_importance_offset=0, center_refinement=False,
                total_modules=192 if a.target == 'opt' else 224, variants=VARIANTS,
                code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    spec = json.loads(json.dumps(spec))
    old = read_json(root / 'run.json')
    if a.resume:
        if any(old.get(key) != value for key, value in spec.items()):
            raise ValueError('Resume provenance mismatch')
        if old.get('status') == 'complete':
            return
    run = dict(spec, status='starting', started_at=old.get('started_at', time.strftime('%F %T')), pid=os.getpid())

    def update(**fields):
        run.update(fields, updated_at=time.strftime('%F %T'))
        if fields.get('status') != 'waiting_for_candidate':
            run.pop('waiting_for', None)
        write_json(root / 'run.json', run)

    update()
    try:
        execute(a, root, run, update)
    except BaseException as exc:
        update(status='failed', error=repr(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
