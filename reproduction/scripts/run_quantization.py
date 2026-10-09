import _bootstrap
import argparse
import importlib.metadata
import time
from multiprocessing import get_context
import numpy as np
import torch
from egquant.common import ROOT, checked_chunk, digest, modules, output_dir, add_experiment_arguments, experiment_from_args, data_path, save_npz, sha, write_json
from egquant.conversion import importance, quantize_row
from egquant.kernels import composition3, composition4

def execute(cfg, method, workers, resume=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = output_dir(cfg, method)
    root.mkdir(parents=True, exist_ok=resume)
    (root / 'centers').mkdir(exist_ok=True)
    bits = cfg['bits']
    spec = dict(model_id=cfg['model_id'], model_type=cfg['model_type'], layers=cfg['layers'], bits=bits, method=method, eta=0.5, group_size=2048, scale_group_size=128, source_parameters=cfg, versions={k: importlib.metadata.version(k) for k in ('numpy', 'scikit-learn', 'torch')}, code={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/egquant').rglob('*.py'))})
    run_path = root / 'run.json'
    if run_path.exists():
        import json
        old = json.loads(run_path.read_text())
        if old['spec'] != spec:
            raise ValueError('Resume parameters or code differ')
    state = dict(spec=spec, status='running', completed_modules=0, total_modules=cfg['layers'] * len(modules(cfg)))
    write_json(run_path, state)
    fusion = method == 'EGComp-AG'
    source_method = 'awq' if method == 'EGConv-A' else 'gptq'
    if fusion:
        import json
        for endpoint in ('EGConv-A', 'EGConv-G'):
            p = output_dir(cfg, endpoint) / 'run.json'
            meta = json.loads(p.read_text())
            if meta['status'] != 'complete' or meta['spec']['source_parameters'] != cfg:
                raise ValueError('Matching, complete conversion endpoints are required')
    metrics = {}
    try:
        with get_context('spawn').Pool(workers) as pool:
            for layer in range(cfg['layers']):
                paths = {k: data_path(cfg['paths'][k]) / f'layer_{layer}.pt' for k in ('original', 'fisher')}
                if not fusion:
                    paths['reference'] = data_path(cfg['paths'][source_method]) / f'layer_{layer}.pt'
                inputs = {k: sha(p) for k, p in paths.items()}
                chunks = {k: checked_chunk(p, cfg) for k, p in paths.items()}
                for name in modules(cfg):
                    start = time.monotonic()
                    key = f'{layer}.{name}'
                    w = chunks['original'][name].float().numpy()
                    f = chunks['fisher'][name].float().numpy()
                    if f.shape != w.shape or (f < 0).any():
                        raise ValueError('Invalid Fisher array')
                    hashes = dict(inputs)
                    shape = (len(w), (w.shape[1] + 2047) // 2048)
                    if fusion:
                        cp_a = output_dir(cfg, 'EGConv-A') / 'centers' / f'{key}.npz'
                        cp_g = output_dir(cfg, 'EGConv-G') / 'centers' / f'{key}.npz'
                        hashes.update(awq_centers=sha(cp_a), gptq_centers=sha(cp_g))
                        with np.load(cp_a, allow_pickle=False) as z:
                            ca, ta = (z['centers'], z['scales'])
                        with np.load(cp_g, allow_pickle=False) as z:
                            cg, tg = (z['centers'], z['scales'])
                        if ca.shape != shape + (2 ** bits,) or cg.shape != ca.shape or ta.shape != shape + (16,) or (tg.shape != ta.shape):
                            raise ValueError('Endpoint shape mismatch')
                        tasks = ((w[r], f[r], ca[r], ta[r], cg[r], tg[r]) for r in range(len(w)))
                        kernel = {3: composition3, 4: composition4}[bits]
                        fn, loss = (kernel.quantize_row, np.zeros(len(kernel.METRICS)))
                    else:
                        q = chunks['reference'][name].float().numpy()
                        s = f * (1 + importance(w, q))
                        tasks = ((w[r], s[r], bits) for r in range(len(w)))
                        fn, loss = (quantize_row, np.zeros(2))
                    fingerprint = digest(dict(inputs=hashes, spec=spec, module=key))
                    cp = root / 'centers' / f'{key}.npz'
                    if cp.exists():
                        with np.load(cp, allow_pickle=False) as z:
                            if str(z['fingerprint']) != fingerprint:
                                raise ValueError('Checkpoint inputs differ: ' + key)
                            centers, scales, loss = (z['centers'], z['scales'], z['cost'])
                            if centers.shape != shape + (2 ** bits,) or scales.shape != shape + (16,):
                                raise ValueError('Checkpoint shape mismatch')
                    else:
                        centers = np.empty(shape + (2 ** bits,), np.float32)
                        scales = np.empty(shape + (16,), np.float32)
                        for r, result in enumerate(pool.imap(fn, tasks, chunksize=4)):
                            centers[r], scales[r] = result[:2]
                            loss += result[-1]
                        if not np.isfinite(centers).all() or not np.isfinite(scales).all():
                            raise ValueError('Nonfinite fit')
                        save_npz(cp, centers=centers, scales=scales, cost=loss, fingerprint=np.asarray(fingerprint))
                    metrics[key] = dict(inputs=hashes, checkpoint_sha256=sha(cp), seconds=time.monotonic() - start, shape=list(w.shape), cost=loss.tolist())
                    state.update(completed_modules=len(metrics), current_module=key)
                    write_json(root / 'metrics.json', metrics)
                    write_json(run_path, state)
                    print(method, key, len(metrics), '/', state['total_modules'], flush=True)
        state['status'] = 'complete'
        write_json(run_path, state)
    except BaseException:
        state['status'] = 'failed'
        write_json(run_path, state)
        raise

def main():
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--method', choices=['EGConv-A', 'EGConv-G', 'EGComp-AG'], required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    execute(experiment_from_args(a), a.method, a.workers, a.resume)
if __name__ == '__main__':
    main()
