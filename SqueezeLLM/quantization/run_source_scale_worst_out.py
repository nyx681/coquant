"""Resumable OPT B fusion, with direct FP16-parameter WikiText-2 evaluation."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from source_scale_worst_out import quantize_row, METRICS


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(root, cfg, state, gpu_lock):
    import torch
    from llama import get_model, awq_compatible_ppl_eval
    from squeezellm.model_parse import get_layers
    from squeezellm.datautils import get_awq_ppl_tokens
    from chunk_gptq_params import OPT_MODULES
    from run_cross_model_delta_only_fusions import gpu_slot
    tokens, token_path = get_awq_ppl_tokens('wikitext', cfg['model'])
    state.update(status='waiting_for_gpu', evaluation_completed_modules=0)
    write_json(root/'run.json', state)
    torch.set_num_threads(2)
    with gpu_slot(gpu_lock, lambda **kw: None):
        model = get_model(cfg['model']).eval().cuda()
        dtype = str(next(model.parameters()).dtype)
        num_weights = 0; parameter_bits = 0
        with torch.no_grad():
            for l, layer in enumerate(get_layers(model, cfg['model_type'])):
                original = torch.load(Path(cfg['original'])/f'layer_{l}.pt', map_location='cpu')
                for name, suffix in OPT_MODULES.items():
                    key = f'{l}.{name}'; module = layer.get_submodule(suffix); w = original[name]
                    if not torch.equal(module.weight.cpu(), w.to(module.weight.dtype)):
                        raise ValueError('Original model mismatch '+key)
                    with np.load(root/'centers'/f'{key}.npz') as z:
                        c, t = z['centers'], z['scales']
                    c16, t16 = c.astype(np.float16), t.astype(np.float16)
                    if not np.isfinite(c16).all() or not np.isfinite(t16).all():
                        raise ValueError('FP16 overflow '+key)
                    rows, cols = w.shape; groups = cols//2048
                    for r in range(0, rows, 16):
                        x = w[r:r+16].to(device='cuda', dtype=torch.float64).reshape(-1, groups, 16, 128)
                        co = torch.from_numpy(c[r:r+16]).to(device='cuda', dtype=torch.float64)
                        to = torch.from_numpy(t[r:r+16]).to(device='cuda', dtype=torch.float64)
                        levels = to[:, :, :, None]*co[:, :, None, :]
                        idx = (x[:, :, :, :, None]-levels[:, :, :, None, :]).abs().argmin(-1)
                        cn = torch.from_numpy(c16[r:r+16]).to(device='cuda', dtype=torch.float64)
                        tn = torch.from_numpy(t16[r:r+16]).to(device='cuda', dtype=torch.float64)
                        newlevels = tn[:, :, :, None]*cn[:, :, None, :]
                        y = torch.gather(newlevels, 3, idx).reshape(-1, cols).to(module.weight.dtype)
                        module.weight[r:r+16].copy_(y)
                    np.savez(root/'centers_fp16'/f'{key}.npz', centers=c16, scales=t16)
                    num_weights += w.numel(); parameter_bits += 16*(c16.size+t16.size)
                    state.update(status='reconstructing_fp16', evaluation_completed_modules=state['evaluation_completed_modules']+1)
                    write_json(root/'run.json', state)
                    print('FP16 reconstructed', key, flush=True)
            state['status'] = 'evaluating'; write_json(root/'run.json', state)
            ppl = awq_compatible_ppl_eval(model, tokens, torch.device('cuda:0'), 'wikitext', block_size=2048, batch_size=8)
    result = dict(ppl=ppl, model=cfg['model'], dataset='wikitext2_test', dataset_path=token_path,
                  tokens=tokens.numel(), block_size=2048, batch_size=8, base_dtype=dtype,
                  bit=4, group_size=2048, small_group_size=128, center_dtype='float16',
                  scale_dtype='float16', fixed_indices=True,
                  actual_quantized_weight_bits=4+parameter_bits/num_weights,
                  bit_accounting='packed 4-bit labels + FP16 centers/scales; excludes unquantized tensors/headers',
                  objective='common Fisher weighted reconstruction error',
                  algorithm='source-scale-aware greedy worst-out 32->16, then joint centers/scales refinement')
    write_json(root/'ppl.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--awq', required=True); p.add_argument('--gptq', required=True)
    p.add_argument('--fisher', required=True); p.add_argument('--target-config', required=True)
    p.add_argument('--output', required=True); p.add_argument('--workers', type=int, default=24)
    p.add_argument('--gpu-lock', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--baseline-csv', required=True)
    args = p.parse_args()
    import torch
    from chunk_gptq_params import OPT_MODULES
    torch.set_num_threads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=args.resume)
    cfg = json.loads(Path(args.target_config).read_text())
    assert cfg['model_type'] == 'opt'
    ca_root, cg_root, fisher_root = Path(args.awq), Path(args.gptq), Path(args.fisher)
    source_runs = [json.loads((r/'run.json').read_text()) for r in (ca_root, cg_root)]
    for source, run in zip((ca_root, cg_root), source_runs):
        assert run['status']=='complete' and run['model']==cfg['model']
        assert run['group_size']==2048 and run['small_group_size']==128
        assert run['design']=='self_floor05' and run['fisher_combination']=='F*(1+d)'
        assert Path(run['fisher_root']).resolve()==fisher_root.resolve()
        assert len(list((source/'centers').glob('*.npz')))==cfg['layers']*len(OPT_MODULES)
    assert source_runs[0]['variant']=='awq_delta' and source_runs[1]['variant']=='gptq_delta'
    spec = dict(awq=str(ca_root), gptq=str(cg_root), fisher=str(fisher_root), target=cfg,
                objective='sum F*(W-t_subgroup*c_label)^2', zero_fisher='ordinary SSE tie-break only',
                selection='32 source-tagged centers with source-specific subgroup scales; smallest removal penalty first',
                projection='geometric mean source scales; selected normalized centers; source-aware initial assignments',
                refinement='30 alternating weighted-LS rounds max, relative tolerance 1e-6; positive scales',
                endpoint_fallback=False, final_parameter_dtype='float16',
                code_sha256={str(q):sha(q) for q in (Path(__file__), Path(__file__).with_name('source_scale_worst_out.py'))})
    if args.resume:
        old = json.loads((root/'run.json').read_text())
        assert old['spec']==spec, 'Resume spec mismatch'
        if old['status']=='complete':
            print('Already complete'); return
    state = dict(spec=spec, status='starting', pid=os.getpid(), workers=args.workers,
                 started_at=time.strftime('%F %T'), completed_modules=0,
                 total_modules=cfg['layers']*len(OPT_MODULES))
    write_json(root/'run.json', state); write_json(root/'target.json', cfg)
    for sub in ('centers', 'centers_fp16'):
        (root/sub).mkdir(exist_ok=True)
    metrics = {}
    try:
        # fork workers are numpy-only; no model/CUDA initialization precedes fork.
        with get_context('fork').Pool(args.workers) as pool:
            for l in range(cfg['layers']):
                wp = Path(cfg['original'])/f'layer_{l}.pt'; fp = fisher_root/f'layer_{l}.pt'
                layer_hashes = {'original':sha(wp), 'fisher':sha(fp)}
                original = torch.load(wp, map_location='cpu')
                fisher = torch.load(fp, map_location='cpu')
                for name in OPT_MODULES:
                    key = f'{l}.{name}'; started = time.monotonic()
                    state.update(status='building_centers', current_module=key, completed_rows=0)
                    write_json(root/'run.json', state)
                    ap = ca_root/'centers'/f'{key}.npz'; gp = cg_root/'centers'/f'{key}.npz'
                    hashes = dict(layer_hashes, awq=sha(ap), gptq=sha(gp))
                    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                    cp = root/'centers'/f'{key}.npz'; reused = cp.exists()
                    w = original[name].float().numpy(); f = fisher[name].float().numpy()
                    if w.shape!=f.shape or w.shape[1]%2048: raise ValueError('Shape/alignment '+key)
                    if reused:
                        with np.load(cp) as z:
                            assert str(z['fingerprint'])==fingerprint
                            stats = z['stats']
                    else:
                        with np.load(ap) as z: ca, ta = z['centers'], z['scales']
                        with np.load(gp) as z: cg, tg = z['centers'], z['scales']
                        shape = (len(w), w.shape[1]//2048, 16)
                        assert ca.shape==ta.shape==cg.shape==tg.shape==shape
                        centers = np.empty(shape, np.float32); scales = np.empty(shape, np.float32)
                        selected = np.empty(shape, np.int16); stats = np.zeros(len(METRICS), np.float64)
                        tasks = ((w[r], f[r], ca[r], ta[r], cg[r], tg[r]) for r in range(len(w)))
                        last = time.monotonic()
                        for r, (c, t, ids, s) in enumerate(pool.imap(quantize_row, tasks, chunksize=4)):
                            centers[r]=c; scales[r]=t; selected[r]=ids; stats+=s
                            if time.monotonic()-last > 15:
                                state.update(completed_rows=r+1, total_rows=len(w))
                                write_json(root/'run.json', state)
                                print('Rows', key, r+1, '/', len(w), flush=True); last=time.monotonic()
                        tmp = cp.with_suffix('.npz.tmp')
                        with tmp.open('wb') as out:
                            np.savez(out, centers=centers, scales=scales, selected_source_ids=selected,
                                     stats=stats, fingerprint=np.asarray(fingerprint))
                            out.flush(); os.fsync(out.fileno())
                        tmp.replace(cp)
                    metrics[key] = dict(zip(METRICS, stats.tolist()))
                    metrics[key].update(seconds=time.monotonic()-started, reused=reused, source_hashes=hashes)
                    write_json(root/'metrics.json', metrics)
                    state.update(completed_modules=len(metrics), completed_rows=len(w))
                    write_json(root/'run.json', state)
                    print('Completed', key, round(metrics[key]['seconds'], 2), flush=True)
        del original, fisher; gc.collect()
        result = evaluate(root, cfg, state, args.gpu_lock)
        with Path(args.baseline_csv).open() as inp:
            baseline_rows = [r for r in csv.DictReader(inp) if r['method'] in ('awq_self_floor05', 'gptq_self_floor05')]
        with (root/'comparison.csv').open('w') as out:
            writer=csv.writer(out); writer.writerow(['method','WikiText2_PPL','quantized_weight_bits','source'])
            for r in baseline_rows:writer.writerow([r['method'],r['FP16_parameters_ppl'],4.25,args.baseline_csv])
            writer.writerow(['source_scale_worst_out_fusion',result['ppl'],result['actual_quantized_weight_bits'],str(root/'ppl.json')])
        state.update(status='complete', finished_at=time.strftime('%F %T'), ppl=result['ppl'])
        write_json(root/'run.json', state); print('RESULT', json.dumps(result), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); write_json(root/'run.json', state); raise


if __name__=='__main__':
    main()
