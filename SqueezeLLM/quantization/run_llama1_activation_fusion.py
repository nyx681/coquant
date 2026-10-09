"""Full LLaMA-1 G128 nonuniform fusion with AWQ mean-absolute input weights.

Candidate books retain the existing AWQ-delta/GPTQ-delta definitions. Only the
selection/refinement objective changes. Persist module center books for reuse;
nearest-original-weight codes can be reconstructed without saving dense models.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import gc
import hashlib
import json
import pickle
import shutil
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from chunk_gptq_params import LLAMA_MODULES
from fuse_group_centers import validate_record
from group_center_selection import assign, select_centers
from nuq import kmeans_fit, build_inverse_channel_relative_weights, normalize_score_row, load_awq_channel_scales
from llama import get_model, awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens

MODEL = '/mnt/dfq/pretrained_model/Llama-7b'
ORIGINAL = Path('/mnt/dfq/squeezellm_awq_delta/original_chunks')
AWQ_ROOT = Path('/home/zch/project/Quantification/llama1_w4_g128_center_fusion/full_20260914')
OLD_ROOT = Path('/mnt/dfq/llama1_dual_nonuniform_g128_20260916')
GPTQ = Path('/mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt')
SCALE = '/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt'
DATASET = '/mnt/dfq/datasets/mit-han-lab/pile-val-backup'
VARIANTS = ('activation_selected', 'activation_refined')


def write_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def atomic_torch_save(value, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def tensor_mean_abs_accumulate(acc, count, inputs):
    flat = inputs.detach().reshape(-1, inputs.shape[-1])
    # Same statistic as AWQ get_act_scale; accumulate in FP32/FP64 rather than
    # reducing long token streams in FP16. Every block has only valid tokens.
    acc.add_(flat.float().abs().sum(0).double())
    return count + flat.shape[0]


def calibration_tokens(tokenizer):
    dataset = load_dataset(DATASET, split='validation').shuffle(seed=42)
    samples, articles = [], []
    for item in dataset:
        text = item['text'].strip()
        ids = tokenizer.encode(text)
        if len(ids) > 512 or not ids:
            continue
        samples.append(torch.tensor([ids], dtype=torch.long))
        articles.append(dict(sha256=hashlib.sha256(text.encode()).hexdigest(), tokens=len(ids)))
        if len(samples) == 512:
            break
    if len(samples) != 512:
        raise ValueError('AWQ calibration requires 512 accepted source texts')
    joined = torch.cat(samples, dim=1)
    nblocks = joined.numel() // 512
    if not nblocks:
        raise ValueError('Empty calibration token stream')
    blocks = joined[:, :nblocks * 512].reshape(nblocks, 512)
    manifest = dict(dataset=DATASET, split='validation', seed=42, accepted_texts=512,
                    source_rule='strip, encode, skip empty or >512 tokens; concatenate then split',
                    block_size=512, blocks=nblocks, tokens=blocks.numel(),
                    dropped_tokens=joined.numel()-blocks.numel(), articles=articles,
                    token_sha256=hashlib.sha256(blocks.numpy().tobytes()).hexdigest())
    return blocks, manifest


def collect_activations(root, batch_size):
    target = root / 'activation_mean_abs.pt'
    if target.exists():
        cached = torch.load(target, map_location='cpu')
        if cached['model'] != MODEL or cached['statistic'] != 'mean_abs_input_fp_original':
            raise ValueError('Incompatible activation cache')
        if len(cached['alpha']) != 32 * len(LLAMA_MODULES):
            raise ValueError('Incomplete activation cache')
        print('Reusing activation_mean_abs.pt', flush=True)
        return cached['alpha'], cached['counts']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    tokens, manifest = calibration_tokens(tokenizer)
    write_json(root / 'calibration.json', manifest)
    model = get_model(MODEL).eval().cuda()
    sums, counts, handles = {}, {}, []
    for l, layer in enumerate(model.model.layers):
        for name, suffix in LLAMA_MODULES.items():
            module = layer.get_submodule(suffix)
            key = f'{l}.{name}'
            sums[key] = torch.zeros(module.in_features, dtype=torch.float64, device='cuda')
            counts[key] = 0
            def hook(module, inputs, key=key):
                counts[key] = tensor_mean_abs_accumulate(sums[key], counts[key], inputs[0])
            handles.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for start in tqdm(range(0, len(tokens), batch_size), desc='FP calibration mean abs', mininterval=5):
            # No logits are needed to collect Linear inputs.
            model.model(tokens[start:start+batch_size].cuda(), use_cache=False)
    for handle in handles:
        handle.remove()
    alpha = {}
    for key, total in sums.items():
        if counts[key] != manifest['tokens']:
            raise ValueError(f'Unexpected calibration token count: {key}')
        values = (total / counts[key]).float().cpu()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid activation importance: {key}')
        alpha[key] = values
    atomic_torch_save(dict(model=MODEL, statistic='mean_abs_input_fp_original',
                           alpha=alpha, counts=counts, calibration=manifest), target)
    write_json(root / 'activation_summary.json', {k: dict(channels=len(v), minimum=float(v.min()),
                maximum=float(v.max()), mean=float(v.mean()), zero_channels=int((v==0).sum()),
                tokens=counts[k]) for k,v in alpha.items()})
    del model, layer, module, sums, handles, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return alpha, counts


def refine_group(x, weights, centers):
    refined = centers.copy()
    best = assign(x, weights, refined)[1]
    iterations = 0
    for _ in range(50):
        labels = assign(x, weights, refined)[0]
        new = refined.copy()
        for k in range(16):
            mask = labels == k
            if weights[mask].sum() > 0:
                new[k] = np.average(x[mask].astype(np.float64), weights=weights[mask])
        loss = assign(x, weights, new)[1]
        if loss >= best:
            break
        refined, best = new, loss
        iterations += 1
    return refined, best, iterations


def row(task):
    w, sg, alpha, ac, cached_gptq = task
    groups = len(w) // 128
    books = np.empty((3, groups, 16), dtype=np.float32)
    totals = np.zeros(5, dtype=np.float64)
    for g in range(groups):
        sl = slice(g*128, (g+1)*128)
        x, s, a = w[sl], alpha[sl], ac[g]
        if cached_gptq is None:
            b, _ = kmeans_fit((x[:, None], sg[sl], 16))
        else:
            b = cached_gptq[g]
        chosen = select_centers(x, s, np.r_[a, b]).astype(np.float32)
        refined, loss, iterations = refine_group(x, s, chosen)
        costs = [assign(x, s, c)[1] for c in (a, b, chosen)] + [loss]
        tol = 1e-10 + 1e-9 * max(costs)
        if costs[2] > min(costs[:2]) + tol or costs[3] > costs[2] + tol:
            raise AssertionError('Activation weighted objective increased')
        books[:, g] = b, chosen, refined
        totals += np.r_[costs, iterations]
    return books, totals


def fill_weight(weight, original, centers):
    """Reconstruct nearest-original labels in float64, matching assign()."""
    if centers.shape != (original.shape[0], original.shape[1]//128, 16):
        raise ValueError('Center shape mismatch')
    with torch.no_grad():
        for start in range(0, len(original), 32):
            stop = min(start+32, len(original))
            x = original[start:stop].reshape(stop-start, -1, 128).astype(np.float64)
            c = centers[start:stop]
            labels = np.abs(x[..., None]-c.astype(np.float64)[:, :, None, :]).argmin(-1)
            q = np.take_along_axis(c, labels, axis=2).reshape(stop-start, -1)
            weight[start:stop].copy_(torch.from_numpy(q))


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def validate_sources():
    config = json.loads((AWQ_ROOT/'awq/kmeans_config.json').read_text())
    expected = dict(original_chunks=str(ORIGINAL), gptq_params=str(GPTQ), awq_scale_cache=SCALE,
                    bit=4, group_size=128, status='complete', partial=False,
                    weight_source='awq_scale_inverse_awq_delta', gamma=1, min_weight=.1,
                    delta_lambda=1, power={'o':3, 'rest':4},
                    kmeans={'random_state':0, 'n_init':'auto', 'max_iter':50})
    for key, val in expected.items():
        if config.get(key) != val:
            raise ValueError(f'AWQ source configuration mismatch: {key}')
    old_config = json.loads((OLD_ROOT/'run.json').read_text())
    if old_config.get('status') != 'complete' or old_config.get('model') != MODEL:
        raise ValueError('Old dual-nonuniform run incomplete/incompatible')
    for l in range(32):
        for p in (ORIGINAL/f'layer_{l}.pt', AWQ_ROOT/'awq/lut'/f'l{l}.pkl'):
            if not p.exists():
                raise FileNotFoundError(p)
    baseline = json.loads((OLD_ROOT/'four_ppl.json').read_text())
    for item in baseline:
        for key, val in dict(model=MODEL,bit=4,group_size=128,dataset='wikitext2_test',block_size=2048,batch_size=8).items():
            if item.get(key) != val:
                raise ValueError(f'Baseline protocol mismatch: {key}')
    return baseline


def execute(args, root, config):
    baselines = validate_sources()
    write_json(root/'reused_baselines.json', dict(source=str(OLD_ROOT/'four_ppl.json'),results=baselines))
    alpha, counts = collect_activations(root, args.calibration_batch_size)
    config.update(status='building_centers', activation_modules=len(alpha), calibration_tokens=next(iter(counts.values())))
    write_json(root/'run.json', config)
    params = torch.load(GPTQ, map_location='cpu', mmap=True)
    if len(params) != 224:
        raise ValueError('Unexpected number of GPTQ records')
    # Three FP32 center books only: 3 * 16 * 4 / 128 bytes per model weight.
    required = sum(r['weight'].numel() for r in params.values()) * 1.5
    checkpoint_root = root/'centers'
    checkpoint_root.mkdir(exist_ok=True)
    existing = sum(p.stat().st_size for p in checkpoint_root.glob('*.npz'))
    if shutil.disk_usage(root).free < max(0, required-existing) + 1024**3:
        raise OSError(f'Insufficient disk space for resumable centers: need {(required-existing)/1024**3:.2f} GiB + reserve')
    scales = load_awq_channel_scales(SCALE)
    models = [get_model(MODEL).eval() for _ in VARIANTS]
    torch.set_num_threads(2)
    metrics = {}
    with get_context('spawn').Pool(args.workers) as pool:
        for l in range(32):
            original = torch.load(ORIGINAL/f'layer_{l}.pt', map_location='cpu')
            with (AWQ_ROOT/'awq/lut'/f'l{l}.pkl').open('rb') as f:
                awq_lut = pickle.load(f)
            targets = [dict(m.model.layers[l].named_modules()) for m in models]
            for name, suffix in LLAMA_MODULES.items():
                key = f'{l}.{name}'
                started = time.monotonic()
                w = original[name].float().numpy()
                if not torch.equal(targets[0][suffix].weight.detach().cpu(), original[name].to(targets[0][suffix].weight.dtype)):
                    raise ValueError(f'Original chunks differ from model: {key}')
                record = params[f'model.layers.{l}.{suffix}']
                reconstruction_error = validate_record(record, w.shape)
                native = record['weight'].float().numpy()
                dg = build_inverse_channel_relative_weights(w, np.abs(w-native), gamma=1, min_weight=.1)
                prior = normalize_score_row(scales[l,name].numpy())**(3 if name=='o' else 4)
                sg = prior * (1+normalize_score_row(dg))
                ac = awq_lut[name]['centers']
                act = alpha[key].numpy().astype(np.float64)
                if (ac.shape != (len(w),w.shape[1]//128,16) or act.shape != (w.shape[1],)
                        or not all(np.isfinite(v).all() for v in (w,ac,act,sg)) or np.any(act<0)):
                    raise ValueError(f'Invalid module input: {key}')
                # No AWQ prior/delta factor, exponent or per-group normalization
                # is applied to act. It is the raw mean absolute input statistic.
                fingerprint = digest_arrays(w,sg,act,ac)
                checkpoint = checkpoint_root/f'{key}.npz'
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        if str(saved['fingerprint']) != fingerprint:
                            raise ValueError(f'Checkpoint fingerprint mismatch: {key}')
                        books, totals = saved['books'], saved['totals']
                    print('Reusing module centers', key, flush=True)
                else:
                    books = np.empty((3,len(w),w.shape[1]//128,16),dtype=np.float32)
                    totals = np.zeros(5,dtype=np.float64)
                    tasks = ((w[r],sg,act,ac[r],None) for r in range(len(w)))
                    for r,(row_books,costs) in enumerate(tqdm(pool.imap(row,tasks,chunksize=2),total=len(w),desc=key,mininterval=10)):
                        books[:,r] = row_books
                        totals += costs
                    temp = Path(str(checkpoint)+'.tmp')
                    with temp.open('wb') as f:
                        np.savez(f,books=books,totals=totals,fingerprint=np.asarray(fingerprint))
                    temp.replace(checkpoint)
                if books.shape != (3,len(w),w.shape[1]//128,16) or not np.isfinite(books).all():
                    raise ValueError(f'Invalid saved books: {key}')
                for i in range(2):
                    fill_weight(targets[i][suffix].weight,w,books[i+1])
                metrics[key] = dict(rows=len(w),groups=w.size//128,
                    activation_weighted_sse=dict(zip(('awq_delta','gptq_delta',*VARIANTS),totals[:4].tolist())),
                    accepted_lloyd_steps=int(totals[4]), seconds=time.monotonic()-started,
                    gptq_reference_reconstruction_error=reconstruction_error,
                    checkpoint=str(checkpoint),fingerprint=fingerprint)
                write_json(root/'metrics.json', metrics)
                config.update(completed_modules=len(metrics),last_module=key)
                write_json(root/'run.json',config)
                print('Completed module',key,'seconds',round(metrics[key]['seconds'],1),flush=True)
    del params,original,awq_lut,targets,record,w,native,ac,books
    gc.collect()
    config['status']='evaluating'
    write_json(root/'run.json',config)
    tokens,path = get_awq_ppl_tokens('wikitext',MODEL)
    results=[]
    for i,variant in enumerate(VARIANTS):
        output=root/f'{variant}_ppl.json'
        if output.exists():
            result=json.loads(output.read_text())
            if result.get('run') != str(root/'run.json') or result.get('variant') != variant:
                raise ValueError('Incompatible saved evaluation')
        else:
            with torch.no_grad():
                ppl=awq_compatible_ppl_eval(models[i],tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
            result=dict(variant=variant,ppl=ppl,model=MODEL,bit=4,group_size=128,
                        dataset='wikitext2_test',dataset_path=path,tokens=tokens.numel(),block_size=2048,batch_size=8,
                        objective='mean_abs_input_activation weighted original-weight SSE',run=str(root/'run.json'))
            write_json(output,result)
        results.append(result)
        models[i]=None
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root/'comparison_ppl.json',dict(reused=baselines,new=results))
    config.update(status='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(root/'run.json',config)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--calibration-batch-size',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or args.calibration_batch_size<1:
        parser.error('Positive worker/batch counts required')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=args.resume)
    codepaths=[Path(__file__),Path(__file__).with_name('group_center_selection.py'),Path(__file__).with_name('nuq.py')]
    sourcefiles=[Path(MODEL)/'config.json',GPTQ,AWQ_ROOT/'awq/kmeans_config.json',Path(SCALE)]
    sourcefiles += sorted(Path(MODEL).glob('*.safetensors')) + sorted(Path(MODEL).glob('*.bin'))
    config=dict(model=MODEL,bit=4,group_size=128,variants=VARIANTS,
        objective='sum_j mean_t(abs(X_tj)) * (W_ij-center[label_ij])**2',
        activation_source='unmodified FP model Linear inputs',activation_power=1,
        fusion_AWQ_scale=False,fusion_delta=False,candidates_AWQ_scale=True,candidates_delta=True,
        calibration_dataset=DATASET,calibration_texts=512,calibration_block_size=512,calibration_seed=42,
        gptq_params=str(GPTQ),awq_centers=str(AWQ_ROOT/'awq/lut'),
        kmeans=dict(random_state=0,n_init='auto',max_iter=50),refine_max_iter=50,
        storage='FP32 GPTQ/selected/refined centers; labels reconstructed nearest original; 224 module checkpoints',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in codepaths},
        source_stat={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sourcefiles})
    # JSON-normalize tuples before comparing saved provenance on resume.
    config=json.loads(json.dumps(config))
    if args.resume:
        old=json.loads((root/'run.json').read_text())
        for k,v in config.items():
            if old.get(k)!=v:
                raise ValueError(f'Cannot resume changed configuration/source: {k}')
        if old.get('status')=='complete':
            print('Already complete:',root/'comparison_ppl.json');return
    config.update(status='collecting_activations',workers=args.workers,
                  calibration_batch_size=args.calibration_batch_size,started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(root/'run.json',config)
    try:
        execute(args,root,config)
    except BaseException as error:
        config.update(status='failed',error=repr(error))
        write_json(root/'run.json',config)
        (root/'error.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
