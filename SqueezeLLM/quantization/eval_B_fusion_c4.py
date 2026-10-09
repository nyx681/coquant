"""C4-only evaluation of a completed FP16 B fusion; no quantization fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
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


def evaluate(root, source, cfg, state, gpu_lock, token_cache):
    import torch
    from llama import get_model, awq_compatible_ppl_eval
    from squeezellm.model_parse import get_layers
    from squeezellm.datautils import get_awq_ppl_tokens
    from chunk_gptq_params import OPT_MODULES
    from run_cross_model_delta_only_fusions import gpu_slot
    cached=torch.load(token_cache, map_location='cpu')
    assert cached['model']==cfg['model']
    tokens=cached['tokens'];token_path=cached['dataset_path']
    protocol=json.loads((Path(token_cache).parent/'evaluation_protocol.json').read_text())
    token_hash=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    assert token_hash==protocol['token_sha256'] and tokens.numel()==protocol['tokens']
    write_json(root/'evaluation_protocol.json',protocol)
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
                    with np.load(source/'centers'/f'{key}.npz') as z:
                        c, t = z['centers'], z['scales']
                    with np.load(source/'centers_fp16'/f'{key}.npz') as fp:
                        c16,t16=fp['centers'],fp['scales']
                    np.testing.assert_array_equal(c16,c.astype(np.float16))
                    np.testing.assert_array_equal(t16,t.astype(np.float16))
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
                    pass  # Reuse source FP16 parameters without rewriting them.
                    num_weights += w.numel(); parameter_bits += 16*(c16.size+t16.size)
                    state.update(status='reconstructing_fp16', evaluation_completed_modules=state['evaluation_completed_modules']+1)
                    write_json(root/'run.json', state)
                    print('FP16 reconstructed', key, flush=True)
            state['status'] = 'evaluating'; write_json(root/'run.json', state)
            ppl = awq_compatible_ppl_eval(model, tokens, torch.device('cuda:0'), 'c4', block_size=2048, batch_size=8)
    result = dict(ppl=ppl, model=cfg['model'], dataset='c4_validation_local_shard', dataset_path=token_path,
                  tokens=tokens.numel(), token_sha256=token_hash, token_cache=str(token_cache),
                  evaluated_tokens=tokens.numel()//2048*2048, blocks=tokens.numel()//2048,
                  source=str(source), source_wikitext_ppl=json.loads((source/'ppl.json').read_text())['ppl'],
                  recomputed_centers=False, recalibrated=False, block_size=2048, batch_size=8, base_dtype=dtype,
                  bit=4, group_size=2048, small_group_size=128, center_dtype='float16',
                  scale_dtype='float16', fixed_indices=True,
                  actual_quantized_weight_bits=4+parameter_bits/num_weights,
                  bit_accounting='packed 4-bit labels + FP16 centers/scales; excludes unquantized tensors/headers',
                  objective='common Fisher weighted reconstruction error',
                  algorithm='source-scale-aware greedy worst-out 32->16, then joint centers/scales refinement')
    write_json(root/'ppl.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--token-cache',required=True);p.add_argument('--gpu-lock',required=True)
    a=p.parse_args();root=Path(a.output);source=Path(a.source)
    root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((source/'target.json').read_text())
    baseline=json.loads((source/'ppl.json').read_text())
    assert json.loads((source/'run.json').read_text())['status']=='complete'
    assert baseline['center_dtype']==baseline['scale_dtype']=='float16'
    assert baseline['actual_quantized_weight_bits']==4.25
    state=dict(status='preflight',pid=os.getpid(),source=str(source),dataset='c4_validation_local_shard',
               completed_modules=192,total_modules=192,evaluation_only=True,recomputed_centers=False,
               started_at=time.strftime('%F %T'),token_cache=a.token_cache,
               source_run_sha256=sha(source/'run.json'),code_sha256=sha(Path(__file__)))
    write_json(root/'run.json',state)
    try:
        result=evaluate(root,source,cfg,state,a.gpu_lock,a.token_cache)
        state.update(status='complete',ppl=result['ppl'],finished_at=time.strftime('%F %T'))
        write_json(root/'run.json',state)
        print('RESULT',json.dumps(result),flush=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e));write_json(root/'run.json',state);raise
if __name__=='__main__':main()
