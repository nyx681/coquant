"""Round B codebooks/scales to FP16, holding original assignments fixed."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from llama import get_model,awq_compatible_ppl_eval
from chunk_gptq_params import OPT_MODULES
from squeezellm.datautils import get_awq_ppl_tokens
from run_cross_model_delta_only_fusions import gpu_slot
r=Path(sys.argv[1]);source=Path('/mnt/data/zch/quantization_experiments/opt67b_B_fisher_delta_20260923_091052/self_floor05');cfg=json.loads(Path('/mnt/data/zch/quantization_experiments/opt67b_ABC_after_qwen_20260922_233548/target.json').read_text());baseline=json.loads((source/'ppl.json').read_text())
state=dict(status='starting',source=str(source),baseline_ppl=baseline['ppl'],index_policy='nearest original FP32 scale*center in float64, frozen before FP16 rounding',center_dtype='float16',scale_dtype='float16',reconstruction='exact FP16 parameter product via float64 then cast to native model dtype',nominal_quantized_weight_bits=4.25,completed_modules=0,total_modules=192,pid=os.getpid(),started_at=time.strftime('%F %T'))
def save():
 p=r/'run.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(r/'run.json')
save();torch.set_num_threads(2);metrics={};(r/'centers_fp16').mkdir()
try:
 tokens,path=get_awq_ppl_tokens('wikitext',cfg['model'])
 assert tokens.numel()==baseline['tokens']
 with gpu_slot('/mnt/data/zch/quantization_experiments/qwen3_self_delta_g2048_20260922_222333/gpu.lock',lambda **kw:None):
  model=get_model(cfg['model']).eval().cuda();assert str(next(model.parameters()).dtype)==baseline['base_dtype']
  with torch.no_grad():
   for l,layer in enumerate(model.model.decoder.layers):
    original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
    for name,suffix in OPT_MODULES.items():
     key=f'{l}.{name}';module=layer.get_submodule(suffix);w=original[name]
     assert torch.equal(module.weight.cpu(),w.to(module.weight.dtype))
     with np.load(source/'centers'/f'{key}.npz',allow_pickle=False) as z:c=z['centers'];t=z['scales']
     assert c.dtype==np.float32 and t.dtype==np.float32
     c16=c.astype(np.float16);t16=t.astype(np.float16)
     assert np.isfinite(c16).all() and np.isfinite(t16).all()
     rows,cols=w.shape;groups=cols//2048;assert c.shape==t.shape==(rows,groups,16)
     changed=0;maxdiff=0.;sqdiff=0.
     for start in range(0,rows,16):
      x=w[start:start+16].to(device='cuda',dtype=torch.float64).reshape(-1,groups,16,128)
      co=torch.from_numpy(c[start:start+16]).to(device='cuda',dtype=torch.float64);to=torch.from_numpy(t[start:start+16]).to(device='cuda',dtype=torch.float64)
      levels=to[:,:,:,None]*co[:,:,None,:]
      idx=(x[:,:,:,:,None]-levels[:,:,:,None,:]).abs().argmin(-1)
      old=torch.gather(levels,3,idx).reshape(-1,cols).to(module.weight.dtype)
      cn=torch.from_numpy(c16[start:start+16]).to(device='cuda',dtype=torch.float64);tn=torch.from_numpy(t16[start:start+16]).to(device='cuda',dtype=torch.float64)
      newlevels=tn[:,:,:,None]*cn[:,:,None,:]
      new=torch.gather(newlevels,3,idx).reshape(-1,cols).to(module.weight.dtype)
      changed+=int((old!=new).sum());diff=new.float()-old.float();maxdiff=max(maxdiff,float(diff.abs().max()));sqdiff+=float(diff.double().square().sum());module.weight[start:start+16].copy_(new)
     np.savez(r/'centers_fp16'/f'{key}.npz',centers=c16,scales=t16)
     metrics[key]=dict(changed_weights=changed,total_weights=w.numel(),max_abs_reconstruction_change=maxdiff,squared_reconstruction_change=sqdiff,source_sha256=hashlib.sha256((source/'centers'/f'{key}.npz').read_bytes()).hexdigest())
     state.update(status='reconstructing',completed_modules=len(metrics),current_module=key);save();print('Reconstructed',key,flush=True)
   state['status']='evaluating';save();ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
 result=dict(ppl=ppl,baseline_ppl=baseline['ppl'],ppl_change=ppl-baseline['ppl'],model=cfg['model'],dataset='wikitext2_test',dataset_path=path,tokens=tokens.numel(),block_size=2048,batch_size=8,bit=4,group_size=2048,small_group_size=128,center_dtype='float16',scale_dtype='float16',base_dtype=baseline['base_dtype'],fixed_indices=True,nominal_quantized_weight_bits=4.25,source=str(source))
 (r/'metrics.json').write_text(json.dumps(metrics,indent=2));(r/'ppl.json').write_text(json.dumps(result,indent=2));state.update(status='complete',finished_at=time.strftime('%F %T'));save();print('RESULT',json.dumps(result),flush=True)
except BaseException as e:state.update(status='failed',error=repr(e));save();raise
