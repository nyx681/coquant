"""Round B codebooks/scales to FP16, holding original assignments fixed."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from llama import get_model,awq_compatible_ppl_eval
from chunk_gptq_params import OPT_MODULES,LLAMA_MODULES
from squeezellm.model_parse import get_layers
from squeezellm.datautils import get_awq_ppl_tokens
from contextlib import contextmanager
import fcntl
@contextmanager
def gpu_slot(lock_path, update):
 lock_path=os.environ.get('C4_GPU_LOCK',lock_path)
 with open(lock_path,'a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  try:
   while torch.cuda.mem_get_info()[0] < float(os.environ.get('C4_MIN_FREE_GIB','40'))*1024**3:
    time.sleep(10)
   yield
  finally:fcntl.flock(handle,fcntl.LOCK_UN)
BATCH_SIZE=int(os.environ.get('C4_BATCH_SIZE','8'))
r=Path(sys.argv[1]);source=Path(sys.argv[2]);cfg=json.loads(Path(sys.argv[3]).read_text());baseline=json.loads((source/'ppl.json').read_text()) if (source/'ppl.json').exists() else None
if baseline is None:
 source_run=json.loads((source/'run.json').read_text())
 module_names=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
 assert source_run['model']==cfg['model'] and source_run['group_size']==2048 and source_run['small_group_size']==128
 assert {p.stem for p in (source/'centers').glob('*.npz')}=={f'{l}.{n}' for l in range(cfg['layers']) for n in module_names}
 model_config=json.loads((Path(cfg['model'])/'config.json').read_text())
 dtype=model_config.get('torch_dtype',model_config.get('dtype'))
 assert dtype in ('float16','bfloat16'),dtype
 baseline={'ppl':None,'base_dtype':'torch.'+dtype}

state=dict(status='starting',source=str(source),source_wikitext_ppl=baseline['ppl'],index_policy='nearest original FP32 scale*center in float64, frozen before FP16 rounding',center_dtype='float16',scale_dtype='float16',reconstruction='exact FP16 parameter product via float64 then cast to native model dtype',nominal_quantized_weight_bits=3.1875,completed_modules=0,total_modules=cfg['layers']*(6 if cfg['model_type']=='opt' else 7),pid=os.getpid(),started_at=time.strftime('%F %T'))
def save():
 p=r/'run.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(r/'run.json')
save();torch.set_num_threads(2);metrics={};total_weights=0;parameter_bits=0;(r/'centers_fp16').mkdir(exist_ok=True)
try:
 cached=torch.load(sys.argv[4],map_location='cpu');assert cached['model']==cfg['model'];tokens=cached['tokens'];path=cached['dataset_path']
 protocol=json.loads((Path(sys.argv[4]).parent/'evaluation_protocol.json').read_text())
 token_hash=hashlib.sha256(tokens.numpy().tobytes()).hexdigest();assert token_hash==protocol['token_sha256']
 assert tokens.numel()==protocol['tokens']
 (r/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
 with gpu_slot('/mnt/data/zch/quantization_experiments/B_c4_evaluation.gpu.lock',lambda **kw:None):
  model=get_model(cfg['model']).eval().cuda();assert str(next(model.parameters()).dtype)==baseline['base_dtype']
  with torch.no_grad():
   for l,layer in enumerate(get_layers(model,cfg['model_type'])):
    original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
    for name,suffix in (OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES).items():
     key=f'{l}.{name}';module=layer.get_submodule(suffix);w=original[name]
     assert torch.equal(module.weight.cpu(),w.to(module.weight.dtype))
     with np.load(source/'centers'/f'{key}.npz',allow_pickle=False) as z:c=z['centers'];t=z['scales']
     assert c.dtype==np.float32 and t.dtype==np.float32
     c16=c.astype(np.float16);t16=t.astype(np.float16)
     assert np.isfinite(c16).all() and np.isfinite(t16).all()
     rows,cols=w.shape;groups=(cols+2047)//2048;assert c.shape==(rows,groups,8) and t.shape==(rows,groups,16)
     changed=0;maxdiff=0.;sqdiff=0.
     for start in range(0,rows,16):
      x=w[start:start+16].to(device='cuda',dtype=torch.float64);x=torch.nn.functional.pad(x,(0,groups*2048-cols)).reshape(-1,groups,16,128)
      co=torch.from_numpy(c[start:start+16]).to(device='cuda',dtype=torch.float64);to=torch.from_numpy(t[start:start+16]).to(device='cuda',dtype=torch.float64)
      levels=to[:,:,:,None]*co[:,:,None,:]
      idx=(x[:,:,:,:,None]-levels[:,:,:,None,:]).abs().argmin(-1)
      old=torch.gather(levels,3,idx).reshape(-1,groups*2048)[:,:cols].to(module.weight.dtype)
      cn=torch.from_numpy(c16[start:start+16]).to(device='cuda',dtype=torch.float64);tn=torch.from_numpy(t16[start:start+16]).to(device='cuda',dtype=torch.float64)
      newlevels=tn[:,:,:,None]*cn[:,:,None,:]
      new=torch.gather(newlevels,3,idx).reshape(-1,groups*2048)[:,:cols].to(module.weight.dtype)
      changed+=int((old!=new).sum());diff=new.float()-old.float();maxdiff=max(maxdiff,float(diff.abs().max()));sqdiff+=float(diff.double().square().sum());module.weight[start:start+16].copy_(new)
     valid_scales=t16.reshape(rows,-1)[:,:cols//128].copy()
     np.savez(r/'centers_fp16'/f'{key}.npz',centers=c16,scales=valid_scales)
     total_weights+=w.numel();parameter_bits+=16*(c16.size+valid_scales.size)
     metrics[key]=dict(changed_weights=changed,total_weights=w.numel(),max_abs_reconstruction_change=maxdiff,squared_reconstruction_change=sqdiff,source_sha256=hashlib.sha256((source/'centers'/f'{key}.npz').read_bytes()).hexdigest())
     state.update(status='reconstructing',completed_modules=len(metrics),current_module=key);save();print('Reconstructed',key,flush=True)
   state['status']='evaluating';save();ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'c4',block_size=2048,batch_size=BATCH_SIZE)
 result=dict(ppl=ppl,source_wikitext_ppl=baseline['ppl'],model=cfg['model'],dataset='c4_validation_local_shard',dataset_path=path,tokens=tokens.numel(),token_sha256=token_hash,token_cache=sys.argv[4],blocks=tokens.numel()//2048,evaluated_tokens=tokens.numel()//2048*2048,recomputed_centers=False,block_size=2048,batch_size=BATCH_SIZE,bit=3,group_size=2048,small_group_size=128,center_dtype='float16',scale_dtype='float16',base_dtype=baseline['base_dtype'],fixed_indices=True,nominal_quantized_weight_bits=3.1875,actual_quantized_weight_bits=3+parameter_bits/total_weights,bit_accounting='3-bit packed indices plus FP16 centers and valid subgroup scales; excludes unquantized tensors and file headers',total_quantized_weights=total_weights,source=str(source))
 (r/'metrics.json').write_text(json.dumps(metrics,indent=2));(r/'ppl.json').write_text(json.dumps(result,indent=2));state.update(actual_quantized_weight_bits=result['actual_quantized_weight_bits'],status='complete',finished_at=time.strftime('%F %T'));save();print('RESULT',json.dumps(result),flush=True)
except BaseException as e:state.update(status='failed',error=repr(e));save();raise
