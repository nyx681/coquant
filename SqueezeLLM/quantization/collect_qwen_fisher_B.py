"""Empirical Fisher: mean squared per-sequence CE gradients, FP32 accumulation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,importlib.util,gc
from pathlib import Path
import torch
from tqdm import tqdm
from llama import get_model
from chunk_gptq_params import LLAMA_MODULES
from run_cross_model_delta_only_fusions import gpu_slot
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
model_path='/mnt/dfq/pretrained_model/Qwen3-8B-Base'
torch.set_num_threads(1);torch.manual_seed(0)
spec=importlib.util.spec_from_file_location('fisher_datautils','/home/zch/project/SqueezeLLM-gradients/datautils.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
loader,_=data.get_loaders('c4',nsamples=100,seed=0,seqlen=512,model=model_path)
tokens=torch.cat([v[0] for v in loader]);torch.save(tokens,root/'calibration_tokens.pt')
manifest=dict(status='collecting',model=model_path,dataset='C4 train',samples=100,seqlen=512,seed=0,statistic='mean per-sequence squared CE gradient',accumulator_dtype='float32',model_dtype='bfloat16',gradient_dtype='bfloat16',sparsity=0,dropout=False,tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),completed_samples=0)
def save():
 p=root/'manifest.json.tmp';p.write_text(json.dumps(manifest,indent=2));p.replace(root/'manifest.json')
save()
lock='/mnt/data/zch/quantization_experiments/qwen3_self_delta_g2048_20260922_222333/gpu.lock'
with gpu_slot(lock,lambda **kw:None):
 model=get_model(model_path).eval().cuda();model.config.use_cache=False;model.requires_grad_(False)
 accum={};handles=[]
 for l,layer in enumerate(model.model.layers):
  for name,suffix in LLAMA_MODULES.items():
   m=layer.get_submodule(suffix);m.weight.requires_grad_(True);key=f'{l}.{name}';accum[key]=torch.zeros_like(m.weight,dtype=torch.float32)
   def hook(grad,key=key):
    with torch.no_grad():accum[key].add_(grad.float().square())
    return grad
   handles.append(m.weight.register_hook(hook))
 for n,x in enumerate(tqdm(tokens,desc='Fisher calibration',mininterval=5)):
  model.zero_grad(set_to_none=True);x=x[None].cuda();loss=model(input_ids=x,labels=x,use_cache=False).loss
  if not torch.isfinite(loss):raise ValueError('Nonfinite calibration loss')
  loss.backward();manifest.update(completed_samples=n+1,last_loss=float(loss.detach()));save();del loss,x
 model.zero_grad(set_to_none=True)
 for h in handles:h.remove()
 for l in range(36):
  chunk={name:(accum[f'{l}.{name}']/len(tokens)).cpu() for name in LLAMA_MODULES}
  if any(not torch.isfinite(v).all() or (v<0).any() for v in chunk.values()):raise ValueError('Invalid Fisher')
  path=root/f'layer_{l}.pt';tmp=root/f'layer_{l}.pt.tmp'
  with tmp.open('wb') as f:torch.save(chunk,f);f.flush();os.fsync(f.fileno())
  tmp.replace(path)
 del model,accum;gc.collect();torch.cuda.empty_cache()
manifest.update(status='complete',finished_at=time.strftime('%F %T'));save();print('FISHER COMPLETE',flush=True)
