"""Empirical Fisher with FP32 CPU accumulation and immediate leaf-grad release."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,importlib.util,gc
from pathlib import Path
import torch
from llama import get_model
from chunk_gptq_params import LLAMA_MODULES,OPT_MODULES
from squeezellm.model_parse import get_layers
from run_cross_model_delta_only_fusions import gpu_slot

def main():
 root=Path(sys.argv[1]);cfg=json.loads(Path(sys.argv[2]).read_text());root.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.manual_seed(0)
 spec=importlib.util.spec_from_file_location('fisher_datautils','/home/zch/project/SqueezeLLM-gradients/datautils.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
 loader,_=data.get_loaders('c4',nsamples=100,seed=0,seqlen=512,model=cfg['model'])
 tokens=torch.cat([v[0] for v in loader]);torch.save(tokens,root/'calibration_tokens.pt')
 state=dict(status='waiting_for_gpu',model=cfg['model'],dataset='C4 train',samples=100,seqlen=512,seed=0,statistic='mean per-sequence squared CE gradient',accumulator_dtype='float32',accumulator_device='cpu',gradient_release='post_accumulate_grad_hook',dropout=False,tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),completed_samples=0,pid=os.getpid())
 def save():
  p=root/'manifest.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'manifest.json')
 save()
 try:
  with gpu_slot('/mnt/data/zch/quantization_experiments/qwen3_self_delta_g2048_20260922_222333/gpu.lock',lambda **kw:None):
   while torch.cuda.mem_get_info()[0]<48*1024**3:time.sleep(10)
   model=get_model(cfg['model']).eval().cuda();model.config.use_cache=False;model.requires_grad_(False)
   layers=get_layers(model,cfg['model_type']);assert len(layers)==cfg['layers']
   mapping=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
   state.update(status='collecting',model_dtype=str(next(model.parameters()).dtype));save()
   accum={};handles=[]
   for l,layer in enumerate(layers):
    for name,suffix in mapping.items():
     w=layer.get_submodule(suffix).weight;w.requires_grad_(True);key=f'{l}.{name}'
     accum[key]=torch.zeros_like(w,device='cpu',dtype=torch.float32)
     def hook(param,key=key):
      with torch.no_grad():
       grad=param.grad.detach().float().cpu()
       if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient '+key)
       accum[key].add_(grad.square());param.grad=None
     handles.append(w.register_post_accumulate_grad_hook(hook))
   for n,x in enumerate(tokens):
    model.zero_grad(set_to_none=True);x=x[None].cuda();loss=model(input_ids=x,labels=x,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();state.update(completed_samples=n+1,last_loss=float(loss.detach()));save()
    print('Fisher',n+1,'/100',float(loss.detach()),flush=True);del x,loss
   for h in handles:h.remove()
   for l in range(cfg['layers']):
    chunk={name:accum.pop(f'{l}.{name}').div_(len(tokens)) for name in mapping}
    if any(not torch.isfinite(v).all() or (v<0).any() for v in chunk.values()):raise ValueError('Invalid Fisher')
    path=root/f'layer_{l}.pt';tmp=path.with_suffix('.pt.tmp')
    with tmp.open('wb') as f:torch.save(chunk,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
   del model,accum;gc.collect();torch.cuda.empty_cache()
  state.update(status='complete',finished_at=time.strftime('%F %T'));save()
 except BaseException as e:state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
