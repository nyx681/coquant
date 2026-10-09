"""W4-G128 second ablation: S=N(d), no AWQ-scale prior and no additive 1.
Kept separate from the running 1+N(d) experiment to preserve provenance.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,gc,hashlib,json,time,traceback,shutil
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from nuq import build_inverse_channel_relative_weights,normalize_score_row
from run_rtn_delta_comparison import row_delta as quantize_row
from chunk_gptq_params import LLAMA_MODULES,OPT_MODULES
from run_llama1_activation_fusion import fill_weight,digest_arrays,write_json
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.model_parse import get_layers
from squeezellm.datautils import get_awq_ppl_tokens
def importance_delta_only(w,q):
 d=build_inverse_channel_relative_weights(w,np.abs(w-q),gamma=1,min_weight=.1)
 s=normalize_score_row(d)
 if s.shape!=(w.shape[1],) or not np.isfinite(s).all() or np.any(s<=0):raise ValueError('Invalid importance')
 return s

def pid_identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None

def metadata(a):
 cfg=json.loads(Path(a.target_config).read_text());base=Path(__file__).parent
 files=[Path(__file__)]+[base/x for x in ['nuq.py','group_center_selection.py','run_llama1_activation_fusion.py','run_rtn_delta_comparison.py']]+[base.parent/'llama.py',base.parent/'squeezellm/datautils.py']
 inputs=[Path(cfg['model'])/'config.json']+[Path(cfg['original'])/f'layer_{l}.pt' for l in range(cfg['layers'])]
 inputs += [Path(cfg['awq'] if a.variant=='awq_delta' else cfg['gptq'])/f'layer_{l}.pt' for l in range(cfg['layers'])]
 return dict(target=a.target,variant=a.variant,model=cfg['model'],bit=4,group_size=128,AWQ_scale=False,
 objective='sum_j N(d)_j*(W_ij-center[label_ij])**2',importance_offset=0,delta=dict(aggregation='module input channels',gamma=1,min_weight=.1,lambda_=1),
 kmeans=dict(random_state=0,n_init='auto',max_iter=50),evaluation=dict(dataset='wikitext2_test',block_size=2048,batch_size=8),
 code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
 source_stat={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs})

def execute(a,root,run):
 cfg=json.loads(Path(a.target_config).read_text());modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
 torch.set_num_threads(1)
 if a.smoke:
  original=torch.load(Path(cfg['original'])/'layer_0.pt',map_location='cpu')
  awq=torch.load(Path(cfg['awq'] if a.variant=='awq_delta' else cfg['gptq'])/'layer_0.pt',map_location='cpu')
  name,suffix=next(iter(modules.items()));w=original[name].float().numpy()
  q=awq[name].float().numpy()
  s=importance_delta_only(w,q);x=w[:2,:256].copy();out=torch.zeros_like(torch.from_numpy(x))
  books=np.stack([quantize_row((row,s[:256]))[0] for row in x]);fill_weight(out,x,books)
  assert np.isfinite(out.numpy()).all() and books.shape==(2,2,16)
  print('SMOKE PASSED',a.variant,dict(groups=4,importance_min=float(s.min()),importance_max=float(s.max())),flush=True);return
 model=get_model(cfg['model']).eval();layers=get_layers(model,cfg['model_type']);expected=len(layers)*len(modules)
 centers_dir=root/'centers';centers_dir.mkdir(exist_ok=True)
 needed=sum(layer.get_submodule(s).weight.numel() for layer in layers for s in modules.values())*.5
 existing=sum(p.stat().st_size for p in centers_dir.glob('*.npz'))
 if shutil.disk_usage(root).free<max(needed-existing,0)+1024**3:raise OSError('Insufficient checkpoint space')
 metrics={};workers=a.workers;pool=get_context('spawn').Pool(workers)
 try:
  for l,layer in enumerate(layers):
   original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
   awq=torch.load(Path(cfg['awq'] if a.variant=='awq_delta' else cfg['gptq'])/f'layer_{l}.pt',map_location='cpu')
   for name,suffix in modules.items():
    if a.boost_after_pid and workers!=a.boost_workers and pid_identity(a.boost_after_pid)!=run.get('boost_pid_identity'):
     pool.close();pool.join();workers=a.boost_workers;pool=get_context('spawn').Pool(workers);print('Worker boost',workers,flush=True)
    start=time.monotonic();key=f'{l}.{name}';module=layer.get_submodule(suffix)
    run.update(status='building_centers',current_module=key,completed_modules=len(metrics),total_modules=expected,workers=workers);write_json(root/'run.json',run)
    if not torch.equal(module.weight.detach().cpu(),original[name].to(module.weight.dtype)):raise ValueError('Original mismatch: '+key)
    w=original[name].float().numpy()
    q=awq[name].float().numpy()
    if w.shape!=q.shape or not np.isfinite(q).all():raise ValueError('Invalid reference')
    s=importance_delta_only(w,q);fingerprint=digest_arrays(w,s);checkpoint=centers_dir/f'{key}.npz';reused=checkpoint.exists()
    if reused:
     with np.load(checkpoint,allow_pickle=False) as z:
      if str(z['fingerprint'])!=fingerprint:raise ValueError('Checkpoint fingerprint mismatch')
      centers,cost=z['centers'],z['cost']
    else:
     centers=np.empty((len(w),w.shape[1]//128,16),np.float32);cost=np.zeros(2,np.float64)
     tasks=((w[r],s) for r in range(len(w)))
     for r,(book,loss) in enumerate(tqdm(pool.imap(quantize_row,tasks,chunksize=4),total=len(w),desc=a.variant+' '+key,mininterval=15)):
      centers[r]=book;cost+=loss
     tmp=Path(str(checkpoint)+'.tmp')
     with tmp.open('wb') as fh:np.savez(fh,centers=centers,cost=cost,fingerprint=np.asarray(fingerprint))
     tmp.replace(checkpoint)
    if not np.isfinite(centers).all():raise ValueError('Nonfinite centers')
    fill_weight(module.weight,w,centers)
    metrics[key]=dict(seconds=time.monotonic()-start,weighted_sse=float(cost[0]),weight_sse=float(cost[1]),reused=reused,checkpoint=str(checkpoint),importance_min=float(s.min()),importance_max=float(s.max()))
    run.update(status='building_centers',completed_modules=len(metrics),total_modules=expected,last_module=key,workers=workers)
    write_json(root/'metrics.json',metrics);write_json(root/'run.json',run);print('Completed',a.target,a.variant,key,round(metrics[key]['seconds'],1),flush=True)
 finally:pool.close();pool.join()
 del layers,layer,module,original,awq,w,q,centers;gc.collect()
 run['status']='evaluating';write_json(root/'run.json',run);torch.set_num_threads(2)
 tokens,path=get_awq_ppl_tokens('wikitext',cfg['model'])
 from run_cross_model_delta_only_fusions import gpu_slot
 with gpu_slot(a.gpu_lock, lambda **kw: None):
  with torch.no_grad():ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
 result=dict(base_dtype=str(next(model.parameters()).dtype),variant=a.variant,model=cfg['model'],ppl=ppl,bit=4,group_size=128,dataset='wikitext2_test',dataset_path=path,block_size=2048,batch_size=8,tokens=tokens.numel(),AWQ_scale=False,importance_offset=0,objective=run['objective'],run=str(root/'run.json'))
 write_json(root/'ppl.json',result)
 run.update(status='complete',finished_at=time.strftime('%F %T'));write_json(root/'run.json',run);print('PPL',ppl,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--target-config',required=True);p.add_argument('--gpu-lock',required=True);p.add_argument('--variant',choices=['awq_delta','gptq_delta'],required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--boost-after-pid',type=int);p.add_argument('--boost-workers',type=int,default=12);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if min(a.workers,a.boost_workers)<1:p.error('Positive workers required')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=a.resume or a.smoke);spec=metadata(a)
 if a.smoke:execute(a,root,spec);return
 path=root/'run.json'
 if a.resume:
  old=json.loads(path.read_text())
  for k,v in spec.items():
   if old.get(k)!=v:raise ValueError('Resume configuration mismatch: '+k)
  if old.get('status')=='complete':print('Already complete');return
 run=dict(spec,status='starting',started_at=time.strftime('%F %T'),workers=a.workers,boost_after_pid=a.boost_after_pid,boost_workers=a.boost_workers,boost_pid_identity=pid_identity(a.boost_after_pid) if a.boost_after_pid else None)
 write_json(path,run)
 try:execute(a,root,run)
 except BaseException as e:
  run.update(status='failed',error=repr(e));write_json(path,run);traceback.print_exc();raise
if __name__=='__main__':main()
