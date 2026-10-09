"""G2048 self-relative delta ablations and matched controls. No activation/scale prior."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,gc,hashlib,json,time,traceback,shutil
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from nuq import build_inverse_channel_relative_weights,normalize_score_row
from qwen_hierarchical_delta import quantize_row,fill_weight,GROUP_SIZE
from chunk_gptq_params import LLAMA_MODULES,OPT_MODULES
from run_llama1_activation_fusion import digest_arrays,write_json
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.model_parse import get_layers
from squeezellm.datautils import get_awq_ppl_tokens
from qwen_hierarchical_delta import importance,DESIGNS

def pid_identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None

def metadata(a):
 cfg=json.loads(Path(a.target_config).read_text());base=Path(__file__).parent;score_meta=torch.load(a.score_cache,map_location='cpu')['metadata']
 files=[Path(__file__)]+[base/x for x in ['qwen_hierarchical_delta.py','qwen_self_delta_g2048.py','nuq.py','group_center_selection.py','run_llama1_activation_fusion.py','run_rtn_delta_comparison.py']]+[base.parent/'llama.py',base.parent/'squeezellm/datautils.py']
 inputs=[Path(a.score_cache),Path(cfg['model'])/'config.json']+[Path(cfg['original'])/f'layer_{l}.pt' for l in range(cfg['layers'])]
 inputs += [Path(cfg['awq'] if a.variant=='awq_delta' else cfg['gptq'])/f'layer_{l}.pt' for l in range(cfg['layers'])]
 return dict(target=a.target,variant=a.variant,model=cfg['model'],bit=4,group_size=GROUP_SIZE,reference_group_size=128,AWQ_scale=False,
 small_group_size=128,scale_dtype='float32',zero_point=False,scale_optimization='alternating weighted least squares, positive scales, 30 iterations',objective='sum_j S_ij*(W_ij-scale[subgroup_j]*center[label_ij])**2',scope='elementwise' if a.design.startswith('self_') else a.scope,design=a.design,importance_offset=0,activation=True,activation_score=score_meta,importance_multiplier='1+s, s in [0,1]',center_dtype='float32',delta=dict(aggregation='none',normalization_scope='none' if a.design.startswith('self_') else ('row' if a.design=='absolute_inverse_control' else 'none'),formula='eta+(1-eta)*(abs(W)+eps)/(abs(W)+abs(W-Q)+eps)' if a.design.startswith('self_') else a.design,min_weight=({'self_floor05':.5,'self_floor08':.8,'absolute_inverse_control':.1}.get(a.design)),epsilon=float(np.finfo(np.float32).eps)),
 kmeans=dict(random_state=0,n_init='auto',max_iter=50),evaluation=dict(dataset='wikitext2_test',block_size=2048,batch_size=8),
 code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
 source_stat={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs})

def execute(a,root,run):
 cfg=json.loads(Path(a.target_config).read_text());modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
 torch.set_num_threads(1)
 cache=torch.load(a.score_cache,map_location='cpu')
 if cache['model']!=cfg['model']:raise ValueError('Score model mismatch')
 scores=cache['scores']
 expected_keys={f'{l}.{name}' for l in range(cfg['layers']) for name in modules}
 if set(scores)!=expected_keys:raise ValueError('Score module mismatch')
 def multiply_score(s,key):
  a=np.asarray(scores[key],dtype=np.float32)
  if a.shape not in [(s.shape[1],),s.shape]:raise ValueError('Score shape mismatch: '+key)
  if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Score outside [0,1]')
  return s*(1+a)

 if a.smoke:
  original=torch.load(Path(cfg['original'])/'layer_0.pt',map_location='cpu')
  awq=torch.load(Path(cfg['awq'] if a.variant=='awq_delta' else cfg['gptq'])/'layer_0.pt',map_location='cpu')
  name,suffix=next(iter(modules.items()));w=original[name].float().numpy()
  q=awq[name].float().numpy()
  s=multiply_score(importance(w,q,a.scope,a.design),'0.'+name);x=w[:2].copy();out=torch.zeros_like(torch.from_numpy(x))
  fits=[quantize_row((row,s[i])) for i,row in enumerate(x)];books=np.stack([v[0] for v in fits]);scales=np.stack([v[1] for v in fits]);fill_weight(out,x,books,scales)
  assert np.isfinite(out.numpy()).all() and books.shape==(2,x.shape[1]//GROUP_SIZE,16)
  print('SMOKE PASSED',a.variant,dict(groups=int(books.shape[0]*books.shape[1]),importance_min=float(s.min()),importance_max=float(s.max())),flush=True);return
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
    assert w.shape[1]%GROUP_SIZE==0, 'G2048 divisibility';s=multiply_score(importance(w,q,a.scope,a.design),key);fingerprint=digest_arrays(w,s);checkpoint=centers_dir/f'{key}.npz';reused=checkpoint.exists()
    if reused:
     with np.load(checkpoint,allow_pickle=False) as z:
      if str(z['fingerprint'])!=fingerprint:raise ValueError('Checkpoint fingerprint mismatch')
      centers,scales,cost=z['centers'],z['scales'],z['cost']
    else:
     centers=np.empty((len(w),w.shape[1]//GROUP_SIZE,16),np.float32);scales=np.empty_like(centers);cost=np.zeros(2,np.float64)
     tasks=((w[r],s[r]) for r in range(len(w)))
     for r,(book,scale,loss) in enumerate(tqdm(pool.imap(quantize_row,tasks,chunksize=4),total=len(w),desc=a.variant+' '+key,mininterval=15)):
      centers[r]=book;scales[r]=scale;cost+=loss
     tmp=Path(str(checkpoint)+'.tmp')
     with tmp.open('wb') as fh:np.savez(fh,centers=centers,scales=scales,cost=cost,fingerprint=np.asarray(fingerprint));fh.flush();os.fsync(fh.fileno())
     tmp.replace(checkpoint)
    if not np.isfinite(centers).all():raise ValueError('Nonfinite centers')
    fill_weight(module.weight,w,centers,scales)
    metrics[key]=dict(seconds=time.monotonic()-start,weighted_sse=float(cost[0]),weight_sse=float(cost[1]),reused=reused,checkpoint=str(checkpoint),importance_min=float(s.min()),importance_max=float(s.max()))
    run.update(status='building_centers',completed_modules=len(metrics),total_modules=expected,last_module=key,workers=workers)
    write_json(root/'metrics.json',metrics);write_json(root/'run.json',run);print('Completed',a.target,a.variant,key,round(metrics[key]['seconds'],1),flush=True)
 finally:pool.close();pool.join()
 del layers,layer,module,original,awq,w,q,s,centers;gc.collect()
 run['status']='evaluating';write_json(root/'run.json',run);torch.set_num_threads(2)
 tokens,path=get_awq_ppl_tokens('wikitext',cfg['model'])
 from run_cross_model_delta_only_fusions import gpu_slot
 with gpu_slot(a.gpu_lock, lambda **kw: None):
  with torch.no_grad():ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
 result=dict(base_dtype=str(next(model.parameters()).dtype),variant=a.variant,scope=run['scope'],model=cfg['model'],ppl=ppl,design=a.design,bit=4,group_size=GROUP_SIZE,small_group_size=128,reference_group_size=128,dataset='wikitext2_test',dataset_path=path,block_size=2048,batch_size=8,tokens=tokens.numel(),AWQ_scale=False,activation=True,activation_score=run['activation_score'],importance_multiplier=run['importance_multiplier'],importance_offset=0,objective=run['objective'],run=str(root/'run.json'))
 write_json(root/'ppl.json',result)
 run.update(status='complete',finished_at=time.strftime('%F %T'));write_json(root/'run.json',run);print('PPL',ppl,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--score-cache',required=True);p.add_argument('--design',choices=DESIGNS,required=True);p.add_argument('--scope',choices=['module','row'],required=True);p.add_argument('--target',required=True);p.add_argument('--target-config',required=True);p.add_argument('--gpu-lock',required=True);p.add_argument('--variant',choices=['awq_delta','gptq_delta'],required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--boost-after-pid',type=int);p.add_argument('--boost-workers',type=int,default=12);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
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
