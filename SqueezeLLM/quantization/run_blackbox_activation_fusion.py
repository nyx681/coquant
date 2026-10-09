"""Black-box no-scale endpoint fusion using cached FP-input mean-absolute activations.
Reads original weights, endpoint centers, and independent activation statistics.
No AWQ scale, delta, or quantizer-specific importance enters fusion.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,gc,hashlib,json,pickle,time,traceback
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from group_center_selection import select_centers,assign
from run_llama1_activation_fusion import fill_weight,digest_arrays,write_json
from chunk_gptq_params import LLAMA_MODULES
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens
MODEL='/mnt/dfq/pretrained_model/Llama-7b'
ORIGINAL=Path('/mnt/dfq/squeezellm_awq_delta/original_chunks')
OLD_A=Path('/home/zch/project/Quantification/llama1_w4_g128_center_fusion/full_20260914/awq')
OLD_G=Path('/home/zch/project/Quantification/llama1_activation_nonuniform_g128_20260919_102506')
BASELINE=Path('/mnt/dfq/llama1_dual_nonuniform_g128_20260916/four_ppl.json')

def select_row(task):
 x,ca,cg,alpha=task;book=np.empty_like(ca);cost=np.zeros(3,np.float64)
 for g in range(len(ca)):
  values=x[g*128:(g+1)*128];weights=alpha[g*128:(g+1)*128];candidate=np.r_[ca[g],cg[g]]
  c=select_centers(values,weights,candidate).astype(np.float32)
  if not np.isin(c,candidate).all():raise AssertionError('Selected center not in union')
  losses=np.array([assign(values,weights,v)[1] for v in [ca[g],cg[g],c]])
  if losses[2]>min(losses[:2])+1e-10+1e-9*max(losses):raise AssertionError('Activation-weighted SSE exceeds endpoint')
  book[g]=c;cost+=losses
 return book,cost

def wait_source(path,state,root,run):
 while not path.exists():
  d=json.loads(state.read_text()) if state.exists() else {}
  jobs_path=state.parent.parent/'jobs.json'
  if jobs_path.exists():
   jobs=json.loads(jobs_path.read_text());job=jobs.get(state.parent.name,{})
   if job.get('pid') and not Path(f'/proc/{job["pid"]}').exists():raise RuntimeError(f'Candidate producer exited before writing {path}')
  if d.get('status') in ['failed','complete','stopped_for_cache_reuse']:
   raise RuntimeError(f'Missing codebook {path}; source status={d.get("status")}')
  run.update(status='waiting_for_candidate',waiting_for=str(path));write_json(root/'run.json',run)
  print('Waiting for existing candidate producer',path,flush=True);time.sleep(30)
 return path

def load_books(a,l,name,root,run):
 if a.source_mode=='with_scale':
  p=OLD_A/'lut'/f'l{l}.pkl'
  with p.open('rb') as f:ca=np.asarray(pickle.load(f)[name]['centers'])
  q=wait_source(OLD_G/'centers'/f'{l}.{name}.npz',OLD_G/'run.json',root,run)
  with np.load(q,allow_pickle=False) as z:cg=z['books'][0];upstream_fingerprint=str(z['fingerprint'])
 else:
  source=Path(a.endpoint_root);p=wait_source(source/'awq_delta/centers'/f'{l}.{name}.npz',source/'awq_delta/run.json',root,run)
  q=wait_source(source/'gptq_delta/centers'/f'{l}.{name}.npz',source/'gptq_delta/run.json',root,run)
  with np.load(p,allow_pickle=False) as z:ca=z['centers']
  with np.load(q,allow_pickle=False) as z:cg=z['centers'];upstream_fingerprint=str(z['fingerprint'])
 return ca,cg,dict(awq=str(p),gptq=str(q),gptq_upstream_fingerprint=upstream_fingerprint)

def comparison(a,root,result=None):
 if a.source_mode=='with_scale':
  old=json.loads(BASELINE.read_text());values={x['variant']:x['ppl'] for x in old}
  rows=[dict(method='AWQ-delta endpoint (with scale)',ppl=values['awq'],source=str(BASELINE)),dict(method='GPTQ-delta endpoint (with scale)',ppl=values['gptq_delta'],source=str(BASELINE)),dict(method='Previous weighted fusion',ppl=values['selected'],source=str(BASELINE))]
 else:
  rows=[]
  for v in ['awq_delta','gptq_delta']:
   p=Path(a.endpoint_root)/v/'ppl.json';rows.append(dict(method=v+' endpoint (no scale)',ppl=json.loads(p.read_text())['ppl'] if p.exists() else None,source=str(p)))
 rows.append(dict(method='Black-box activation-weighted fusion',ppl=result['ppl'] if result else None,source=str(root/'ppl.json')))
 write_json(root/'comparison.json',dict(model=MODEL,bit=4,group_size=128,source_mode=a.source_mode,rows=rows))
 lines=['| 方法 | PPL ↓ |','|---|---:|']
 for row in rows:lines.append('| '+row['method']+' | '+('运行中' if row['ppl'] is None else f'{row["ppl"]:.6f}')+' |')
 (root/'comparison.md').write_text('\n'.join(lines)+'\n')

def execute(a,root,run):
 torch.set_num_threads(1)
 cache=torch.load(a.activation_cache,map_location='cpu')
 if cache['model']!=MODEL or cache['statistic']!='mean_abs_input_fp_original' or len(cache['alpha'])!=224:raise ValueError('Invalid activation cache')
 if any(n!=cache['calibration']['tokens'] for n in cache['counts'].values()):raise ValueError('Activation token count mismatch')
 alpha={k:v.float().numpy() for k,v in cache['alpha'].items()}
 if any(not np.isfinite(v).all() or np.any(v<0) for v in alpha.values()):raise ValueError('Invalid activation weights')
 if a.source_mode=='no_scale':
  for variant in ['awq_delta','gptq_delta']:
   metadata=json.loads((Path(a.endpoint_root)/variant/'run.json').read_text())
   for key,expected in dict(model=MODEL,bit=4,group_size=128,AWQ_scale=False,variant=variant).items():
    if metadata.get(key)!=expected:raise ValueError('Candidate provenance mismatch: '+key)
 if a.smoke:
  w=torch.load(ORIGINAL/'layer_0.pt',map_location='cpu')['q'].float().numpy();ca,cg,_=load_books(a,0,'q',root,run)
  for i in range(2):
   c,cost=select_row((w[i,:256],ca[i,:2],cg[i,:2],alpha['0.q'][:256]));assert c.shape==(2,16);assert cost[2]<=min(cost[:2])+1e-8
  print('SMOKE PASSED: 4 real groups; union membership; activation-weighted SSE <= both endpoints',flush=True);return
 model=get_model(MODEL).eval();metrics={};centers_dir=root/'centers';centers_dir.mkdir(exist_ok=True)
 comparison(a,root);workers=a.workers;pool=get_context('spawn').Pool(workers)
 try:
  for l,layer in enumerate(model.model.layers):
   original=torch.load(ORIGINAL/f'layer_{l}.pt',map_location='cpu')
   for name,suffix in LLAMA_MODULES.items():
    if a.boost_after_pid and workers!=a.boost_workers and not Path(f'/proc/{a.boost_after_pid}').exists():
     pool.close();pool.join();workers=a.boost_workers;pool=get_context('spawn').Pool(workers)
    key=f'{l}.{name}';start=time.monotonic();module=layer.get_submodule(suffix)
    if not torch.equal(module.weight.detach().cpu(),original[name].to(module.weight.dtype)):raise ValueError('Original mismatch')
    w=original[name].float().numpy();ca,cg,provenance=load_books(a,l,name,root,run)
    shape=(len(w),w.shape[1]//128,16)
    if ca.shape!=shape or cg.shape!=shape or not np.isfinite(ca).all() or not np.isfinite(cg).all():raise ValueError('Invalid candidate shape/data')
    if alpha[key].shape!=(w.shape[1],):raise ValueError('Activation shape mismatch')
    fingerprint=digest_arrays(w,ca,cg,alpha[key]);path=centers_dir/f'{key}.npz';reused=path.exists()
    run.update(status='selecting_centers',current_module=key,completed_modules=len(metrics),workers=workers);run.pop('waiting_for',None);write_json(root/'run.json',run)
    if reused:
     with np.load(path,allow_pickle=False) as z:
      if str(z['fingerprint'])!=fingerprint:raise ValueError('Checkpoint fingerprint mismatch')
      centers,cost=z['centers'],z['cost']
    else:
     centers=np.empty(shape,np.float32);cost=np.zeros(3,np.float64)
     tasks=((w[i],ca[i],cg[i],alpha[key]) for i in range(len(w)))
     for i,(c,error) in enumerate(tqdm(pool.imap(select_row,tasks,chunksize=4),total=len(w),desc=key,mininterval=15)):
      centers[i]=c;cost+=error
     tmp=Path(str(path)+'.tmp')
     with tmp.open('wb') as f:np.savez(f,centers=centers,cost=cost,fingerprint=np.asarray(fingerprint))
     tmp.replace(path)
    fill_weight(module.weight,w,centers)
    metrics[key]=dict(seconds=time.monotonic()-start,groups=w.size//128,activation_weighted_sse=dict(zip(['awq_endpoint_nearest','gptq_endpoint_nearest','selected'],cost.tolist())),sources=provenance,candidate_fingerprint=fingerprint,reused=reused)
    run.update(completed_modules=len(metrics),last_module=key);write_json(root/'metrics.json',metrics);write_json(root/'run.json',run)
    print('Completed',key,round(metrics[key]['seconds'],1),flush=True)
 finally:pool.close();pool.join()
 del layer,module,w,original,ca,cg,centers;gc.collect();torch.set_num_threads(2)
 run['status']='evaluating';write_json(root/'run.json',run);tokens,path=get_awq_ppl_tokens('wikitext',MODEL)
 with torch.no_grad():ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
 result=dict(variant='blackbox_activation_selected',ppl=ppl,model=MODEL,bit=4,group_size=128,dataset='wikitext2_test',dataset_path=path,tokens=tokens.numel(),block_size=2048,batch_size=8,objective='sum_j mean_t(abs(X_tj)) * min_c (W_ij-c)**2',fusion_AWQ_scale=False,fusion_delta=False,fusion_activation=True,source_mode=a.source_mode,run=str(root/'run.json'))
 write_json(root/'ppl.json',result)
 if a.source_mode=='no_scale':
  for variant in ['awq_delta','gptq_delta']:
   endpoint=Path(a.endpoint_root)/variant;wait_source(endpoint/'ppl.json',endpoint/'run.json',root,run)
 comparison(a,root,result);run.update(status='complete',finished_at=time.strftime('%F %T'));write_json(root/'run.json',run)
 print('PPL',ppl,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--activation-cache',default='/home/zch/project/Quantification/llama1_activation_nonuniform_g128_20260919_102506/activation_mean_abs.pt');p.add_argument('--output',required=True);p.add_argument('--source-mode',choices=['with_scale','no_scale'],default='with_scale');p.add_argument('--endpoint-root');p.add_argument('--workers',type=int,default=4);p.add_argument('--boost-after-pid',type=int);p.add_argument('--boost-workers',type=int,default=8);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.source_mode=='no_scale' and not a.endpoint_root:p.error('no_scale requires endpoint-root')
 if min(a.workers,a.boost_workers)<1:p.error('Positive workers required')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=a.resume or a.smoke)
 codepaths=[Path(__file__),Path(__file__).with_name('group_center_selection.py'),Path(__file__).with_name('run_llama1_activation_fusion.py')]
 spec=dict(model=MODEL,bit=4,group_size=128,total_modules=224,source_mode=a.source_mode,endpoint_root=a.endpoint_root,
 objective='sum_j mean_t(abs(X_tj))*min_c (W_ij-c)**2',sample_weight='mean_abs_input_fp_original',activation_cache=a.activation_cache,activation_cache_sha256=hashlib.sha256(Path(a.activation_cache).read_bytes()).hexdigest(),fusion_AWQ_scale=False,fusion_delta=False,fusion_activation=True,
 candidate_source_AWQ_scale=a.source_mode=='with_scale',code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in codepaths})
 if a.resume:
  old=json.loads((root/'run.json').read_text())
  for k,v in spec.items():
   if old.get(k)!=v:raise ValueError('Resume mismatch: '+k)
  if old['status']=='complete':print('Already complete');return
 run=dict(spec,status='starting',started_at=time.strftime('%F %T'),workers=a.workers)
 if not a.smoke:write_json(root/'run.json',run)
 try:execute(a,root,run)
 except BaseException as e:
  if not a.smoke:run.update(status='failed',error=repr(e));write_json(root/'run.json',run)
  traceback.print_exc();raise
if __name__=='__main__':main()
