"""C4 evaluation only: reuse eight completed Qwen/Mistral W4 S=N(d) center checkpoints."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens
from squeezellm.model_parse import get_layers
from chunk_gptq_params import LLAMA_MODULES,OPT_MODULES

from run_llama1_activation_fusion import fill_weight,write_json,atomic_torch_save
DATA=Path('/mnt/data/zch/quantization_experiments')
VARIANTS=['awq_delta','gptq_delta','unit','activation']
NAMES=['AWQ-delta：去 scale、去1+','GPTQ-delta：去 scale、去1+','精确32选16：系数1','精确32选16：系数activation']

SOURCE_ROOT=DATA/'extended_five_models_delta_only_20260921_090804'
TARGETS={tag:json.loads((SOURCE_ROOT/tag/'target.json').read_text()) for tag in ('qwen3_8b_base','mistral7b')}

def sources(target):
 base=SOURCE_ROOT/target
 return [base/'endpoints/awq_delta',base/'endpoints/gptq_delta',base/'fusion/unit',base/'fusion/activation']

def load(p):return json.loads(p.read_text()) if p.exists() else {}

def worker(a):
 root=Path(a.output)/a.worker;root.mkdir(exist_ok=True);cfg=TARGETS[a.worker]
 modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
 state=dict(status='preflight',model=cfg['model'],started_at=time.strftime('%F %T'),dataset='c4',bit=4,group_size=128,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy='reuse saved centers; no KMeans, fusion, refinement or calibration',pid=os.getpid())
 def update(**kw):state.update(kw,updated_at=time.strftime('%F %T'));write_json(root/'run.json',state)
 try:
  update();torch.set_num_threads(2)
  provenance={}
  for variant,source in zip(VARIANTS,sources(a.worker)):
   meta=load(source/'run.json');result=load(source/'ppl.json')
   if meta.get('status')!='complete':raise ValueError('Source incomplete: '+str(source))
   for k,v in dict(model=cfg['model'],bit=4,group_size=128,block_size=2048,batch_size=8).items():
    if result.get(k)!=v:raise ValueError('Source protocol mismatch: '+str(source)+'/'+k)
   files=list((source/'centers').glob('*.npz'))
   if len(files)!=cfg['layers']*len(modules):raise ValueError('Incomplete center modules')
   if variant in ('awq_delta','gptq_delta') and (result.get('AWQ_scale') is not False or result.get('importance_offset')!=0):
    raise ValueError('Requires S=N(d) endpoints')
   provenance[variant]=dict(source=str(source),wikitext_ppl=result['ppl'],source_run_sha256=hashlib.sha256((source/'run.json').read_bytes()).hexdigest())
  write_json(root/'sources.json',provenance)
  update(status='preparing_c4_tokens')
  token_path=root/'c4_tokens.pt'
  if token_path.exists():
   cached=torch.load(token_path,map_location='cpu');assert cached['model']==cfg['model'];tokens=cached['tokens'];dataset_path=cached['dataset_path']
  else:
   tokens,dataset_path=get_awq_ppl_tokens('c4',cfg['model'],c4_num_proc=2)
   atomic_torch_save(dict(model=cfg['model'],tokens=tokens,dataset_path=dataset_path),token_path)
  token_hash=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
  info=dict(dataset_path=dataset_path,tokens=tokens.numel(),evaluated_tokens=tokens.numel()//2048*2048,
            dropped_tokens=tokens.numel()%2048,blocks=tokens.numel()//2048,block_size=2048,batch_size=8,
            token_sha256=token_hash,tokenizer='use_fast=False',sampling='all documents in local C4 validation shard, in stored order')
  write_json(root/'evaluation_protocol.json',info)
  update(status='loading_model');model=get_model(cfg['model']).eval()
  for variant,source in zip(VARIANTS,sources(a.worker)):
   out=root/variant;out.mkdir(exist_ok=True)
   if load(out/'run.json').get('status')=='complete':continue
   update(status='reconstructing',current_variant=variant,completed_modules=0)
   write_json(out/'run.json',dict(status='reconstructing',source=str(source)))
   model.cpu();gc.collect();torch.cuda.empty_cache();fingerprints={}
   for l,layer in enumerate(get_layers(model,cfg['model_type'])):
    original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
    for name,suffix in modules.items():
     w=original[name].float().numpy();module=layer.get_submodule(suffix)
     if variant=='awq_delta' and not torch.equal(module.weight.detach().cpu(),original[name].to(module.weight.dtype)):
      raise ValueError('Original model/chunk mismatch')
     key=f'{l}.{name}'
     with np.load(source/'centers'/f'{key}.npz',allow_pickle=False) as z:
      centers=z['centers'];fingerprints[key]=str(z['fingerprint'])
      if not np.isfinite(centers).all():raise ValueError('Nonfinite center')
      fill_weight(module.weight,w,centers)
    update(completed_modules=(l+1)*len(modules))
   write_json(out/'center_fingerprints.json',fingerprints)
   del original,w,module,layer,centers;gc.collect()
   update(status='waiting_for_gpu',current_variant=variant)
   while torch.cuda.mem_get_info()[0]<(60 if a.worker=='qwen3_8b_base' else 32)*1024**3:time.sleep(15)
   update(status='evaluating',current_variant=variant)
   write_json(out/'run.json',dict(status='evaluating',source=str(source),started_at=time.strftime('%F %T')))
   with torch.no_grad():ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'c4',block_size=2048,batch_size=8)
   result=dict(info,model=cfg['model'],variant=variant,ppl=ppl,bit=4,group_size=128,dataset='c4_validation_local_shard',
               base_dtype=str(next(model.parameters()).dtype),source=str(source),source_wikitext_ppl=provenance[variant]['wikitext_ppl'],
               candidate_AWQ_scale=False,candidate_importance_offset=0,recomputed_centers=False,center_refinement=False)
   write_json(out/'ppl.json',result);write_json(out/'run.json',dict(status='complete',finished_at=time.strftime('%F %T'),source=str(source)))
   print('C4 PPL',a.worker,variant,ppl,flush=True)
   model.cpu();gc.collect();torch.cuda.empty_cache()
  update(status='complete',finished_at=time.strftime('%F %T'))
 except BaseException as e:update(status='failed',error=repr(e));traceback.print_exc();raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',choices=['qwen3_8b_base','mistral7b']);a=p.parse_args()
 if a.worker:return worker(a)
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 write_json(root/'run.json',dict(status='running',started_at=time.strftime('%F %T'),max_parallel_models=2,evaluation_only=True))
 jobs={};procs={};queue=['qwen3_8b_base','mistral7b']
 while True:
  active=sum(c.poll() is None for c in procs.values())
  while queue and active<2:
   target=queue.pop(0);cmd=[os.sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(root),'--worker',target]
   log=root/(target+'.log')
   with log.open('xb') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   procs[target]=child;jobs[target]=dict(pid=child.pid,command=cmd,log=str(log),output=str(root/target));write_json(root/'jobs.json',jobs);active+=1
  rows=[]
  for target,label in [('qwen3_8b_base','Qwen3-8B-Base'),('mistral7b','Mistral-7B-v0.1')]:
   state=load(root/target/'run.json')
   for variant,name in zip(VARIANTS,NAMES):
    out=root/target/variant;result=load(out/'ppl.json');vstate=load(out/'run.json')
    rows.append(dict(model=label,method=name,ppl=result.get('ppl'),status=vstate.get('status',state.get('status','queued')),source=str(out/'ppl.json')))
  write_json(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),dataset='C4 local validation shard',rows=rows))
  lines=['| 方法 | Qwen3-8B-Base | Mistral-7B-v0.1 |','|---|---:|---:|']
  for i,name in enumerate(NAMES):
   cells=[f'{rows[i+j*4]["ppl"]:.6f}' if rows[i+j*4]['ppl'] is not None else '—' for j in range(2)]
   lines.append('| '+' | '.join([name]+cells)+' |')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
  if not queue and all(c.poll() is not None for c in procs.values()):
   failed={t:c.returncode for t,c in procs.items() if c.returncode!=0}
   write_json(root/'run.json',dict(status='failed' if failed else 'complete',failures=failed,finished_at=time.strftime('%F %T'),evaluation_only=True));break
  time.sleep(20)
if __name__=='__main__':main()
