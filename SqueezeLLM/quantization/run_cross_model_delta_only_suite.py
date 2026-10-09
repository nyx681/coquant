"""Direct S=N(d) AWQ/GPTQ runs on LLaMA-2 and OPT; no first-round dependency.
The user paused S=1+N(d); those partial records remain read-only in the table.
"""
import argparse,csv,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
BASELINES={
 'llama1':Path('/mnt/dfq/llama1_dual_nonuniform_g128_20260916/four_ppl.json'),
 'llama2':Path('/mnt/dfq/llama2_7b_dual_nonuniform_g128_20260917_reuse_awq/three_ppl.json'),
 'opt':Path('/mnt/dfq/opt67b_dual_nonuniform_g128_20260917/three_ppl.json')}
PAUSED_ROOT=Path('/mnt/data/zch/quantization_experiments/llama2_opt_delta_ablation_20260920_095209')
MODELS={'llama1':'/mnt/dfq/pretrained_model/Llama-7b','llama2':'/mnt/dfq/pretrained_model/llama-2-7b','opt':'/mnt/dfq/pretrained_model/opt-6.7b'}
TITLES={'llama1':'LLaMA-1-7B','llama2':'LLaMA-2-7B','opt':'OPT-6.7B'}
SCRIPTS={'keep_one':'run_delta_without_awq_scale.py','remove_one':'run_delta_without_scale_or_offset.py'}
OBJECTIVES={'keep_one':'sum_j (1+N(d)_j)*(W_ij-center[label_ij])**2','remove_one':'sum_j N(d)_j*(W_ij-center[label_ij])**2'}
OLD_ROOTS={'keep_one':Path('/mnt/data/zch/quantization_experiments/llama1_delta_no_awq_scale_20260919_150656'),'remove_one':Path('/mnt/data/zch/quantization_experiments/llama1_delta_no_scale_no_offset_20260919_172224')}

def load(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,d):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def validate_result(d,target,variant,stage):
 expected=dict(model=MODELS[target],variant=variant,bit=4,group_size=128,block_size=2048,batch_size=8,AWQ_scale=False,objective=OBJECTIVES[stage])
 for k,v in expected.items():
  if d.get(k)!=v:raise ValueError('Result mismatch '+target+' '+variant+' '+stage+' '+k)
 if not isinstance(d.get('ppl'),(int,float)):raise ValueError('Missing numeric PPL')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--max-parallel',type=int,default=4);a=p.parse_args()
 if a.workers<1 or not 1<=a.max_parallel<=4:p.error('Positive workers and max-parallel between 1 and 4 required')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();(root/'scripts').mkdir()
 baseline={};manifest={}
 for target,path in BASELINES.items():
  values=load(path)
  baseline[target]={r['variant']:r['ppl'] for r in values}
  manifest[target]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot=values)
 for stage,path in OLD_ROOTS.items():
  for variant in ['awq_delta','gptq_delta']:validate_result(load(path/variant/'ppl.json'),'llama1',variant,stage)
 for name in [SCRIPTS['remove_one']]:
  src=Path(__file__).parent/name;(root/'scripts'/name).write_bytes(src.read_bytes())
 write(root/'reused_baselines.json',manifest)
 run=dict(status='running',started_at=time.strftime('%F %T'),targets=['llama2','opt'],stages={'remove_one':OBJECTIVES['remove_one']},skipped_stage='keep_one',workers_per_job=a.workers,max_parallel=a.max_parallel,total_new_experiments=4,fusion='not scheduled')
 write(root/'run.json',run)
 jobs={};processes={}
 for stage in ['remove_one']:
  for target in ['llama2','opt']:
   for variant in ['awq_delta','gptq_delta']:
    key=f'{target}_{stage}_{variant}';jobs[key]=dict(target=target,stage=stage,variant=variant,status='pending',output=str(root/target/stage/variant))
 def aggregate():
  rows=[]
  for target in ['llama1','llama2','opt']:
   for variant in ['awq_delta','gptq_delta']:
    bkey='awq' if target=='llama1' and variant=='awq_delta' else variant
    row=dict(model=TITLES[target],method=variant,with_scale=baseline[target][bkey],baseline_source=str(BASELINES[target]))
    for stage in ['keep_one','remove_one']:
     out=OLD_ROOTS[stage]/variant if target=='llama1' else (PAUSED_ROOT if stage=='keep_one' else root)/target/stage/variant
     result=load(out/'ppl.json');state=load(out/'run.json');job=jobs.get(f'{target}_{stage}_{variant}',{})
     if result:validate_result(result,target,variant,stage)
     row[stage]=result.get('ppl');row[stage+'_status']=job['status'] if job.get('status') in ['failed','blocked'] else state.get('status',job.get('status','not_scheduled'))
     row[stage+'_completed_modules']=state.get('completed_modules');row[stage+'_source']=str(out/'ppl.json')
    row['difference_remove_minus_keep']=row['remove_one']-row['keep_one'] if row['remove_one'] is not None and row['keep_one'] is not None else None
    rows.append(row)
  write(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),protocol='W4-G128 WikiText-2 test block_size=2048 batch_size=8',rows=rows))
  with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  lines=['| 模型 | 方法 | 原版含 scale | 去 scale：1+N(d) | 再去1+：N(d) | 状态（第一轮 / 第二轮） |','|---|---|---:|---:|---:|---|']
  for r in rows:
   fmt=lambda x:'—' if x is None else f'{x:.6f}'
   lines.append('| '+' | '.join([r['model'],r['method'],fmt(r['with_scale']),fmt(r['keep_one']),fmt(r['remove_one']),r['keep_one_status']+' / '+r['remove_one_status']])+' |')
  lines.append('\nLLaMA-1 两轮及含 scale 基线复用。LLaMA-2、OPT 的1+N(d)按用户要求暂停；现在直接运行两模型的N(d)，共四项，不启动融合。')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
 try:
  while True:
   for key,child in list(processes.items()):
    code=child.poll()
    if code is None:continue
    j=jobs[key];state=load(Path(j['output'])/'run.json');result=load(Path(j['output'])/'ppl.json')
    if code==0 and state.get('status')=='complete':validate_result(result,j['target'],j['variant'],j['stage']);j['status']='complete'
    else:j.update(status='failed',error=state.get('error','Child exited before complete result'))
    j.update(exit_code=code,finished_at=time.strftime('%F %T'));del processes[key]
   for key,j in jobs.items():
    if j['status']!='pending':continue
    if len(processes)>=a.max_parallel:continue
    script=Path(__file__).parent/SCRIPTS[j['stage']]
    command=[os.sys.executable,'-u',str(script),'--target',j['target'],'--variant',j['variant'],'--output',j['output'],'--workers',str(a.workers)]
    log=root/'logs'/(key+'.log')
    with log.open('xb') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    j.update(status='running',pid=child.pid,command=command,log=str(log),started_at=time.strftime('%F %T'));processes[key]=child
    print('Launched',key,child.pid,flush=True)
   write(root/'jobs.json',jobs);aggregate()
   if all(j['status'] in ['complete','failed','blocked'] for j in jobs.values()):break
   time.sleep(30)
  run.update(status='complete' if all(j['status']=='complete' for j in jobs.values()) else 'failed',finished_at=time.strftime('%F %T'));write(root/'run.json',run)
 except BaseException as e:
  run.update(status='failed',error=repr(e));write(root/'run.json',run);traceback.print_exc();raise
if __name__=='__main__':main()
