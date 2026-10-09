"""Finish S=1+N(d), then evaluate S=N(d); defer fusion until comparison."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path

def write(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def load(p):return json.loads(p.read_text()) if p.exists() else {}

def main():
 p=argparse.ArgumentParser();p.add_argument('--first-root',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);first=Path(a.first_root)
 baseline_path=Path('/mnt/dfq/llama1_dual_nonuniform_g128_20260916/four_ppl.json')
 baseline={r['variant']:r['ppl'] for r in load(baseline_path)}
 run=dict(status='waiting_for_first_round',started_at=time.strftime('%F %T'),first_root=str(first),second_objective='N(d)',fusion_status='deferred_until_result_comparison',workers_per_endpoint=a.workers)
 write(root/'run.json',run)
 def aggregate():
  rows=[]
  for v,b in [('awq_delta','awq'),('gptq_delta','gptq_delta')]:
   r1=load(first/v/'ppl.json');s1=load(first/v/'run.json');r2=load(root/v/'ppl.json');s2=load(root/v/'run.json')
   rows.append(dict(method=v,with_scale=baseline[b],no_scale_keep_one=r1.get('ppl'),no_scale_remove_one=r2.get('ppl'),
    first_status=s1.get('status','not_started'),second_status=s2.get('status','not_started'),first_completed_modules=s1.get('completed_modules'),second_completed_modules=s2.get('completed_modules'),
    difference_remove_minus_keep=(r2['ppl']-r1['ppl']) if 'ppl' in r2 and 'ppl' in r1 else None,
    sources=dict(with_scale=str(baseline_path),no_scale_keep_one=str(first/v/'ppl.json'),no_scale_remove_one=str(root/v/'ppl.json'))))
  write(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),protocol='LLaMA-1-7B W4-G128 WikiText-2 test block_size=2048 batch_size=8',fusion_status=run['fusion_status'],rows=rows))
  lines=['| 方法 | 原版含 scale | 去 scale：1+N(d) | 再去1+：N(d) | 第二轮状态 |','|---|---:|---:|---:|---|']
  for r in rows:
   fmt=lambda x:'—' if x is None else f'{x:.6f}'
   lines.append('| '+' | '.join([r['method'],fmt(r['with_scale']),fmt(r['no_scale_keep_one']),fmt(r['no_scale_remove_one']),r['second_status']])+' |')
  lines.append('\n融合暂缓：先比较两轮单侧 PPL，再确定后续融合使用的中心版本。')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
 try:
  while True:
   aggregate();states=[load(first/v/'run.json') for v in ['awq_delta','gptq_delta']]
   if any(s.get('status')=='failed' for s in states):raise RuntimeError('First-round endpoint failed')
   if all(s.get('status')=='complete' for s in states):
    for v in ['awq_delta','gptq_delta']:
     d=load(first/v/'ppl.json')
     for k,value in dict(model='/mnt/dfq/pretrained_model/Llama-7b',bit=4,group_size=128,AWQ_scale=False,block_size=2048,batch_size=8,objective='sum_j (1+N(d)_j)*(W_ij-center[label_ij])**2').items():
      if d.get(k)!=value:raise ValueError('First-round protocol mismatch: '+v+' '+k)
     if not isinstance(d.get('ppl'),(float,int)):raise ValueError('Missing PPL')
    break
   time.sleep(30)
  run.update(status='running_second_round',first_round_completed_at=time.strftime('%F %T'));write(root/'run.json',run)
  script=Path(__file__).parent/'run_delta_without_scale_or_offset.py';jobs={};procs={}
  (root/script.name).write_bytes(script.read_bytes())
  for v in ['awq_delta','gptq_delta']:
   command=[os.sys.executable,'-u',str(script),'--target','llama1','--variant',v,'--output',str(root/v),'--workers',str(a.workers)]
   log=root/(v+'.log')
   with log.open('xb') as fh:child=subprocess.Popen(command,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
   procs[v]=child;jobs[v]=dict(pid=child.pid,command=command,log=str(log));write(root/'jobs.json',jobs);print('Launched',v,child.pid,flush=True)
  while True:
   aggregate()
   if all(c.poll() is not None for c in procs.values()):
    if any(c.returncode!=0 for c in procs.values()):raise RuntimeError('Second-round child failed')
    for v in procs:
     if load(root/v/'run.json').get('status')!='complete' or 'ppl' not in load(root/v/'ppl.json'):raise RuntimeError('Incomplete second-round output')
    break
   time.sleep(30)
  run.update(status='complete',finished_at=time.strftime('%F %T'),next_action='Compare endpoint PPLs before choosing fusion source codebooks');write(root/'run.json',run);aggregate()
  print('Both ablations complete; fusion remains deferred for result comparison',flush=True)
 except BaseException as e:
  run.update(status='failed',error=repr(e));write(root/'run.json',run);traceback.print_exc();raise
if __name__=='__main__':main()
