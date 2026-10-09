"""Wait for both no-scale endpoint PPLs, then run two controlled fusions."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path

def write(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
 p=argparse.ArgumentParser();p.add_argument('--endpoint-root',required=True);p.add_argument('--uniform-root',required=True);p.add_argument('--activation-root',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);end=Path(a.endpoint_root)
 endpoint_objectives={v:load(end/v/'run.json').get('objective') for v in ['awq_delta','gptq_delta']}
 endpoint_label='S=N(d)' if load(end/'awq_delta/run.json').get('importance_offset')==0 else 'S=1+N(d)'
 sources=[('AWQ-delta '+endpoint_label,end/'awq_delta'),('GPTQ-delta '+endpoint_label,end/'gptq_delta'),('融合：系数1',Path(a.uniform_root)),('融合：activation',Path(a.activation_root))]
 run=dict(status='waiting_for_endpoints',started_at=time.strftime('%F %T'),endpoint_root=str(end),endpoint_objectives=endpoint_objectives,uniform_root=a.uniform_root,activation_root=a.activation_root,workers_per_fusion=a.workers)
 write(root/'run.json',run)
 def aggregate():
  rows=[]
  for name,d in sources:
   state=load(d/'run.json');result=load(d/'ppl.json')
   rows.append(dict(method=name,ppl=result.get('ppl'),status=state.get('status','not_started'),completed_modules=state.get('completed_modules'),source=str(d/'ppl.json')))
  write(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),protocol='LLaMA-1-7B W4-G128 WikiText-2 test block_size=2048 batch_size=8',rows=rows))
  lines=['| 方法 | PPL ↓ | 状态 |','|---|---:|---|']
  for r in rows:lines.append('| '+r['method']+' | '+('—' if r['ppl'] is None else f'{r["ppl"]:.6f}')+' | '+r['status']+' |')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
  return rows
 try:
  while True:
   rows=aggregate();states=[load(end/v/'run.json') for v in ['awq_delta','gptq_delta']]
   if any(x.get('status')=='failed' for x in states):raise RuntimeError('An endpoint failed; no fusion launched')
   if all(x.get('status')=='complete' for x in states):
    for v in ['awq_delta','gptq_delta']:
     result=load(end/v/'ppl.json')
     for k,expected in dict(bit=4,group_size=128,AWQ_scale=False,block_size=2048,batch_size=8,model='/mnt/dfq/pretrained_model/Llama-7b').items():
      if result.get(k)!=expected:raise ValueError('Endpoint evaluation mismatch: '+v+' '+k)
     if not isinstance(result.get('ppl'),(int,float)):raise ValueError('Endpoint PPL missing')
    break
   time.sleep(30)
  run.update(status='running_fusions',endpoints_finished_at=time.strftime('%F %T'));write(root/'run.json',run)
  jobs={};procs={}
  for label,script,out,resume in [('uniform','run_blackbox_center_fusion.py',a.uniform_root,(Path(a.uniform_root)/'run.json').exists()),('activation','run_blackbox_activation_fusion.py',a.activation_root,False)]:
   command=[os.sys.executable,'-u',str(Path(__file__).parent/script),'--source-mode','no_scale','--endpoint-root',str(end),'--output',out,'--workers',str(a.workers)]
   if resume:command.append('--resume')
   log=root/(label+'.log')
   with log.open('xb') as fh:child=subprocess.Popen(command,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
   procs[label]=child;jobs[label]=dict(pid=child.pid,command=command,log=str(log),output=out);write(root/'jobs.json',jobs)
   print('Launched',label,child.pid,flush=True)
  while True:
   aggregate()
   if all(c.poll() is not None for c in procs.values()):
    failed={label:c.returncode for label,c in procs.items() if c.returncode!=0}
    if failed:raise RuntimeError('Fusion child failed: '+str(failed))
    if any(load(Path(d)/'run.json').get('status')!='complete' for d in [a.uniform_root,a.activation_root]):raise RuntimeError('Child exited without complete result')
    break
   time.sleep(30)
  run.update(status='complete',finished_at=time.strftime('%F %T'));write(root/'run.json',run);aggregate()
 except BaseException as e:
  run.update(status='failed',error=repr(e));write(root/'run.json',run);traceback.print_exc();raise
if __name__=='__main__':main()
