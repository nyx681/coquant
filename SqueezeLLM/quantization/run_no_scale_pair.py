"""Launch both endpoint ablations and keep a live, provenance-backed table."""
import argparse,json,os,subprocess,time,hashlib
from pathlib import Path

def write(path,data):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--target',default='llama1',choices=['llama1','llama2','opt']);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--boost-after-pid',type=int);p.add_argument('--boost-workers',type=int,default=12);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 script=Path(__file__).parent/'run_delta_without_awq_scale.py';jobs={};procs={}
 status=dict(target=a.target,status='running',started_at=time.strftime('%F %T'),objective='1+N(d); no AWQ scale prior',bit=4,group_size=128)
 write(root/'run.json',status)
 # Keep an exact runnable source snapshot with the invocation record.
 (root/script.name).write_bytes(script.read_bytes())
 for v in ['awq_delta','gptq_delta']:
  command=[os.sys.executable,'-u',str(script),'--target',a.target,'--variant',v,'--output',str(root/v),'--workers',str(a.workers),'--boost-workers',str(a.boost_workers)]
  if a.boost_after_pid:command+=['--boost-after-pid',str(a.boost_after_pid)]
  log=root/f'{v}.log'
  with log.open('xb') as fh:child=subprocess.Popen(command,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
  procs[v]=child;jobs[v]=dict(pid=child.pid,command=command,log=str(log),output=str(root/v))
  write(root/'jobs.json',jobs);print('Launched',v,child.pid,flush=True)
 while True:
  rows=[]
  for v,child in procs.items():
   d=root/v;old=d/'baseline_with_scale.json';new=d/'ppl.json';state=d/'run.json'
   r=dict(method=v,with_scale=json.loads(old.read_text())['ppl'] if old.exists() else None,
     without_scale=json.loads(new.read_text())['ppl'] if new.exists() else None,
     status=json.loads(state.read_text())['status'] if state.exists() else 'starting',source=str(new))
   if child.poll() is not None and child.returncode!=0:r.update(status='failed',exit_code=child.returncode)
   r['difference_without_minus_with']=r['without_scale']-r['with_scale'] if r['without_scale'] is not None and r['with_scale'] is not None else None
   rows.append(r)
  write(root/'comparison.json',dict(target=a.target,updated_at=time.strftime('%F %T'),rows=rows,protocol='W4-G128; WikiText-2 test; block_size=2048; batch_size=8'))
  lines=['| 方法 | 含 AWQ scale | 去掉 AWQ scale | 差值（去掉−保留） | 状态 |','|---|---:|---:|---:|---|']
  for r in rows:
   fmt=lambda x:'—' if x is None else f'{x:.6f}'
   lines.append('| '+ ' | '.join([r['method'],fmt(r['with_scale']),fmt(r['without_scale']),fmt(r['difference_without_minus_with']),r['status']])+' |')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
  if all(c.poll() is not None for c in procs.values()):
   status.update(status='complete' if all(r['status']=='complete' for r in rows) else 'failed',finished_at=time.strftime('%F %T'));write(root/'run.json',status);break
  time.sleep(30)
if __name__=='__main__':main()
