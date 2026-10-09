"""Temporarily reserve CPU for the four fine-delta runs, then resume saved PIDs."""
import os,signal,json,time,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);record=root/'priority.json'
def load(p):return json.loads(p.read_text())
def save(d):
 t=record.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(record)
def identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None
def tree(pid):
 result=[pid]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().split(') ',1)[1].split()
   if int(fields[1])==pid:result+=tree(int(p.name))
  except (FileNotFoundError,ProcessLookupError):pass
 return result
state=dict(status='preparing',pid=os.getpid(),paused=[],started_at=time.strftime('%F %T'));save(state)
children={}
try:
 low=Path('/mnt/data/zch/quantization_experiments/extended_five_models_delta_only_20260921_090804')
 roots=[load(low/'launch.json')['pid'],1431082]+[v['pid'] for v in load(low/'jobs.json').values() if identity(v['pid'])]
 for pid in dict.fromkeys(p for r in roots for p in tree(r)):
  ident=identity(pid)
  if ident:
   state['paused'].append(dict(pid=pid,identity=ident));save(state);os.kill(pid,signal.SIGSTOP)
 # Old fine supervisor cannot launch duplicate work while pools are restarted.
 old=2428181
 if identity(old):os.kill(old,signal.SIGTERM)
 jobs=load(root/'jobs.json')
 for job in jobs.values():
  pid=job['pid']
  if identity(pid):os.killpg(pid,signal.SIGTERM)
 time.sleep(3)
 queue=[(s,v) for v in ('awq_delta','gptq_delta') for s in ('module','group')]
 state.update(status='running',workers_per_job=12,max_parallel=2);save(state)
 while True:
  while queue and sum(p.poll() is None for p in children.values())<2:
   scope,v=queue.pop(0);key=scope+'_'+v;out=root/key
   if (out/'ppl.json').exists():continue
   cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_fine_delta.py')),'--target','qwen3_8b_base','--target-config',str(root/'target.json'),'--scope',scope,'--variant',v,'--workers','12','--gpu-lock',str(low/'gpu.lock'),'--output',str(out)]
   if (out/'run.json').exists():cmd+=['--resume']
   with (root/(key+'.log')).open('ab') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   children[key]=p;jobs[key]=dict(pid=p.pid,output=str(out),command=cmd)
   (root/'jobs.json').write_text(json.dumps(jobs,indent=2))
  rows=[]
  for scope in ('baseline','module','group'):
   for v in ('awq_delta','gptq_delta'):
    out=low/'qwen3_8b_base/endpoints'/v if scope=='baseline' else root/(scope+'_'+v)
    s=load(out/'run.json') if (out/'run.json').exists() else {}
    result=load(out/'ppl.json') if (out/'ppl.json').exists() else {}
    rows.append(dict(scope=scope,variant=v,ppl=result.get('ppl'),status=s.get('status','queued'),completed_modules=s.get('completed_modules')))
  (root/'comparison.json').write_text(json.dumps(dict(rows=rows,updated_at=time.strftime('%F %T')),indent=2))
  lines=['| 方案 | AWQ-delta PPL | GPTQ-delta PPL |','|---|---:|---:|']
  for scope in ('baseline','module','group'):
   lines.append('| '+' | '.join([scope]+[str(r['ppl']) if r['ppl'] is not None else r['status'] for r in rows if r['scope']==scope])+' |')
  (root/'comparison.md').write_text('\n'.join(lines)+'\n')
  if not queue and all(p.poll() is not None for p in children.values()):break
  state['updated_at']=time.strftime('%F %T');save(state);time.sleep(20)
 state['failures']={k:p.returncode for k,p in children.items() if p.returncode!=0}
 state['status']='failed' if state['failures'] else 'complete'
 (root/'run.json').write_text(json.dumps(state,indent=2))
finally:
 for item in reversed(state['paused']):
  if identity(item['pid'])==item['identity']:
   try:os.kill(item['pid'],signal.SIGCONT)
   except ProcessLookupError:pass
 state['resumed_at']=time.strftime('%F %T');save(state)
