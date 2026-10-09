"""Run five delta-only row ablations, preserving the previous search by suspension."""
import os,sys,json,time,signal,subprocess
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
parent=Path('/mnt/data/zch/quantization_experiments/qwen3_row_delta_ablation_20260922_111948')
oldpid=2516725
state=dict(status='starting',pid=os.getpid(),tasks=[],paused=[],activation=False,AWQ_scale=False,group_size=-1,bit=4)
def save():
 state['updated_at']=time.strftime('%F %T');p=root/'run.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'run.json')
def info(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  return dict(pid=pid,ppid=int(fields[1]),identity=fields[19],cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
 except (FileNotFoundError,ProcessLookupError):return None
save();children=[]
try:
 old=info(oldpid)
 if old and 'queue_qwen_weight_search.py' in old['cmd']:
  os.kill(oldpid,signal.SIGSTOP);state['paused'].append(old)
  proc=[v for p in Path('/proc').iterdir() if p.name.isdigit() for v in [info(int(p.name))] if v]
  seen={oldpid}
  while True:
   new=[v for v in proc if v['ppid'] in seen and v['pid'] not in seen]
   if not new:break
   for v in new:
    try:os.kill(v['pid'],signal.SIGSTOP);state['paused'].append(v)
    except ProcessLookupError:pass
    seen.add(v['pid'])
  save()
 trials=[('uniform','awq_delta')]+[(d,v) for d in ['absolute_floor05','absolute_floor08'] for v in ['awq_delta','gptq_delta']]
 for design,variant in trials:
  name='uniform' if design=='uniform' else variant+'_'+design
  cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_soft_delta.py')),'--target','qwen3_8b_base','--target-config',str(parent/'target.json'),'--scope','row','--design',design,'--variant',variant,'--workers','5','--gpu-lock',str(root/'gpu.lock'),'--output',str(root/name)]
  with (root/(name+'.log')).open('wb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  entry=dict(name=name,design=design,variant=variant,pid=p.pid,output=str(root/name),command=cmd,status='running');state['tasks'].append(entry);children.append((p,entry));save()
 state['status']='running';save()
 while children:
  for p,e in children[:]:
   if p.poll() is None:continue
   e['returncode']=p.returncode;e['status']='complete' if p.returncode==0 and (Path(e['output'])/'ppl.json').exists() else 'failed'
   if e['status']=='complete':e['ppl']=json.loads((Path(e['output'])/'ppl.json').read_text())['ppl']
   children.remove((p,e));save()
  if children:time.sleep(15)
 state['status']='complete' if all(e['status']=='complete' for e in state['tasks']) else 'completed_with_failures'
finally:
 for v in reversed(state['paused']):
  cur=info(v['pid'])
  if cur and cur['identity']==v['identity']:
   try:os.kill(v['pid'],signal.SIGCONT)
   except ProcessLookupError:pass
 state['old_search_resumed_at']=time.strftime('%F %T');save()
