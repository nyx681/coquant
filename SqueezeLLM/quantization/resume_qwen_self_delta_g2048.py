import sys,os,time,json,subprocess,shutil,csv
from pathlib import Path
root=Path(sys.argv[1]);state=json.loads((root/'run.json').read_text());stamp=time.strftime('%Y%m%d_%H%M%S')
shutil.copy(root/'run.json',root/('run.before_resume_'+stamp+'.json'))
state.update(status='resuming_after_reboot',pid=os.getpid(),resumed_at=time.strftime('%F %T'));children=[]
def save():
 state['updated_at']=time.strftime('%F %T');tmp=root/'run.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'run.json')
save()
for e in state['tasks']:
 out=Path(e['output']);shutil.copy(out/'run.json',out/('run.before_resume_'+stamp+'.json'))
 cmd=e['command']+([] if '--resume' in e['command'] else ['--resume']);log=root/(e['design']+'.resume_'+stamp+'.log')
 with log.open('wb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 e.update(pid=p.pid,status='running',command=cmd,resume_log=str(log));children.append((p,e));save()
state['status']='running';save()
while children:
 for p,e in children[:]:
  if p.poll() is None:continue
  out=Path(e['output'])/'ppl.json';e['returncode']=p.returncode;e['status']='complete' if p.returncode==0 and out.exists() else 'failed'
  if e['status']=='complete':e['ppl']=json.loads(out.read_text())['ppl']
  children.remove((p,e));save()
 if children:time.sleep(15)
state['status']='complete' if all(e['status']=='complete' for e in state['tasks']) else 'completed_with_failures';save()
with (root/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['design','status','group_size','ppl','source'])
 for e in state['tasks']:w.writerow([e['design'],e['status'],2048,e.get('ppl',''),str(Path(e['output'])/'ppl.json')])
