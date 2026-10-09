import os,sys,time,json,signal,subprocess
from pathlib import Path
root=Path(sys.argv[1]);statefile=Path(str(root)+'_priority.json')
def ident(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None
def tree(pid):
 out=[pid]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    if int((p/'stat').read_text().split(') ',1)[1].split()[1])==pid:out+=tree(int(p.name))
   except FileNotFoundError:pass
 return out
def save():statefile.write_text(json.dumps(state,indent=2))
state=dict(status='preparing',paused=[],pid=os.getpid());save()
try:
 old=Path('/mnt/data/zch/quantization_experiments/qwen3_fine_delta_ablation_20260922_104823')
 controller=json.loads((old/'priority.json').read_text())['pid']
 os.kill(controller,signal.SIGSTOP)
 state['paused'].append(dict(pid=controller,identity=ident(controller)));save()
 for pid in tree(controller)[1:]:
  identity=ident(pid)
  if identity:
   state['paused'].append(dict(pid=pid,identity=identity));save();os.kill(pid,signal.SIGSTOP)
 cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_row_delta_suite.py')),str(root)]
 with Path(str(root)+'_supervisor.log').open('ab') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='running',child_pid=p.pid,command=cmd);save()
 rc=p.wait();state.update(status='complete' if rc==0 else 'failed',returncode=rc)
finally:
 for item in reversed(state['paused']):
  if ident(item['pid'])==item['identity']:
   try:os.kill(item['pid'],signal.SIGCONT)
   except ProcessLookupError:pass
 state['resumed_at']=time.strftime('%F %T');save()
