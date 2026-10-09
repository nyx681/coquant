"""Launch a prepared W3 suite only after every W4 model completes successfully."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path(sys.argv[1])
def save(state):
    tmp=root/'queue.json.tmp'
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(root/'queue.json')
state=json.loads((root/'queue.json').read_text())
if state['status']!='waiting_for_w4':raise RuntimeError('Queue already launched; inspect before restarting')
state['pid']=os.getpid();save(state)
while True:
    dependency=json.loads(Path(state['depends_on']).read_text())
    state['checked_at']=time.strftime('%F %T');state['dependency_status']=dependency.get('status');save(state)
    if dependency.get('status')=='complete':break
    time.sleep(30)
cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_extended_four_ppl_suite_w3.py')),'--output',str(root),'--manifest',str(root/'manifest.json')]
try:
    with (root/'supervisor.log').open('ab') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(status='launched',child_pid=child.pid,command=cmd,started_at=time.strftime('%F %T'));save(state)
    (root/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=cmd,output=str(root)),indent=2))
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save(state);raise
