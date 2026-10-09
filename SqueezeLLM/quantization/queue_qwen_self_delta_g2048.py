"""Four matched G2048 AWQ-reference experiments; independent of prior suites."""
import os,sys,json,time,subprocess,csv
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
config=Path('/mnt/data/zch/quantization_experiments/qwen3_row_delta_ablation_20260922_111948/target.json')
state=dict(status='starting',pid=os.getpid(),model='Qwen3-8B-Base',bit=4,group_size=2048,centers_per_group=16,reference='existing AWQ W4-G128',activation=False,AWQ_scale=False,center_dtype='float32',tasks=[])
def save():
 state['updated_at']=time.strftime('%F %T');tmp=root/'run.json.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(root/'run.json')
save();children=[]
for design in ['self_floor05','self_floor08','uniform','absolute_inverse_control']:
 cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_self_delta_g2048.py')),'--target','qwen3_8b_base','--target-config',str(config),'--scope','row','--design',design,'--variant','awq_delta','--workers','6','--gpu-lock',str(root/'gpu.lock'),'--output',str(root/design)]
 with (root/(design+'.log')).open('wb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 e=dict(design=design,status='running',pid=p.pid,output=str(root/design),command=cmd);state['tasks'].append(e);children.append((p,e));save()
state['status']='running';save()
while children:
 for p,e in children[:]:
  if p.poll() is None:continue
  out=Path(e['output'])/'ppl.json';e['returncode']=p.returncode
  e['status']='complete' if p.returncode==0 and out.exists() else 'failed'
  if e['status']=='complete':e['ppl']=json.loads(out.read_text())['ppl']
  children.remove((p,e));save()
 if children:time.sleep(15)
state['status']='complete' if all(e['status']=='complete' for e in state['tasks']) else 'completed_with_failures';save()
with (root/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['design','status','group_size','ppl','source'])
 for e in state['tasks']:w.writerow([e['design'],e['status'],2048,e.get('ppl',''),str(Path(e['output'])/'ppl.json')])
