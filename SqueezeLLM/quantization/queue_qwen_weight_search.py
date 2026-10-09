"""Threshold-triggered followups; all attempts retained, never equate failure with success."""
import os,json,time,subprocess,sys,math
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
parent=Path('/mnt/data/zch/quantization_experiments/qwen3_row_delta_ablation_20260922_111948')
thresholds={'awq_delta':7.24,'gptq_delta':7.26}
designs=['column_power_half','column_power2','column_power4','absolute_inverse','absolute_power2','absolute_power4']
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def save():
 state['updated_at']=time.strftime('%F %T');tmp=root/'run.json.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(root/'run.json')
state=dict(status='waiting_for_row_ab',pid=os.getpid(),thresholds=thresholds,dependency=str(parent),trials=[],best={},designs=designs,test_used_for_selection=True);save()
while load(parent/'run.json').get('status')=='running':time.sleep(20)
if load(parent/'run.json').get('status')!='complete':
 state['status']='dependency_failed';save();sys.exit(1)
for variant in thresholds:
 results=[]
 for scope in ['module','row']:
  path=parent/(scope+'_'+variant)/'ppl.json';d=load(path)
  if 'ppl' not in d:raise RuntimeError('Missing completed baseline')
  results.append(dict(ppl=d['ppl'],source=str(path),design=scope))
 state['best'][variant]=min(results,key=lambda x:x['ppl'])
state['status']='searching';save();children={};next_trial={v:0 for v in thresholds}
while True:
 for variant,threshold in thresholds.items():
  if variant in children:
   child,entry=children[variant]
   if child.poll() is None:continue
   result=load(Path(entry['output'])/'ppl.json');entry['returncode']=child.returncode
   if child.returncode==0 and math.isfinite(result.get('ppl',float('nan'))):
    entry.update(status='complete',ppl=result['ppl'])
    if entry['ppl']<state['best'][variant]['ppl']:state['best'][variant]=dict(ppl=entry['ppl'],source=str(Path(entry['output'])/'ppl.json'),design=entry['design'])
   else:entry['status']='failed'
   del children[variant];save()
  if state['best'][variant]['ppl']<threshold or next_trial[variant]>=len(designs):continue
  design=designs[next_trial[variant]];next_trial[variant]+=1;out=root/(variant+'_'+design)
  cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_weight_design.py')),'--target','qwen3_8b_base','--target-config',str(parent/'target.json'),'--scope','row','--design',design,'--variant',variant,'--workers','12','--gpu-lock',str(parent/'search_gpu.lock'),'--output',str(out)]
  with (root/(out.name+'.log')).open('ab') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  entry=dict(variant=variant,design=design,status='running',pid=p.pid,output=str(out),command=cmd)
  state['trials'].append(entry);children[variant]=(p,entry);save()
 if not children:
  passed=all(state['best'][v]['ppl']<t for v,t in thresholds.items())
  state.update(status='thresholds_met' if passed else 'needs_further_design',finished_at=time.strftime('%F %T'));save();break
 time.sleep(20)
