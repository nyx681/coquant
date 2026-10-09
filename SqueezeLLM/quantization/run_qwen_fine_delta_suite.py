"""Two concurrent per-weight endpoint ablations; reuse existing Qwen references."""
import json,os,subprocess,time,sys
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
source=Path('/mnt/data/zch/quantization_experiments/extended_five_models_delta_only_20260921_090804/qwen3_8b_base')
def save(p,obj):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def load(p):return json.loads(p.read_text()) if p.exists() else {}
config=load(source/'target.json');save(root/'target.json',config)
queue=[(scope,v) for v in ('awq_delta','gptq_delta') for scope in ('module','group')];children={};jobs={}
save(root/'run.json',dict(status='running',started_at=time.strftime('%F %T'),max_parallel=2,workers_per_job=2,bit=4,group_size=128,scope='four endpoint PPLs; fusion after comparison'))
while True:
 while queue and sum(p.poll() is None for p in children.values())<2:
  scope,v=queue.pop(0);key=scope+'_'+v;out=root/key
  cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_qwen_fine_delta.py')),'--target','qwen3_8b_base','--target-config',str(root/'target.json'),'--scope',scope,'--variant',v,'--workers','2','--gpu-lock',str(source.parent/'gpu.lock'),'--output',str(out)]
  with (root/(key+'.log')).open('xb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children[key]=p;jobs[key]=dict(pid=p.pid,output=str(out),command=cmd);save(root/'jobs.json',jobs)
 rows=[]
 for scope in ('baseline','module','group'):
  for v in ('awq_delta','gptq_delta'):
   out=source/'endpoints'/v if scope=='baseline' else root/(scope+'_'+v)
   state=load(out/'run.json');result=load(out/'ppl.json')
   rows.append(dict(scope=scope,variant=v,ppl=result.get('ppl'),status=state.get('status','queued'),completed_modules=state.get('completed_modules'),total_modules=252,source=str(out/'ppl.json')))
 save(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),rows=rows))
 lines=['| 系数方案 | AWQ-delta PPL | GPTQ-delta PPL |','|---|---:|---:|']
 for scope,label in [('baseline','每列系数（已有）'),('module','逐权重 A：模块中位数'),('group','逐权重 B：G128 中位数')]:
  cells=[str(r['ppl']) if r['ppl'] is not None else r['status'] for r in rows if r['scope']==scope];lines.append('| '+' | '.join([label]+cells)+' |')
 (root/'comparison.md').write_text('\n'.join(lines)+'\n')
 if not queue and all(p.poll() is not None for p in children.values()):
  failed={k:p.returncode for k,p in children.items() if p.returncode!=0}
  save(root/'run.json',dict(status='failed' if failed else 'complete',failures=failed,finished_at=time.strftime('%F %T')));break
 time.sleep(20)
