"""Adopt running delta jobs; run Qwen fusion concurrently, without duplicates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,csv,subprocess
from pathlib import Path
ROOT=Path(sys.argv[1]);PLAN=json.loads((ROOT/'plan.json').read_text())
STATE=json.loads((ROOT/'status.json').read_text())
BASE=Path('/home/zch/project/Quantification/SqueezeLLM')
PY='/home/zch/miniconda3/envs/awq/bin/python'
LOCK='/mnt/data/zch/quantization_experiments/qwen3_self_delta_g2048_20260922_222333/gpu.lock'
ENV=os.environ.copy();ENV['PYTHONPATH']='.:quantization'

def read(p):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {}

def save():
    STATE['updated_at']=time.strftime('%F %T')
    p=ROOT/'status.json.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'status.json')

class Adopted:
    def __init__(self,t):self.t=t;self.pid=t['pid'];self.returncode=None
    def poll(self):
        run=read(Path(self.t['output'])/'run.json')
        if run.get('status')=='complete' and (Path(self.t['output'])/'ppl.json').exists():
            self.returncode=0;return 0
        try:
            p=Path('/proc')/str(self.pid)
            state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if state=='Z' or self.t['output'] not in command:
                self.returncode=1;return 1
        except OSError:self.returncode=1;return 1
        return None

def start(name,cmd,out,log):
    matches=[t for t in STATE['tasks'] if t['name']==name]
    if matches:
        t=matches[0];return Adopted(t),t
    t=dict(name=name,status='running',output=str(out),command=cmd)
    with Path(log).open('ab') as f:
        proc=subprocess.Popen(cmd,cwd=BASE,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,stdin=subprocess.DEVNULL)
    t['pid']=proc.pid;STATE['tasks'].append(t);save();return proc,t

def finish(proc,t):
    result=read(Path(t['output'])/'ppl.json')
    t.update(returncode=proc.returncode,status='complete' if proc.returncode==0 and result else 'failed')
    if result:t['ppl']=result['ppl']
    save()
    if t['status']=='failed':raise RuntimeError('Task failed: '+t['name'])

def wait(proc,t):
    while proc.poll() is None:time.sleep(15)
    finish(proc,t)

def fusion_start(fusion,workers,boost=()):
    baseline=ROOT/(fusion['name']+'_baselines.csv')
    with baseline.open('w') as out:
        writer=csv.writer(out);writer.writerow(['method','FP16_parameters_ppl','actual_quantized_weight_bits','source'])
        for method,key in [('awq_self_floor05','baseline_awq'),('gptq_self_floor05','baseline_gptq')]:
            r=read(fusion[key])
            if not r:raise RuntimeError('Missing FP16 baseline '+fusion[key])
            writer.writerow([method,r['ppl'],r['actual_quantized_weight_bits'],fusion[key]])
    out=ROOT/fusion['name']
    cmd=[PY,'-u','quantization/run_source_scale_worst_out_tail.py','--awq',fusion['awq'],
         '--gptq',fusion['gptq'],'--fisher',fusion['fisher'],'--target-config',fusion['config'],
         '--output',str(out),'--workers',str(workers),'--gpu-lock',LOCK,'--baseline-csv',str(baseline)]
    if boost:cmd+=['--boost-workers','24','--boost-after-runs',*boost]
    return start(fusion['name'],cmd,out,ROOT/(fusion['name']+'.log'))

def main():
    STATE.update(pid=os.getpid(),status='llama2_delta_and_qwen_fusion_parallel',
                 scheduling='Qwen 8 workers concurrent; boost to 24 after all LLaMA2 center jobs finish; GPU lock retained')
    d=Path(PLAN['llama2_delta']);(d/'fp16').mkdir(exist_ok=True)
    delta=[t for t in STATE['tasks'] if t['name'].startswith('llama2_') and not t['name'].startswith('llama2_fp16_')]
    assert len(delta)==4
    active=[(Adopted(t),t) for t in delta if t['status']!='complete']
    eval_tasks=[t for t in STATE['tasks'] if t['name'].startswith('llama2_fp16_')]
    eval_active=next(((Adopted(t),t) for t in eval_tasks if t['status']=='running'),None)
    evaluated={Path(t['output']).name for t in eval_tasks}
    eval_pending=[t for t in delta if t['status']=='complete' and Path(t['output']).name not in evaluated]
    save()
    qwen=fusion_start(PLAN['fusions'][0],8,[t['output'] for t in delta]);qwen_finished=False
    while active or eval_pending or eval_active:
        for proc,t in active[:]:
            if proc.poll() is not None:
                finish(proc,t);active.remove((proc,t))
                if Path(t['output']).name not in evaluated:eval_pending.append(t)
        if eval_active and eval_active[0].poll() is not None:
            finish(*eval_active);eval_active=None
        if not qwen_finished and qwen[0].poll() is not None:
            finish(*qwen);qwen_finished=True
        if eval_active is None and eval_pending:
            src=Path(eval_pending.pop(0)['output']);out=d/'fp16'/src.name;out.mkdir()
            evaluated.add(src.name)
            cmd=[PY,'-u','quantization/eval_B_fp16_codebooks_generic.py',str(out),str(src),str(d/'target.json')]
            eval_active=start('llama2_fp16_'+src.name,cmd,out,out/'eval.log')
        save()
        if active or eval_pending or eval_active:time.sleep(15)
    if not qwen_finished:
        STATE['status']='waiting_for_qwen_fusion';save();wait(*qwen)
    for fusion in PLAN['fusions'][1:]:
        STATE['status']='running_'+fusion['name'];save();wait(*fusion_start(fusion,24))
    STATE.update(status='complete',finished_at=time.strftime('%F %T'));save()

try:main()
except BaseException as error:
    STATE.update(status='failed',error=repr(error));save();raise
