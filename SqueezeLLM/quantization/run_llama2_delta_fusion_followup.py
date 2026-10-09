"""Persisted sequence: LLaMA2 Fisher-delta B, Qwen/LLaMA1/LLaMA2 fusion."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,csv,subprocess
from pathlib import Path

ROOT=Path(sys.argv[1])
PLAN=json.loads((ROOT/'plan.json').read_text())
BASE=Path('/home/zch/project/Quantification/SqueezeLLM')
PY='/home/zch/miniconda3/envs/awq/bin/python'
LOCK='/mnt/data/zch/quantization_experiments/qwen3_self_delta_g2048_20260922_222333/gpu.lock'
ENV=os.environ.copy();ENV['PYTHONPATH']='.:quantization'
STATE=dict(status='waiting_for_current_jobs',pid=os.getpid(),tasks=[],started_at=time.strftime('%F %T'))

def read(p):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {}

def save():
    STATE['updated_at']=time.strftime('%F %T')
    p=ROOT/'status.json.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'status.json')

def start(name,cmd,out,log):
    t=dict(name=name,status='running',output=str(out),command=cmd)
    with Path(log).open('ab') as f:
        proc=subprocess.Popen(cmd,cwd=BASE,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,stdin=subprocess.DEVNULL)
    t['pid']=proc.pid;STATE['tasks'].append(t);save()
    return proc,t

def finish(proc,t):
    result=read(Path(t['output'])/'ppl.json')
    t.update(returncode=proc.returncode,status='complete' if proc.returncode==0 and result else 'failed')
    if result:t['ppl']=result['ppl']
    save()
    if t['status']=='failed':raise RuntimeError('Task failed: '+t['name'])

def wait(proc,t):
    while proc.poll() is None:time.sleep(15)
    finish(proc,t)

def main():
    save()
    while True:
        opt=read(Path(PLAN['cpu_dependency'])/'run.json')
        llama=read(PLAN['llama1_eval_dependency'])
        if llama.get('status')=='completed_with_failures':
            raise RuntimeError('LLaMA1 FP16 dependency failed')
        free=opt.get('status') in ('waiting_for_gpu','reconstructing_fp16','evaluating','complete','failed')
        STATE['dependencies']=dict(opt_status=opt.get('status'),llama1_fp16_status=llama.get('status'))
        save()
        if free and llama.get('status')=='complete':break
        time.sleep(15)
    STATE['status']='running_llama2_delta';save()
    d=Path(PLAN['llama2_delta']);(d/'fp16').mkdir(exist_ok=True)
    active=[];eval_pending=[];eval_active=None
    for variant in ('awq_delta','gptq_delta'):
        for design in ('self_floor05','self_floor08'):
            name=variant+'_'+design;out=d/name
            cmd=[PY,'-u','quantization/run_hierarchical_fisher_tail.py','--fisher-root',str(d/'fisher'),
                 '--target-config',str(d/'target.json'),'--design',design,'--scope','row',
                 '--target','llama2_7b','--variant',variant,'--workers','6','--output',str(out),
                 '--gpu-lock',LOCK]
            active.append(start('llama2_'+name,cmd,out,d/(name+'.log')))
    while active or eval_pending or eval_active:
        for proc,t in active[:]:
            if proc.poll() is not None:
                finish(proc,t);active.remove((proc,t));eval_pending.append(t)
        if eval_active and eval_active[0].poll() is not None:
            finish(*eval_active);eval_active=None
        if eval_active is None and eval_pending:
            src=Path(eval_pending.pop(0)['output']);out=d/'fp16'/src.name;out.mkdir()
            cmd=[PY,'-u','quantization/eval_B_fp16_codebooks_generic.py',str(out),str(src),str(d/'target.json')]
            eval_active=start('llama2_fp16_'+src.name,cmd,out,out/'eval.log')
        if active or eval_pending or eval_active:time.sleep(15)
    for fusion in PLAN['fusions']:
        STATE['status']='running_'+fusion['name'];save()
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
             '--output',str(out),'--workers','24','--gpu-lock',LOCK,'--baseline-csv',str(baseline)]
        wait(*start(fusion['name'],cmd,out,ROOT/(fusion['name']+'.log')))
    STATE.update(status='complete',finished_at=time.strftime('%F %T'));save()

try:main()
except BaseException as error:
    STATE.update(status='failed',error=repr(error));save();raise
