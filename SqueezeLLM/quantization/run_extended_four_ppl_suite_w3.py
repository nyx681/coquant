"""Two concurrent model pipelines, five-model inventory, restartable children."""
import argparse,csv,io,json,os,subprocess,time,traceback
from pathlib import Path

def load(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'run.json').exists() and not a.resume:raise FileExistsError('Suite already exists; inspect existing jobs before relaunch')
    run=load(root/'run.json') if a.resume else dict(started_at=time.strftime('%F %T'))
    run.update(status='running',max_active_models=None,max_cpu_model_pipelines=2,
               scheduling='GPU source preparation independent of CPU pipeline slots')
    write(root/'run.json',run);jobs=load(root/'jobs.json') if a.resume else {};children={};pipelines=load(root/'pipelines.json') if a.resume else {};code=Path(__file__).parent
    class Adopted:
        def __init__(self,job):self.job=job;self.returncode=None
        def poll(self):
            proc=Path('/proc')/str(self.job['pid'])
            if proc.exists():
                stat=(proc/'stat').read_text().split(') ',1)[1].split()[0]
                if stat!='Z':
                    cmd=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
                    if self.job['output'] not in cmd:raise RuntimeError('Adopted PID identity mismatch')
                    return None
            self.returncode=0 if load(Path(self.job['output'])/'run.json').get('status')=='complete' else 1
            return self.returncode
    for key,job in jobs.items():children[key]=Adopted(job)
    def launch(key,script,args,out):
        log=root/(key+'.log');cmd=[os.sys.executable,'-u',str(code/script),*args,'--output',str(out)]
        if (out/'run.json').exists():cmd.append('--resume')
        with log.open('ab') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children[key]=child;jobs[key]=dict(pid=child.pid,output=str(out),log=str(log),command=cmd)
        write(root/'jobs.json',jobs);print('LAUNCHED',key,child.pid,flush=True)
    def aggregate(manifest):
        rows=[]
        for entry in manifest:
            tag=entry['tag'];base=root/tag
            for method,relative in [('AWQ-delta：去 scale、去1+','endpoints/awq_delta'),('GPTQ-delta：去 scale、去1+','endpoints/gptq_delta'),('融合：系数1','fusion/unit'),('融合：系数activation','fusion/activation')]:
                out=base/relative;state=load(out/'run.json');result=load(out/'ppl.json')
                if not state and relative.startswith('fusion'):state=load(base/'fusion/run.json')
                rows.append(dict(model=entry['name'],method=method,ppl=result.get('ppl'),
                    status=state.get('status',pipelines.get(tag,{}).get('stage','not_started')),
                    completed_modules=state.get('completed_modules'),source=str(out/'ppl.json')))
        write(root/'comparison.json',dict(updated_at=time.strftime('%F %T'),protocol='W3-G128 WikiText-2 test block2048 batch8',rows=rows))
        lines=['| 模型 | 方法 | PPL ↓ | 状态 |','|---|---|---:|---|']
        for row in rows:
            value='—' if row['ppl'] is None else f'{row["ppl"]:.6f}'
            lines.append(f'| {row["model"]} | {row["method"]} | {value} | {row["status"]} |')
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (root/'comparison.csv').write_text(buf.getvalue())
    try:
        while True:
            manifest=json.loads(Path(a.manifest).read_text())
            for entry in manifest:
                tag=entry['tag'];base=root/tag;base.mkdir(exist_ok=True)
                pipe=pipelines.setdefault(tag,dict(stage='waiting_for_model'))
                if pipe['stage'] in ('failed','complete'):continue
                if not Path(entry['model']).is_dir() or not (Path(entry['model'])/'config.json').exists():
                    pipe.update(stage='waiting_for_model',model=entry['model']);continue
                if pipe['stage']=='waiting_for_model':pipe['stage']='queued'
                if pipe['stage']=='queued':
                    mc=load(Path(entry['model'])/'config.json')
                    if mc['model_type']!=entry['architecture']:raise ValueError('Model architecture mismatch: '+tag)
                    cfg=dict(model=entry['model'],model_type='opt' if entry['architecture']=='opt' else 'llama',
                             layers=mc['num_hidden_layers'],scale=entry['scale'],activation_cache=entry['activation_cache'],
                             original=entry['original'],awq=str(base/'sources/awq'),gptq=str(base/'sources/gptq'))
                    config=base/'target.json';write(config,cfg)
                    launch(tag+'_sources','prepare_extended_delta_sources_w3.py',
                           ['--target-config',str(config),'--gpu-lock',str(root/'gpu.lock')],base/'sources')
                    pipe.update(stage='preparing_sources',model=entry['model'],config=str(config));continue
                if pipe['stage'] in ('preparing_sources','waiting_for_cpu'):
                    child=children[tag+'_sources']
                    if child.poll() is None:continue
                    if child.returncode!=0 or load(base/'sources/run.json').get('status')!='complete':
                        pipe.update(stage='failed',reason='source preparation failed');continue
                    active=sum(v['stage'] in ('running_endpoints','running_fusions') for v in pipelines.values())
                    if active>=2:
                        pipe['stage']='waiting_for_cpu';continue
                    (base/'endpoints').mkdir(exist_ok=True)
                    write(base/'endpoints/run.json',dict(status='running'))
                    for variant in ('awq_delta','gptq_delta'):
                        launch(tag+'_'+variant,'run_extended_delta_only_w3.py',
                               ['--target',tag,'--target-config',pipe['config'],'--gpu-lock',str(root/'gpu.lock'),
                                '--variant',variant,'--workers','6'],base/'endpoints'/variant)
                    pipe['stage']='running_endpoints'
                if pipe['stage'] in ('running_endpoints','running_fusions'):
                    states=[load(base/'endpoints'/v/'run.json') for v in ('awq_delta','gptq_delta')]
                    if any(children[tag+'_'+v].poll() not in (None,0) for v in ('awq_delta','gptq_delta')):
                        write(base/'endpoints/run.json',dict(status='failed'));pipe.update(stage='failed',reason='endpoint failed');continue
                    if all(s.get('status')=='complete' for s in states):
                        write(base/'endpoints/run.json',dict(status='complete'))
                    if tag+'_fusion' not in children and all(s.get('model') for s in states):
                        launch(tag+'_fusion','run_extended_delta_fusions_w3.py',
                               ['--target',tag,'--target-config',pipe['config'],'--endpoint-root',str(base/'endpoints'),
                                '--producer-jobs',str(root/'jobs.json'),'--producer-run',str(base/'endpoints/run.json'),
                                '--gpu-lock',str(root/'gpu.lock'),'--workers','4','--boost-workers','16'],base/'fusion')
                        pipe['stage']='running_fusions'
                    if tag+'_fusion' in children:
                        child=children[tag+'_fusion']
                        if child.poll() is not None:
                            if child.returncode!=0:pipe.update(stage='failed',reason='fusion failed')
                            elif all(s.get('status')=='complete' for s in states):pipe['stage']='complete'
            write(root/'pipelines.json',pipelines);aggregate(manifest)
            if all(pipelines[e['tag']]['stage']=='complete' for e in manifest):
                run.update(status='complete',finished_at=time.strftime('%F %T'));write(root/'run.json',run);break
            if all(v['stage'] in ('failed','waiting_for_model','complete') for v in pipelines.values()):
                run['status']='waiting_for_inputs_or_repair'
            else:run['status']='running'
            write(root/'run.json',run);time.sleep(30)
    except BaseException as exc:
        run.update(status='failed',error=repr(exc));write(root/'run.json',run);traceback.print_exc();raise
if __name__=='__main__':main()
