"""Complete the three-model RTN uniform vs RTN-delta W4-G128 PPL table.

Only missing cells are generated. Single-side delta codebooks are persisted
module by module; native RTN uses the same min/max helper as the earlier runs.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import gc
import hashlib
import json
import subprocess
import time
import traceback
import shutil
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from nuq import kmeans_fit,build_inverse_channel_relative_weights,normalize_score_row,load_awq_channel_scales
from group_center_selection import assign
from opt_rtn_gptq_fusion import rtn
from chunk_gptq_params import LLAMA_MODULES,OPT_MODULES
from run_cross_model_dual_nonuniform_g128 import TARGETS as CROSS_TARGETS
from run_llama1_activation_fusion import fill_weight,digest_arrays,write_json
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.model_parse import get_layers
from squeezellm.datautils import get_awq_ppl_tokens

TARGETS={
    'llama1':dict(model='/mnt/dfq/pretrained_model/Llama-7b',model_type='llama',
                  original='/mnt/dfq/squeezellm_awq_delta/original_chunks',
                  scale='/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt'),
    **{k:{n:v[n] for n in ('model','model_type','original','scale')} for k,v in CROSS_TARGETS.items()}
}
TITLES={'llama1':'LLaMA-1-7B','llama2':'LLaMA-2-7B','opt':'OPT-6.7B'}


def quantize_rtn(w):
    centers,codes=rtn(w.reshape(len(w),-1,128))
    q=np.take_along_axis(centers,codes.astype(np.int64),axis=-1).reshape(w.shape)
    return q


def row_delta(task):
    w,importance=task
    centers=np.empty((len(w)//128,16),np.float32)
    costs=np.zeros(2,np.float64)
    for g in range(len(centers)):
        sl=slice(g*128,(g+1)*128)
        c,_=kmeans_fit((w[sl,None],importance[sl],16))
        centers[g]=c
        _,weighted,ordinary=assign(w[sl],importance[sl],c)
        costs+=weighted,ordinary
    return centers,costs


def execute_child(args,root,config):
    cfg=TARGETS[args.target]
    modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
    model=get_model(cfg['model']).eval()
    layers=get_layers(model,cfg['model_type'])
    if len(layers)!=32:
        raise ValueError('Expected 32 model layers')
    torch.set_num_threads(1)
    scales=load_awq_channel_scales(cfg['scale']) if args.variant=='delta' else None
    centers_dir=root/'centers'
    if args.variant=='delta':
        centers_dir.mkdir(exist_ok=True)
        needed=sum(layer.get_submodule(suffix).weight.numel() for layer in layers for suffix in modules.values())*.5
        existing=sum(p.stat().st_size for p in centers_dir.glob('*.npz'))
        if shutil.disk_usage(root).free<max(needed-existing,0)+512*1024**2:
            raise OSError('Insufficient space for RTN-delta codebooks')
    pool=get_context('spawn').Pool(args.workers) if args.variant=='delta' else None
    metrics={}
    try:
        for l,layer in enumerate(layers):
            original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
            for name,suffix in modules.items():
                key=f'{l}.{name}';started=time.monotonic()
                module=layer.get_submodule(suffix)
                if not torch.equal(module.weight.detach().cpu(),original[name].to(module.weight.dtype)):
                    raise ValueError(f'Original chunks differ from model: {key}')
                w=original[name].float().numpy()
                q=quantize_rtn(w)
                if args.variant=='uniform':
                    with torch.no_grad():module.weight.copy_(torch.from_numpy(q))
                    info=dict(weight_sse=float(np.square(w.astype(np.float64)-q).sum()))
                else:
                    delta=build_inverse_channel_relative_weights(w,np.abs(w-q),gamma=1,min_weight=.1)
                    prior=normalize_score_row(scales[l,name].numpy())**(3 if name=='o' else 4)
                    importance=prior*(1+normalize_score_row(delta))
                    if importance.shape!=(w.shape[1],) or not np.isfinite(importance).all() or np.any(importance<=0):
                        raise ValueError(f'Invalid RTN-delta importance: {key}')
                    fingerprint=digest_arrays(w,importance)
                    checkpoint=centers_dir/f'{key}.npz'
                    reused=checkpoint.exists()
                    if reused:
                        with np.load(checkpoint,allow_pickle=False) as saved:
                            if str(saved['fingerprint'])!=fingerprint:
                                raise ValueError(f'Checkpoint source mismatch: {key}')
                            centers,costs=saved['centers'],saved['costs']
                    else:
                        centers=np.empty((len(w),w.shape[1]//128,16),np.float32)
                        costs=np.zeros(2,np.float64)
                        tasks=((w[r],importance) for r in range(len(w)))
                        for r,(c,loss) in enumerate(tqdm(pool.imap(row_delta,tasks,chunksize=4),total=len(w),desc=key,mininterval=10)):
                            centers[r]=c;costs+=loss
                        temp=Path(str(checkpoint)+'.tmp')
                        with temp.open('wb') as stream:
                            np.savez(stream,centers=centers,costs=costs,fingerprint=np.asarray(fingerprint))
                        temp.replace(checkpoint)
                    if not np.isfinite(centers).all():raise ValueError('Nonfinite centers')
                    fill_weight(module.weight,w,centers)
                    info=dict(weighted_sse=float(costs[0]),weight_sse=float(costs[1]),
                              centers=str(checkpoint),reused=reused,fingerprint=fingerprint)
                metrics[key]=dict(info,seconds=time.monotonic()-started)
                write_json(root/'metrics.json',metrics)
                config.update(completed_modules=len(metrics),last_module=key)
                write_json(root/'run.json',config)
                print('Completed',args.target,args.variant,key,round(metrics[key]['seconds'],1),flush=True)
    finally:
        if pool is not None:
            pool.close();pool.join()
    del layers,layer,module,original,w,q
    gc.collect()
    config['status']='evaluating';write_json(root/'run.json',config)
    torch.set_num_threads(2)
    tokens,path=get_awq_ppl_tokens('wikitext',cfg['model'])
    with torch.no_grad():
        ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
    result=dict(variant='rtn_uniform' if args.variant=='uniform' else 'rtn_delta',model=cfg['model'],
                ppl=ppl,bit=4,group_size=128,dataset='wikitext2_test',dataset_path=path,
                block_size=2048,batch_size=8,tokens=tokens.numel(),AWQ_scale=args.variant=='delta',
                run=str(root/'run.json'))
    write_json(root/'ppl.json',result)
    config.update(status='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(root/'run.json',config)
    print('PPL',args.target,args.variant,ppl,flush=True)


def validate_reused(value,target,variant):
    for k,v in dict(bit=4,group_size=128,block_size=2048,batch_size=8).items():
        if value.get(k)!=v:raise ValueError(f'Reused result protocol mismatch: {k}')
    if value.get('model') not in (TARGETS[target]['model'],TITLES[target]):
        raise ValueError('Reused result model mismatch')
    if value.get('dataset') not in ('wikitext2_test','WikiText-2 test'):
        raise ValueError('Reused dataset mismatch')
    if value.get('variant',value.get('method')) not in (variant,'rtn' if variant=='rtn_uniform' else variant):
        raise ValueError('Reused variant mismatch')


def suite(args,root):
    # Both new nonuniform models need about 6 GiB together, outside the existing
    # activation run's project-volume reservation.
    if shutil.disk_usage(root).free<7*1024**3:
        raise OSError('Need 7 GiB free for both RTN-delta checkpoint sets plus reserve')
    refs={('llama1','delta'):Path('/mnt/dfq/llama1_rtn_gptq_nonuniform_20260916/rtn_delta_ppl.json'),
          ('opt','uniform'):Path('/mnt/dfq/opt67b_w4_g128_rtn_gptq/rtn_ppl.json')}
    for (target,variant),path in refs.items():
        value=json.loads(path.read_text())
        validate_reused(value,target,'rtn_delta' if variant=='delta' else 'rtn_uniform')
    old=json.loads(Path('/mnt/dfq/llama1_rtn_gptq_nonuniform_20260916/run.json').read_text())
    if old.get('status')!='complete' or old.get('AWQ_scale') is not True:
        raise ValueError('Old RTN-delta run incomplete or missing AWQ scale')
    jobs={};procs={}
    def launch(target,variant):
        key=f'{target}_{variant}';out=root/key;log=root/f'{key}.log'
        command=[os.sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),
                 '--target',target,'--variant',variant,'--workers',str(args.workers)]
        with log.open('xb') as stream:
            p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        procs[key]=p
        jobs[key]=dict(pid=p.pid,target=target,variant=variant,output=str(out),log=str(log),command=command)
        write_json(root/'jobs.json',jobs)
        print('Launched',key,p.pid,flush=True)
    def aggregate():
        rows=[];complete=True
        for target in ('llama1','llama2','opt'):
            row=dict(model=TITLES[target])
            for variant in ('uniform','delta'):
                source=refs.get((target,variant),root/f'{target}_{variant}'/'ppl.json')
                if source.exists():
                    value=json.loads(source.read_text());row[variant]=value['ppl'];row[variant+'_source']=str(source)
                else:row[variant]=None;complete=False
            row['difference']=row['delta']-row['uniform'] if row['delta'] is not None and row['uniform'] is not None else None
            rows.append(row)
        write_json(root/'comparison.json',dict(status='complete' if complete else 'running',rows=rows,
                   protocol='W4-G128 WikiText-2 test block_size=2048 batch_size=8; RTN-delta includes AWQ scale'))
        lines=['| 模型 | 原始 RTN 均匀 | RTN-delta 非均匀 | 结果 |','|---|---:|---:|---|']
        for row in rows:
            fmt=lambda value:f'{value:.6f}' if value is not None else '运行中'
            outcome='待完成' if row['difference'] is None else ('改善' if row['difference']<0 else '持平' if row['difference']==0 else '变差')
            lines.append(f"| {row['model']} | {fmt(row['uniform'])} | {fmt(row['delta'])} | {outcome} |")
        (root/'comparison.md').write_text('\n'.join(lines)+'\n\nW4-G128，WikiText-2 test；RTN-delta 含 AWQ scale。\n')
        return complete
    launch('llama2','delta');launch('opt','delta')
    launch('llama1','uniform')
    aggregate()
    second_uniform_launched=False
    while True:
        if not second_uniform_launched and procs['llama1_uniform'].poll() is not None:
            launch('llama2','uniform');second_uniform_launched=True
        complete=aggregate()
        failed={k:p.returncode for k,p in procs.items() if p.poll() not in (None,0)}
        if failed:write_json(root/'failed_jobs.json',failed)
        if complete:
            print('Complete RTN comparison',root/'comparison.md',flush=True);return
        if second_uniform_launched and all(p.poll() is not None for p in procs.values()):
            raise RuntimeError(f'Jobs stopped without full comparison: {failed}')
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--target',choices=TARGETS)
    parser.add_argument('--variant',choices=['uniform','delta'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.suite and args.resume:parser.error('Resume a specific child job, not the suite')
    if not args.suite and (not args.target or not args.variant):parser.error('target and variant required')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=args.resume)
    config=dict(suite=args.suite,target=args.target,variant=args.variant,workers=args.workers,
                model=TARGETS[args.target]['model'] if args.target else None,
                bit=4,group_size=128,gamma=1,min_weight=.1,delta_lambda=1,power={'o':3,'rest':4},
                kmeans={'random_state':0,'n_init':'auto','max_iter':50},
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.resume:
        old=json.loads((root/'run.json').read_text())
        for k,v in config.items():
            if k!='workers' and old.get(k)!=v:raise ValueError(f'Resume configuration mismatch: {k}')
        if old.get('status')=='complete':print('Already complete',root);return
    config.update(status='running',started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(root/'run.json',config)
    try:
        if args.suite:
            suite(args,root)
            config.update(status='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            write_json(root/'run.json',config)
        else:execute_child(args,root,config)
    except BaseException as error:
        config.update(status='failed',error=repr(error));write_json(root/'run.json',config)
        (root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
