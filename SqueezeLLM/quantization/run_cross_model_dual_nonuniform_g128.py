"""AWQ/GPTQ nonuniform G128 endpoints and exact union selection, RAM only.

Shared AWQ scale prior, channel inverse deltas, geometric modulation objective.
No test-driven parameter search, no center refinement, no sparse component.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import pickle
import gc
import json
import traceback
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from nuq import kmeans_fit, build_inverse_channel_relative_weights, normalize_score_row, load_awq_channel_scales
from group_center_selection import assign, select_centers
from chunk_gptq_params import LLAMA_MODULES, OPT_MODULES
from fuse_group_centers import validate_record
from llama import get_model, awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens
from squeezellm.model_parse import get_layers

VARIANTS=('awq_delta','gptq_delta','selected')
TARGETS={
 'llama2':dict(model='/mnt/dfq/pretrained_model/llama-2-7b',model_type='llama',
   original='/mnt/dfq/squeezellm_mixed/model_chunks/llama-2-7b',
   awq='/mnt/dfq/squeezellm_awq_delta/llama2/awq_dequant_chunks',
   params='/home/zch/project/Quantification/llama2_static_g128_retry_20260916/llama-2-7b-w4-g128-static-gptq_params.pt',
   scale='/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-2-7b-w4-g128.pt'),
 'opt':dict(model='/mnt/dfq/pretrained_model/opt-6.7b',model_type='opt',
   original='/home/zch/project/Quantification/opt67b_delta_ds/shared/original_chunks',
   awq='/home/zch/project/Quantification/opt67b_delta_ds/w4/awq_dequant_chunks',
   params='/mnt/dfq/opt67b_w4_g128_rtn_gptq/static_gptq_params.pt',
   scale='/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/opt-6.7b-w4-g128.pt')}


def write_json(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)


def row(task):
    w,sa,sg,shared=task[:4]
    cached=task[4] if len(task)==5 else None
    out=np.empty((3,len(w)),np.float32);metrics=np.zeros(3)
    for start in range(0,len(w),128):
        sl=slice(start,start+128);x=w[sl];s=shared[sl]
        if cached is None:
            ca,_=kmeans_fit((x[:,None],sa[sl],16))
        else:
            ca=cached[start//128]
        cg,_=kmeans_fit((x[:,None],sg[sl],16))
        chosen=select_centers(x,s,np.r_[ca,cg]).astype(np.float32)
        costs=[]
        for i,c in enumerate((ca,cg,chosen)):
            idx,loss,_=assign(x,s,c);out[i,sl]=c[idx];costs.append(loss)
        if costs[2]>min(costs[:2])+1e-10+1e-9*max(costs):
            raise AssertionError('Exact selection increased common SSE')
        metrics+=costs
    return out,metrics


def execute(a,root,config):
    cfg=TARGETS[a.target];modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
    prefix='model.decoder.layers' if cfg['model_type']=='opt' else 'model.layers'
    for key in ('model','original','awq','params','scale'):
        if not Path(cfg[key]).exists():raise FileNotFoundError(cfg[key])
    for key in ('original','awq'):
        if {p.name for p in Path(cfg[key]).glob('layer_*.pt')}!={f'layer_{i}.pt' for i in range(32)}:
            raise ValueError(f'Incomplete chunks: {key}')
    cached_root=Path(a.reuse_awq) if a.reuse_awq else None
    reused_result=None
    if cached_root is not None:
        old=json.loads((cached_root/'awq/kmeans_config.json').read_text())
        expected=dict(original_chunks=cfg['original'],awq_chunks=cfg['awq'],awq_scale_cache=cfg['scale'],
                      bit=4,group_size=128,gamma=1,min_weight=.1,delta_lambda=1,
                      power={'o':3,'rest':4},status='complete',partial=False,variant='awq',
                      weight_source='awq_scale_inverse_awq_delta')
        for key,value in expected.items():
            if old.get(key)!=value:raise ValueError(f'Incompatible AWQ cache: {key}')
        if old.get('kmeans')!={'random_state':0,'n_init':'auto','max_iter':50}:raise ValueError('Kmeans mismatch')
        for l in range(32):
            if not (cached_root/'awq/lut'/f'l{l}.pkl').exists():raise FileNotFoundError(f'Missing cached layer {l}')
        result_path=cached_root/'ppl/awq_wikitext.json'
        reused_result=json.loads(result_path.read_text())
        for key,value in dict(model=cfg['model'],group_size=128,bit=4,block_size=2048,batch_size=8,dataset='wikitext2_test',artifact=str(cached_root/'awq')).items():
            if reused_result.get(key)!=value:raise ValueError(f'Incompatible AWQ PPL: {key}')
        reused_result=dict(reused_result,variant='awq_delta',reused_from=str(result_path))
        write_json(root/'awq_delta_ppl.json',reused_result)
    torch.set_num_threads(1)
    scales=load_awq_channel_scales(cfg['scale'])
    params=torch.load(cfg['params'],map_location='cpu',mmap=True)
    if len(params)!=32*len(modules):raise ValueError('GPTQ record count mismatch')
    models=[get_model(cfg['model']).eval() for _ in VARIANTS]
    layer_lists=[get_layers(model,cfg['model_type']) for model in models]
    if any(len(layers)!=32 for layers in layer_lists):raise ValueError('Unexpected layer count')
    metrics={}
    with get_context('spawn').Pool(a.workers) as pool:
        for l in range(32):
            original=torch.load(Path(cfg['original'])/f'layer_{l}.pt',map_location='cpu')
            awq=torch.load(Path(cfg['awq'])/f'layer_{l}.pt',map_location='cpu')
            cached_lut=None
            if cached_root is not None:
                with (cached_root/'awq/lut'/f'l{l}.pkl').open('rb') as f:cached_lut=pickle.load(f)
            targets=[dict(layers[l].named_modules()) for layers in layer_lists]
            for name,suffix in modules.items():
                w=original[name].float().numpy();aw=awq[name].float().numpy()
                if w.shape!=aw.shape or not np.isfinite(w).all() or not np.isfinite(aw).all():raise ValueError('Invalid reference weights')
                # Ensure chunks belong to the evaluated base model before replacement.
                if not torch.equal(targets[0][suffix].weight.detach().cpu(),original[name].to(targets[0][suffix].weight.dtype)):
                    raise ValueError(f'Original chunks differ from model: {l}.{name}')
                record=params[f'{prefix}.{l}.{suffix}'];error=validate_record(record,w.shape)
                native=record['weight'].float().numpy()
                scale=scales[l,name].numpy()
                if scale.shape!=(w.shape[1],) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid scales')
                da=build_inverse_channel_relative_weights(w,np.abs(w-aw),gamma=1,min_weight=.1)
                dg=build_inverse_channel_relative_weights(w,np.abs(w-native),gamma=1,min_weight=.1)
                prior=normalize_score_row(scale)**(3 if name=='o' else 4)
                ma=1+normalize_score_row(da);mg=1+normalize_score_row(dg)
                sa=prior*ma;sg=prior*mg;shared=prior*np.sqrt(ma*mg)
                if any(not np.isfinite(s).all() or np.any(s<=0) for s in (sa,sg,shared)):raise ValueError('Invalid sample weights')
                total=np.zeros(3)
                cached_centers=None if cached_lut is None else cached_lut[name]['centers']
                if cached_centers is not None and (cached_centers.shape!=(len(w),w.shape[1]//128,16) or not np.isfinite(cached_centers).all()):
                    raise ValueError('Invalid cached centers')
                tasks=((w[r],sa,sg,shared) if cached_centers is None else (w[r],sa,sg,shared,cached_centers[r]) for r in range(len(w)))
                for r,(reconstructed,costs) in enumerate(tqdm(pool.imap(row,tasks),total=len(w),desc=f'{l}.{name}',mininterval=5)):
                    with torch.no_grad():
                        for i in range(3):targets[i][suffix].weight[r].copy_(torch.from_numpy(reconstructed[i]))
                    total+=costs
                metrics[f'{l}.{name}']=dict(rows=len(w),groups=w.size//128,gptq_reconstruction_error=error,
                  common_sse=dict(zip(VARIANTS,total.tolist())))
                write_json(root/'metrics.json',metrics)
    del params,original,awq,targets,layer_lists,record,w,aw,native,tasks
    gc.collect()
    config['status']='evaluating';write_json(root/'run.json',config)
    torch.set_num_threads(4)
    tokens,path=get_awq_ppl_tokens('wikitext',cfg['model'])
    results=[]
    for i,v in enumerate(VARIANTS):
        if v=='awq_delta' and reused_result is not None:
            results.append(reused_result);models[i]=None;gc.collect();continue
        with torch.no_grad():
            ppl=awq_compatible_ppl_eval(models[i],tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
        result=dict(variant=v,ppl=ppl,model=cfg['model'],bit=4,group_size=128,
          dataset='wikitext2_test',dataset_path=path,batch_size=8,block_size=2048,
          tokens=tokens.numel(),AWQ_scale=True,run=str(root/'run.json'))
        write_json(root/f'{v}_ppl.json',result);results.append(result)
        models[i]=None;gc.collect();torch.cuda.empty_cache()
    write_json(root/'three_ppl.json',results)
    config['status']='complete';write_json(root/'run.json',config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=TARGETS,required=True)
    p.add_argument('--output',required=True);p.add_argument('--reuse_awq');p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(TARGETS[a.target],bit=4,group_size=128,gamma=1,min_weight=.1,lambda_=1,
      power={'o':3,'rest':4},objective='scale_norm**p * sqrt((1+dA_norm)*(1+dG_norm))',
      AWQ_scale=True,delta_aggregation='module input channels',seed=0,kmeans_max_iter=50,
      kmeans_n_init='auto',variants=VARIANTS,workers=a.workers,status='running',
      storage='RAM float simulation; no LUT checkpoint',exploratory=True,reuse_awq=a.reuse_awq)
    write_json(root/'run.json',config)
    try:execute(a,root,config)
    except BaseException as error:
        config['status']='failed';config['error']=repr(error);write_json(root/'run.json',config)
        (root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
