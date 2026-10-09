"""OPT W4-G128: native GPTQ, asymmetric min/max RTN, exact MSE fusion."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from group_center_selection import select_centers,assign
from fuse_group_centers import validate_record
from chunk_gptq_params import OPT_MODULES
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens
from squeezellm.fake_quant import apply_squeezellm_fake_quant


def rtn(x):
    lo=x.min(axis=-1);hi=x.max(axis=-1)
    scale=np.maximum(hi-lo,1e-5)/15
    zero=np.clip(-np.round(lo/scale),0,15)
    centers=scale[...,None]*(np.arange(16,dtype=np.float32)-zero[...,None])
    labels=np.clip(np.round(x/scale[...,None])+zero[...,None],0,15).astype(np.uint8)
    return centers,labels


def row_task(task):
    row,scales,zeros=task
    x=row.reshape(-1,128);rc,rl=rtn(x)
    gc=scales[:,None]*(np.arange(16,dtype=np.float32)-zeros[:,None])
    books=np.empty_like(rc);labels=np.empty_like(rl);totals=np.zeros(3)
    for g,w in enumerate(x):
        ones=np.ones(128)
        c=select_centers(w,ones,np.r_[rc[g],gc[g]]).astype(np.float32)
        lab,loss,_=assign(w,ones,c)
        er=float(np.square(w.astype(float)-rc[g,rl[g]]).sum())
        eg=assign(w,ones,gc[g])[1]
        if loss>min(er,eg)+1e-9:raise AssertionError('MSE increased')
        books[g]=c;labels[g]=lab;totals+=er,eg,loss
    return books,labels,totals


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--params',required=True)
    p.add_argument('--mode',choices=['build','gptq','rtn','selected','generated'],required=True)
    p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    root=Path(a.root);torch.set_num_threads(1)
    model=get_model('/mnt/dfq/pretrained_model/opt-6.7b').eval()
    layers=model.model.decoder.layers
    params=torch.load(a.params,map_location='cpu',mmap=True) if a.mode in ('build','gptq') else None
    if a.mode=='build':
        out=root/'selected';(out/'lut').mkdir(parents=True,exist_ok=False)
        (out/'kmeans_config.json').write_text(json.dumps(dict(weight_source='uniform_MSE',group_size=128,bit=4,AWQ_scale=False,delta=False)))
        metrics={}
        with get_context('spawn').Pool(a.workers) as pool:
            for l,layer in enumerate(layers):
                modules=dict(layer.named_modules());lut={}
                for short,suffix in OPT_MODULES.items():
                    w=modules[suffix].weight.detach().float().numpy()
                    record=params[f'model.decoder.layers.{l}.{suffix}'];validate_record(record,w.shape)
                    rows,cols=w.shape
                    centers=np.empty((rows,cols//128,16),np.float32);labels=np.empty((rows,cols//128,128),np.uint8)
                    tasks=((w[r],record['scale'][r].numpy(),record['zero'][r].numpy()) for r in range(rows))
                    totals=np.zeros(3)
                    for r,(c,idx,losses) in enumerate(tqdm(pool.imap(row_task,tasks),total=rows,desc=f'{l}.{short}')):
                        centers[r]=c;labels[r]=idx;totals+=losses
                    lut[short]=dict(format='squeezellm_group_lut_v1',group_size=128,centers=centers,labels=labels)
                    metrics[f'{l}.{short}']=dict(zip(('rtn_sse','gptq_nearest_sse','selected_sse'),totals.tolist()))
                    (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
                with (out/'lut'/f'l{l}.pkl').open('wb') as f:pickle.dump(lut,f,protocol=pickle.HIGHEST_PROTOCOL)
        (root/'complete.json').write_text(json.dumps(dict(layers=len(layers),status='complete')))
        return
    target=root/f'{a.mode}_ppl.json'
    if target.exists():raise FileExistsError(target)
    with torch.no_grad():
        if a.mode in ('selected','generated'):
            if not (root/'complete.json').exists():raise ValueError('Incomplete LUT')
            apply_squeezellm_fake_quant(model,str(root/a.mode),include_sparse=False)
        else:
            for l,layer in enumerate(layers):
                modules=dict(layer.named_modules())
                for suffix in OPT_MODULES.values():
                    module=modules[suffix]
                    if a.mode=='gptq':
                        record=params[f'model.decoder.layers.{l}.{suffix}'];validate_record(record,module.weight.shape)
                        module.weight.copy_(record['weight'])
                    else:
                        w=module.weight.detach().float().numpy();shape=w.shape
                        c,idx=rtn(w.reshape(shape[0],-1,128))
                        q=np.take_along_axis(c,idx.astype(int),axis=-1).reshape(shape)
                        module.weight.copy_(torch.from_numpy(q))
            del params
        tokens,path=get_awq_ppl_tokens('wikitext','/mnt/dfq/pretrained_model/opt-6.7b')
        ppl=awq_compatible_ppl_eval(model,tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
    result=dict(method=a.mode,ppl=ppl,model='OPT-6.7B',bit=4,group_size=128,block_size=2048,batch_size=8,dataset='WikiText-2 test',dataset_path=path,gptq_params=a.params,objective='MSE',AWQ_scale=False,delta=False)
    target.write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
