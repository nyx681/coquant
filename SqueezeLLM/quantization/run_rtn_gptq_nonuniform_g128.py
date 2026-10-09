"""LLaMA-1 RTN/GPTQ dual nonuniform G128, streaming float weights to RAM, no new LUTs."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from nuq import kmeans_fit,build_inverse_channel_relative_weights,normalize_score_row,load_awq_channel_scales
from group_center_selection import assign,select_centers
from opt_rtn_gptq_fusion import rtn
from chunk_gptq_params import LLAMA_MODULES
from llama import get_model,awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens

BASE=Path('/home/zch/project/Quantification/llama1_w4_g128_center_fusion/full_20260914')
MODEL='/mnt/dfq/pretrained_model/Llama-7b'
VARIANTS=('rtn_delta','gptq_delta','selected','refined')


def row(task):
    w,sg,shared,sr=task
    out=np.empty((4,len(w)),np.float32);metrics=np.zeros(4)
    for g in range(len(w)//128):
        sl=slice(g*128,(g+1)*128);x=w[sl];s=shared[sl]
        a,_=kmeans_fit((x[:,None],sr[sl],16))
        b,_=kmeans_fit((x[:,None],sg[sl],16))
        chosen=select_centers(x,s,np.r_[a,b]).astype(np.float32)
        refined=chosen.copy();best=assign(x,s,refined)[1]
        for _ in range(50):
            idx,_,_=assign(x,s,refined);new=refined.copy()
            for k in range(16):
                mask=idx==k
                if mask.any():new[k]=np.average(x[mask].astype(float),weights=s[mask])
            loss=assign(x,s,new)[1]
            if loss>=best:break
            refined,best=new,loss
        costs=[assign(x,s,c)[1] for c in (a,b,chosen,refined)]
        if costs[2]>min(costs[:2])+1e-9 or costs[3]>costs[2]+1e-9:raise AssertionError('Common objective increased')
        metrics+=costs
        for i,c in enumerate((a,b,chosen,refined)):
            idx=assign(x,s,c)[0];out[i,sl]=c[idx]
    return out,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(model=MODEL,bit=4,group_size=128,source="RTN minmax G128 and GPTQ fixed.pt",gamma=1,min_weight=.1,
                lambda_=1,power={'o':3,'rest':4},objective='scale_norm**p * sqrt((1+dR_norm)*(1+dG_norm))',
                AWQ_scale=True,delta_aggregation='module input channels',max_iter=50,seed=0,
                storage='RAM float simulation; no new LUT checkpoint',status='running')
    (root/'run.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(1)
    scales=load_awq_channel_scales('/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt')
    params=torch.load('/mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt',map_location='cpu',mmap=True)
    models=[get_model(MODEL).eval() for _ in VARIANTS];metrics={}
    with get_context('spawn').Pool(a.workers) as pool:
        for l in range(32):
            original=torch.load(f'/mnt/dfq/squeezellm_awq_delta/original_chunks/layer_{l}.pt',map_location='cpu')
            targets=[dict(m.model.layers[l].named_modules()) for m in models]
            for name,suffix in LLAMA_MODULES.items():
                w=original[name].float().numpy()
                native=params[f'model.layers.{l}.{suffix}']['weight'].float().numpy()
                rc,ri=rtn(w.reshape(len(w),-1,128))
                rw=np.take_along_axis(rc,ri.astype(np.int64),axis=-1).reshape(w.shape)
                da=build_inverse_channel_relative_weights(w,np.abs(w-rw),gamma=1,min_weight=.1)
                dg=build_inverse_channel_relative_weights(w,np.abs(w-native),gamma=1,min_weight=.1)
                prior=normalize_score_row(scales[l,name].numpy())**(3 if name=='o' else 4)
                ma=1+normalize_score_row(da);mg=1+normalize_score_row(dg)
                sg=prior*mg;sr=prior*ma;shared=prior*np.sqrt(ma*mg)
                if not np.isfinite(shared).all() or not np.isfinite(sg).all():raise ValueError('Invalid weights')
                total=np.zeros(4)
                tasks=((w[r],sg,shared,sr) for r in range(len(w)))
                for r,(reconstructed,costs) in enumerate(tqdm(pool.imap(row,tasks),total=len(w),desc=f'{l}.{name}')):
                    with torch.no_grad():
                        for i in range(4):targets[i][suffix].weight[r].copy_(torch.from_numpy(reconstructed[i]))
                    total+=costs
                metrics[f'{l}.{name}']=dict(zip(('rtn_common_sse','gptq_common_sse','selected_common_sse','refined_common_sse'),total.tolist()))
                (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
    del params,original,targets
    torch.set_num_threads(4)
    tokens,path=get_awq_ppl_tokens('wikitext',MODEL)
    results=[]
    for i,v in enumerate(VARIANTS):
        with torch.no_grad():
            ppl=awq_compatible_ppl_eval(models[i],tokens,torch.device('cuda:0'),'wikitext',block_size=2048,batch_size=8)
        result=dict(variant=v,ppl=ppl,model=MODEL,bit=4,group_size=128,dataset='wikitext2_test',dataset_path=path,batch_size=8,block_size=2048)
        (root/f'{v}_ppl.json').write_text(json.dumps(result,indent=2));results.append(result)
        models[i]=None;torch.cuda.empty_cache()
    (root/'four_ppl.json').write_text(json.dumps(results,indent=2))
    config['status']='complete';(root/'run.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
