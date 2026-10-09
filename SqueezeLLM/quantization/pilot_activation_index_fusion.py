"""Calibration-only OPT pilot: equal-budget activation reconstruction of LUT centers.

Three article-disjoint WikiText train subsets, no WikiText test access.
Fixed code assignments, 16 movable centers/G128, no AWQ scale/delta (uniform arm).
"""
import os
os.environ['OMP_NUM_THREADS']='4'
import argparse,json,pickle,re,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from llama import get_model
from chunk_gptq_params import OPT_MODULES
from opt_rtn_gptq_fusion import rtn

MODEL='/mnt/dfq/pretrained_model/opt-6.7b'
BASE=Path('/mnt/dfq/opt67b_w4_g128_rtn_gptq')
LAYERS=(0,15,31)


def calibration(tokenizer):
    rows=load_dataset('/home/public/dataset/wikitext-2-raw-v1',split='train')['text']
    articles=[];current=[]
    for text in rows:
        if re.match(r'^\s*= [^=].*[^=] =\s*$',text) and current:
            articles.append(''.join(current));current=[]
        current.append(text)
    if current:articles.append(''.join(current))
    previous=json.loads(Path('/mnt/dfq/opt67b_activation_fusion_pilot_20260917/calibration.json').read_text())
    previous+=json.loads(Path('/mnt/dfq/opt67b_activation_ridge_pilot_20260917/calibration.json').read_text())
    excluded={item['sha256'] for item in previous}
    rng=np.random.default_rng(20260919);order=rng.permutation(len(articles))
    blocks=[];manifest=[];seen=set()
    for i in order:
        digest=hashlib.sha256(articles[i].encode()).hexdigest()
        if digest in seen or digest in excluded:continue
        ids=tokenizer(articles[i],return_tensors='pt').input_ids
        if ids.shape[1]<512:continue
        split='fit' if len(blocks)<8 else 'select' if len(blocks)<12 else 'guard'
        blocks.append((split,ids[:,:512]));manifest.append(dict(article=int(i),sha256=digest,split=split,token_start=0,tokens=512));seen.add(digest)
        if len(blocks)==16:break
    if len(blocks)!=16:raise ValueError('Not enough calibration articles')
    return blocks,manifest


def reconstruct(c,labels):
    return torch.gather(c,2,labels).flatten(1)


def optimize(initial,labels,w,xfit,xselect,xguard):
    c=torch.as_tensor(initial,device='cuda',dtype=torch.float32).half().float()
    idx=torch.as_tensor(labels,device='cuda',dtype=torch.long)
    with torch.no_grad():
        q0=reconstruct(c,idx);q=q0.clone()
        def score(weight,x):return float((x@(weight-w).T).square().mean())
        residual=xfit@(q-w).T
        norms=xfit.square().sum(dim=0).clamp_min(1e-12)
        # Each output row has an independent least-squares objective. Given all
        # other codes, the scalar optimum is q_j - X_j.T residual / ||X_j||^2.
        for j in range(w.shape[1]):
            col=xfit[:,j]
            optimum=q[:,j]-(col@residual)/norms[j]
            candidates=c[:,j//128,:]
            code=(candidates-optimum[:,None]).square().argmin(dim=1)
            replacement=candidates.gather(1,code[:,None]).squeeze(1)
            change=replacement-q[:,j]
            residual+=col[:,None]*change[None,:]
            q[:,j]=replacement
        options=[(score(q0,xselect),0,q0),(score(q,xselect),1,q)]
        best=min(options,key=lambda v:(v[0],v[1]))
        before=score(q0,xfit);after=score(q,xfit)
        if after>before*(1+1e-4)+1e-8:raise AssertionError('Coordinate sweep increased fit error')
        return dict(selected_step=best[1],select_output_mse=best[0],
                    guard_output_mse=score(best[2],xguard),initial_guard_output_mse=score(q0,xguard),
                    fit_before=before,fit_after=after,changed_fraction=float((q!=q0).float().mean()),
                    selection_trace=[dict(step=t,select_output_mse=v) for v,t,_ in options])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(model=MODEL,layers=LAYERS,output_rows=64,group_size=128,centers=16,
                loss='activation output SSE; one exact coordinate sweep of codes with fixed centers',
                AWQ_scale=False,delta=False,solver='activation-aware code coordinate descent',fusion_initial_codes='nearest to native GPTQ weights in union-selected codebook',selection_steps=[0,1],
                data='article-disjoint WikiText train',test_access=False,status='running')
    (root/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=False)
    blocks,manifest=calibration(tokenizer);(root/'calibration.json').write_text(json.dumps(manifest,indent=2))
    model=get_model(MODEL).eval().cuda();captured={};original={};handles=[];active=['fit']
    for l in LAYERS:
        mods=dict(model.model.decoder.layers[l].named_modules())
        for short,suffix in OPT_MODULES.items():
            key=f'{l}.{short}';original[key]=mods[suffix].weight[:64].detach().float().cpu().clone()
            captured[key]={s:[] for s in ('fit','select','guard')}
            def hook(module,args,key=key):
                x=args[0].detach().reshape(-1,args[0].shape[-1])
                captured[key][active[0]].append(x[::8].float().cpu())
            handles.append(mods[suffix].register_forward_pre_hook(hook))
    with torch.no_grad():
        for split,ids in blocks:
            active[0]=split;model(ids.cuda(),use_cache=False)
    for h in handles:h.remove()
    del model,mods,handles;gc.collect();torch.cuda.empty_cache()
    params=torch.load(BASE/'static_gptq_params.pt',map_location='cpu',mmap=True)
    results={}
    for l in LAYERS:
        with (BASE/'selected/lut'/f'l{l}.pkl').open('rb') as f:lut=pickle.load(f)
        for short,suffix in OPT_MODULES.items():
            key=f'{l}.{short}';w=original[key].cuda();x={s:torch.cat(v).cuda() for s,v in captured[key].items()}
            if not all(torch.isfinite(v).all() for v in x.values()):raise ValueError('Nonfinite activation')
            record=params[f'model.decoder.layers.{l}.{suffix}']
            scale=record['scale'][:64].numpy();zero=record['zero'][:64].numpy()
            gcodes=scale[:,:,None]*(np.arange(16,dtype=np.float32)-zero[:,:,None])
            native=record['weight'][:64].float().numpy().reshape(64,-1,128)
            glabels=np.clip(np.round(native/scale[:,:,None]+zero[:,:,None]),0,15).astype(np.uint8)
            rcodes,rlabels=rtn(w.cpu().numpy().reshape(64,-1,128))
            fusion_centers=lut[short]['centers'][:64]
            fusion_labels=np.abs(native[:,:,:,None]-fusion_centers[:,:,None,:]).argmin(axis=-1).astype(np.uint8)
            starts={'gptq':(gcodes,glabels),'rtn':(rcodes,rlabels),
                    'fusion':(fusion_centers,fusion_labels)}
            results[key]={}
            for method,(centers,labels) in starts.items():
                results[key][method]=optimize(centers,labels,w,x['fit'],x['select'],x['guard'])
            # Native GPTQ is scored directly, preserving any export rounding.
            with torch.no_grad():
                e=torch.as_tensor(native.reshape(64,-1),device='cuda')-w
                results[key]['native_gptq_guard_output_mse']=float((x['guard']@e.T).square().mean())
            (root/'results.json').write_text(json.dumps(results,indent=2))
            print(key,json.dumps(results[key]),flush=True)
            del w,x
    count=sum(v['fusion']['guard_output_mse']<min(v['gptq']['guard_output_mse'],v['rtn']['guard_output_mse']) for v in results.values())
    config.update(status='complete',fusion_guard_wins=count,modules=len(results))
    (root/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)


if __name__=='__main__':main()
