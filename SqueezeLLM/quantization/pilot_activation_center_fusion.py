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
    rng=np.random.default_rng(20260917);order=rng.permutation(len(articles))
    blocks=[];manifest=[];seen=set()
    for i in order:
        digest=hashlib.sha256(articles[i].encode()).hexdigest()
        if digest in seen:continue
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
    c0=torch.as_tensor(initial,device='cuda',dtype=torch.float32)
    idx=torch.as_tensor(labels,device='cuda',dtype=torch.long)
    scale=c0.std(dim=-1,keepdim=True).clamp_min(1e-5)
    offset=torch.zeros_like(c0,requires_grad=True)
    optimizer=torch.optim.Adam([offset],lr=.01)
    variance=((xfit@w.T).square().mean()).clamp_min(1e-12)
    @torch.no_grad()
    def evaluate(c,x):
        # Evaluate the FP16 weights actually used by fake-quant inference.
        e=reconstruct(c,idx).half().float()-w
        return float((x@e.T).square().mean())
    candidates=[]
    for step in range(33):
        c=c0+scale*offset
        if step in (0,8,32):
            candidates.append((evaluate(c,xselect),step,c.detach().clone()))
        if step==32:break
        error=reconstruct(c,idx)-w
        loss=(xfit@error.T).square().mean()/variance+1e-3*offset.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():offset.clamp_(-.5,.5)
    best=min(candidates,key=lambda item:(item[0],item[1]))
    result=dict(selected_step=best[1],select_output_mse=best[0],guard_output_mse=evaluate(best[2],xguard),
                initial_guard_output_mse=evaluate(c0,xguard),
                selection_trace=[dict(step=t,select_output_mse=v) for v,t,_ in candidates])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(model=MODEL,layers=LAYERS,output_rows=64,group_size=128,centers=16,
                loss='mean((X @ (Wq-W).T)**2) / reference_output_energy + 1e-3*mean(offset**2)',
                AWQ_scale=False,delta=False,steps=32,lr=.01,offset_bound=.5,selection_steps=[0,8,32],
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
            starts={'gptq':(gcodes,glabels),'rtn':(rcodes,rlabels),
                    'fusion':(lut[short]['centers'][:64],lut[short]['labels'][:64])}
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
