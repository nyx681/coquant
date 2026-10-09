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
    excluded={item['sha256'] for item in previous}
    rng=np.random.default_rng(20260918);order=rng.permutation(len(articles))
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
    c0=torch.as_tensor(initial,device='cuda',dtype=torch.float32)
    idx=torch.as_tensor(labels,device='cuda',dtype=torch.long)
    step=c0.std(dim=-1,keepdim=True).clamp_min(1e-5)
    rows,groups,k=c0.shape
    def evaluate(c,x):
        e=reconstruct(c,idx).half().float()-w
        return float((x@e.T).square().mean())
    # A[r,t,g,k] sums input channels assigned to center k of group g.
    # Solve in token space: delta = A.T (A A.T + ridge I)^-1 residual.
    c1=c0.clone();normal_checks=[]
    with torch.no_grad():
        for begin in range(0,rows,8):
            end=min(rows,begin+8);batch=end-begin;t=xfit.shape[0]
            design=torch.zeros((batch,t,groups,k),device='cuda')
            ids=idx[begin:end,None,:,:].expand(batch,t,groups,128)
            source=xfit.reshape(1,t,groups,128).expand(batch,t,groups,128)
            design.scatter_add_(3,ids,source)
            design=(design*step[begin:end,None,:,:]).reshape(batch,t,groups*k)
            residual=(xfit@(w[begin:end]-reconstruct(c0[begin:end],idx[begin:end])).T).T
            gram=design@design.transpose(1,2)
            ridge=1e-3*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
            gram.diagonal(dim1=-2,dim2=-1).add_(ridge[:,None])
            dual=torch.linalg.solve(gram,residual[:,:,None])
            delta=(design.transpose(1,2)@dual).reshape(batch,groups,k)
            if not torch.isfinite(delta).all():raise ValueError('Nonfinite ridge solution')
            c1[begin:end]=c0[begin:end]+step[begin:end]*delta
            normal_checks.append(float((gram@dual-residual[:,:,None]).abs().max()))
        candidates=[(evaluate(c0,xselect),0,c0),(evaluate(c1,xselect),1,c1)]
        best=min(candidates,key=lambda item:(item[0],item[1]))
        return dict(selected_step=best[1],select_output_mse=best[0],
                    guard_output_mse=evaluate(best[2],xguard),initial_guard_output_mse=evaluate(c0,xguard),
                    fit_before=evaluate(c0,xfit),fit_after=evaluate(c1,xfit),
                    solve_max_residual=max(normal_checks),
                    selection_trace=[dict(step=t,select_output_mse=v) for v,t,_ in candidates])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(model=MODEL,layers=LAYERS,output_rows=64,group_size=128,centers=16,
                loss='ridge regression in normalized center offsets; ridge=1e-3*mean(diag(A@A.T))',
                AWQ_scale=False,delta=False,solver='dual ridge least squares',selection_steps=[0,1],
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
