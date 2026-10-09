"""Build reusable original/AWQ/GPTQ chunks with validated static W3-G128 GPTQ."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, gc, hashlib, json, random, re, sys, time, traceback
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from llama import get_model
from squeezellm.model_parse import get_layers
from chunk_gptq_params import LLAMA_MODULES, OPT_MODULES
from validate_w3_gptq import validate_record
from run_llama1_activation_fusion import atomic_torch_save, write_json
from run_cross_model_delta_only_fusions import gpu_slot


def load_reference_scales(path, modules):
    cache=torch.load(path,map_location='cpu')
    if not isinstance(cache.get('scale'),list):raise ValueError('Invalid AWQ cache')
    suffixes={v.rsplit('.',1)[-1]:k for k,v in modules.items()}
    result={}
    for item in cache['scale']:
        _,names,scale=item[:3]
        # AWQ search explicitly stores None for unscaled GQA projections.
        # entry.py maps these records to identity scale in its export path.
        if scale is None:continue
        for name in names:
            match=re.search(r'(?:^|\.)layers\.(\d+)\.',name)
            short=suffixes.get(name.rsplit('.',1)[-1])
            if match and short:
                key=(int(match.group(1)),short);value=torch.as_tensor(scale).float().cpu()
                if key in result and not torch.equal(result[key],value):raise ValueError('Conflicting AWQ scales')
                result[key]=value
    return result


def calibration(model_path, nsamples):
    tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=False)
    dataset=load_dataset('/mnt/dfq/datasets/mit-han-lab/pile-val-backup',split='validation').shuffle(seed=0)
    rng=random.Random(0); blocks=[]; intervals=[]
    for index,item in enumerate(dataset):
        text=item['text'].strip()
        if not text:continue
        ids=tokenizer(text,return_tensors='pt').input_ids
        if ids.shape[1]<=2048:continue
        start=rng.randint(0,ids.shape[1]-2048)
        blocks.append(ids[:,start:start+2048]);intervals.append([index,start])
        if len(blocks)==nsamples:break
    if len(blocks)!=nsamples:raise ValueError('Insufficient calibration texts')
    return blocks,dict(dataset='local Pile validation',seed=0,nsamples=nsamples,seqlen=2048,
                       shuffled_document_and_start=intervals,
                       token_sha256=hashlib.sha256(torch.cat(blocks).numpy().tobytes()).hexdigest())


@torch.no_grad()
def execute(a,cfg,root,update):
    sys.path.insert(0,'/home/zch/project/awq_code/llm-awq_zch0722')
    from awq.quantize.quantizer import _fake_awq_candidate
    sys.path.insert(0,'/home/zch/project/gptq_zch0722')
    from gptq import GPTQ
    from quant import Quantizer
    torch.set_num_threads(2);torch.manual_seed(0)
    modules=OPT_MODULES if cfg['model_type']=='opt' else LLAMA_MODULES
    with gpu_slot(a.gpu_lock,update):
        update(status='loading_model')
        model=get_model(cfg['model']).eval()
        layers=get_layers(model,cfg['model_type'])
        if len(layers)!=cfg['layers']:raise ValueError('Model layer count mismatch')
        scales=load_reference_scales(cfg['scale'],modules)
        fallback=[];quantized={}
        for field in ('original','awq','gptq'):
            Path(cfg[field]).mkdir(parents=True,exist_ok=True)
        for l,layer in enumerate(layers):
            original=Path(cfg['original'])/f'layer_{l}.pt'
            awq=Path(cfg['awq'])/f'layer_{l}.pt'
            update(status='preparing_awq',current_layer=l)
            weights={n:layer.get_submodule(s).weight.detach().cpu().contiguous() for n,s in modules.items()}
            if not original.exists():atomic_torch_save(weights,original)
            else:
                cached=torch.load(original,map_location='cpu')
                if set(cached)!=set(weights) or any(not torch.equal(cached[n],w) for n,w in weights.items()):
                    raise ValueError('Original checkpoint mismatch')
                del cached
            fallback.extend(f'{l}.{name}' for name in weights if (l,name) not in scales)
            if awq.exists():continue
            quantized={}
            for name,w in weights.items():
                scale=scales.get((l,name))
                if scale is None:
                    # Match the repository's AWQ-only mixed-source export for
                    # GQA output projections with no scale search entry.
                    scale=torch.ones(w.shape[1],dtype=w.dtype)
                if scale.numel()!=w.shape[1] or not torch.isfinite(scale).all() or (scale<=0).any():
                    raise ValueError(f'Invalid AWQ scale: {l}.{name}')
                quantized[name]=_fake_awq_candidate(w.cuda(),scale.cuda(),3,True,128).cpu()
                if not torch.isfinite(quantized[name]).all():raise ValueError('Nonfinite AWQ reference')
            atomic_torch_save(quantized,awq)
            print('AWQ reference',l,flush=True)
        write_json(root/'awq_reference.json',dict(source_cache=cfg['scale'],
                   definition='repository _fake_awq_candidate: quantize(W*a)/a',identity_scale_modules=fallback))
        del weights,quantized,layer
        # w is absent when every AWQ layer is reused.
        if 'w' in locals():del w
        gc.collect();torch.cuda.empty_cache()
        update(status='preparing_gptq_calibration')
        blocks,manifest=calibration(cfg['model'],a.nsamples)
        manifest.update(wbits=3,groupsize=128,sym=False,percdamp=.01,act_order=True,
                        static_groups=True,true_sequential=cfg['model_type']!='opt')
        write_json(root/'gptq_calibration.json',manifest)
        model.config.use_cache=False
        model.cuda()
        inps=torch.zeros((a.nsamples,2048,model.config.hidden_size),device='cuda',dtype=next(model.parameters()).dtype)
        captured={}
        class CaptureDone(Exception):pass
        def catch(module,args,kwargs):
            inps[captured['index']].copy_(args[0][0])
            captured['kwargs']={k:v for k,v in kwargs.items() if k!='use_cache'}
            raise CaptureDone()
        handle=layers[0].register_forward_pre_hook(catch,with_kwargs=True)
        for index,tokens in enumerate(blocks):
            captured['index']=index
            try:model.model(tokens.cuda(),use_cache=False)
            except CaptureDone:pass
            else:raise RuntimeError('Failed to capture first layer input')
        handle.remove();model.cpu();torch.cuda.empty_cache()
        kwargs=captured['kwargs'];outs=torch.empty_like(inps)
        for l,layer in enumerate(layers):
            update(status='building_gptq',current_layer=l,completed_layers=l)
            layer.cuda();path=Path(cfg['gptq'])/f'layer_{l}.pt';params_path=root/f'gptq_layer_{l}_params.pt'
            if path.exists() and params_path.exists():
                records=torch.load(params_path,map_location='cpu');cached=torch.load(path,map_location='cpu')
                if set(cached)!=set(modules) or set(records)!=set(modules):raise ValueError('Incomplete GPTQ checkpoint')
                for name,suffix in modules.items():
                    validate_record(dict(records[name],weight=cached[name]),cached[name].shape)
                    layer.get_submodule(suffix).weight.copy_(cached[name].cuda())
                del cached,records
            else:
                sequences=[list(modules)] if cfg['model_type']=='opt' else [['k','v','q'],['o'],['up','gate'],['down']]
                records={}
                for names in sequences:
                    quantizers={};handles=[]
                    for name in names:
                        module=layer.get_submodule(modules[name]);g=GPTQ(module);g.quantizer=Quantizer()
                        g.quantizer.configure(3,perchannel=True,sym=False,mse=False)
                        quantizers[name]=g
                        def hook(module,inputs,output,name=name):quantizers[name].add_batch(inputs[0].data,output.data)
                        handles.append(module.register_forward_hook(hook))
                    for j in range(a.nsamples):layer(inps[j].unsqueeze(0),**kwargs)
                    for h in handles:h.remove()
                    for name,g in quantizers.items():
                        print('GPTQ',l,name,flush=True)
                        g.fasterquant(percdamp=.01,groupsize=128,actorder=True,static_groups=True)
                        q=g.quantizer
                        record=dict(scale=q.scale.detach().cpu(),zero=q.zero.detach().cpu(),maxq=int(q.maxq),
                                    group_size=128,static_groups=True,g_idx=q.gptq_g_idx.detach().cpu())
                        weight=layer.get_submodule(modules[name]).weight.detach().cpu()
                        validate_record(dict(record,weight=weight),weight.shape)
                        records[name]=record;g.free()
                    del quantizers,handles,g,q,weight
                atomic_torch_save(records,params_path)
                atomic_torch_save({n:layer.get_submodule(s).weight.detach().cpu().contiguous() for n,s in modules.items()},path)
                del records
            for j in range(a.nsamples):outs[j]=layer(inps[j].unsqueeze(0),**kwargs)[0]
            layer.cpu();inps,outs=outs,inps;torch.cuda.empty_cache()
            print('GPTQ layer complete',l,flush=True)
            if a.smoke_layers and l+1>=a.smoke_layers:
                update(status='smoke_complete',completed_layers=l+1);return
        update(status='complete',completed_layers=len(layers),finished_at=time.strftime('%F %T'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--target-config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu-lock',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--nsamples',type=int,default=128);p.add_argument('--smoke-layers',type=int,default=0);a=p.parse_args()
    cfg=json.loads(Path(a.target_config).read_text());root=Path(a.output);root.mkdir(parents=True,exist_ok=a.resume)
    sources=[Path(__file__),Path(cfg['scale']),Path(cfg['model'])/'config.json',
             Path('/home/zch/project/gptq_zch0722/gptq.py'),Path('/home/zch/project/gptq_zch0722/quant.py'),
             Path('/home/zch/project/awq_code/llm-awq_zch0722/awq/quantize/quantizer.py')]
    spec=dict(config=cfg,nsamples=a.nsamples,smoke_layers=a.smoke_layers,
              source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    old=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else {}
    if a.resume:
        if any(old.get(k)!=v for k,v in spec.items()):raise ValueError('Resume source/config mismatch')
        if old.get('status')=='complete':return
    run=dict(spec,status='starting',started_at=old.get('started_at',time.strftime('%F %T')))
    def update(**kw):
        run.update(kw,updated_at=time.strftime('%F %T'));write_json(root/'run.json',run)
    update()
    try:execute(a,cfg,root,update)
    except BaseException as exc:
        update(status='failed',error=repr(exc));traceback.print_exc();raise
if __name__=='__main__':main()
