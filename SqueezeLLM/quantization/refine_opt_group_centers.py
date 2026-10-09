"""Generate movable G128 centers from the existing exact 32-to-16 selection."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle,shutil
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
from tqdm import tqdm
from llama import get_model
from chunk_gptq_params import OPT_MODULES


def refine_row(task):
    row,initial=task
    x=row.astype(np.float64).reshape(-1,128)
    centers=initial.copy()
    def score(c):
        distances=(x[:,:,None]-c[:,None,:].astype(np.float64))**2
        labels=distances.argmin(axis=2)
        loss=np.take_along_axis(distances,labels[:,:,None],axis=2).sum(axis=(1,2))
        return labels,loss
    labels,loss=score(centers);before=loss.copy()
    for _ in range(50):
        new=centers.copy()
        for k in range(16):
            mask=labels==k;count=mask.sum(axis=1)
            mean=(x*mask).sum(axis=1)/np.maximum(count,1)
            new[:,k]=np.where(count>0,mean,centers[:,k])
        new_labels,new_loss=score(new)
        improved=new_loss<loss
        if not improved.any():break
        centers[improved]=new[improved];labels[improved]=new_labels[improved];loss[improved]=new_loss[improved]
    if np.any(loss>before):raise AssertionError('Refinement increased MSE')
    return centers,labels.astype(np.uint8),float(before.sum()),float(loss.sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();root=Path(a.output);root.mkdir(exist_ok=False,parents=True)
    if not (Path(a.source)/'complete.json').exists():raise ValueError('Incomplete source')
    if shutil.disk_usage(root).free<11*1024**3:raise OSError('Need 11 GiB free including reserve')
    out=root/'generated';(out/'lut').mkdir(parents=True)
    config=dict(weight_source='uniform_MSE',group_size=128,bit=4,AWQ_scale=False,delta=False,initialization=a.source,max_iter=50,empty_clusters='retain',method='selected_then_Lloyd',status='running')
    (out/'kmeans_config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(1);model=get_model('/mnt/dfq/pretrained_model/opt-6.7b').eval();metrics={}
    with get_context('spawn').Pool(a.workers) as pool:
        for l,layer in enumerate(model.model.decoder.layers):
            with (Path(a.source)/'selected/lut'/f'l{l}.pkl').open('rb') as f:lut=pickle.load(f)
            modules=dict(layer.named_modules())
            for short,suffix in OPT_MODULES.items():
                w=modules[suffix].weight.detach().float().numpy();record=lut[short]
                if record['centers'].shape!=(w.shape[0],w.shape[1]//128,16):raise ValueError('Shape mismatch')
                tasks=((w[r],record['centers'][r]) for r in range(len(w)))
                before=after=0.
                for r,(c,idx,b,e) in enumerate(tqdm(pool.imap(refine_row,tasks),total=len(w),desc=f'{l}.{short}')):
                    record['centers'][r]=c;record['labels'][r]=idx;before+=b;after+=e
                metrics[f'{l}.{short}']=dict(selected_sse=before,generated_sse=after)
                (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
            target=out/'lut'/f'l{l}.pkl'
            with Path(str(target)+'.incomplete').open('wb') as f:pickle.dump(lut,f,protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(str(target)+'.incomplete',target)
    config['status']='complete';(out/'kmeans_config.json').write_text(json.dumps(config,indent=2))
    (root/'complete.json').write_text(json.dumps(dict(status='complete',layers=32)))


if __name__=='__main__':main()
