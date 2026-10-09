"""Merge validated per-layer GPTQ artifacts into a dense reference .pt file."""
import _bootstrap
import argparse
import json
import torch
from egquant.common import add_experiment_arguments, experiment_from_args, data_path, load_torch, modules, save_torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(p)
    p.add_argument('--output', required=True)
    a=p.parse_args(); c=experiment_from_args(a)
    root=data_path(c['paths']['gptq'])
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Incomplete GPTQ generation')
    records={};prefix='model.decoder.layers' if c['model_type']=='opt' else 'model.layers'
    for i in range(c['layers']):
        weights=load_torch(root/f'layer_{i}.pt'); params=load_torch(root/f'params_{i}.pt')
        for n,suffix in modules(c).items():
            rec=dict(params[n],weight=weights[n]);w=rec['weight']
            if rec['maxq']!=2**c['bits']-1 or rec['scale'].shape!=(w.shape[0],w.shape[1]//128):
                raise ValueError('Wrong bit width or incomplete G128 scales')
            records[f'{prefix}.{i}.{suffix}']=rec
    target=data_path(a.output)
    if target.exists():raise FileExistsError(target)
    save_torch(target,records)
    print(f'Saved {len(records)} records to {target}')
if __name__=='__main__':main()
