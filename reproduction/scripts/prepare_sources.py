import _bootstrap
import argparse
import torch
from egquant.baselines import awq_reference, awq_scales, gptq_reference
from egquant.common import checked_chunk, get_model, layers, load_torch, modules, add_experiment_arguments, experiment_from_args, data_path, save_torch, sha, write_json

def main():
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--source', required=True, choices=['original', 'awq-cache', 'gptq-records'])
    p.add_argument('--artifact', metavar='/to/path/source.pt')
    p.add_argument('--device', default='cuda:0', help='AWQ export arithmetic device')
    a = p.parse_args()
    cfg = experiment_from_args(a)
    field = {'original': 'original', 'awq-cache': 'awq', 'gptq-records': 'gptq'}[a.source]
    root = data_path(cfg['paths'][field])
    root.mkdir(parents=True, exist_ok=False)
    meta = dict(model_id=cfg['model_id'], bits=cfg['bits'], group_size=128, source=a.source, status='running')
    if a.source == 'original':
        model = get_model(cfg)
        with torch.no_grad():
            for l, layer in enumerate(layers(model, cfg)):
                chunk = {n: layer.get_submodule(s).weight.detach().cpu().contiguous() for n, s in modules(cfg).items()}
                save_torch(root / f'layer_{l}.pt', chunk)
    else:
        if not a.artifact:
            p.error('--artifact is required for source exports')
        artifact = data_path(a.artifact)
        cache = load_torch(artifact)
        meta['artifact_sha256'] = sha(artifact)
        scales = awq_scales(cache, modules(cfg)) if a.source == 'awq-cache' else None
        prefix = 'model.decoder.layers' if cfg['model_type'] == 'opt' else 'model.layers'
        fallback = []
        with torch.no_grad():
            for l in range(cfg['layers']):
                original = checked_chunk(data_path(cfg['paths']['original']) / f'layer_{l}.pt', cfg)
                chunk = {}
                for name, suffix in modules(cfg).items():
                    w = original[name]
                    if scales is not None:
                        scale = scales.get((l, name))
                        if scale is None:
                            fallback.append(f'{l}.{name}')
                            scale = torch.ones(w.shape[1], dtype=w.dtype)
                        chunk[name] = awq_reference(w.to(a.device), scale.to(a.device), cfg['bits']).cpu()
                    else:
                        key = f'{prefix}.{l}.{suffix}'
                        chunk[name] = gptq_reference(cache[key], w, cfg['bits'])
                save_torch(root / f'layer_{l}.pt', chunk)
        if scales is not None:
            meta['identity_scale_modules'] = fallback
            meta['definition'] = 'quantize(W*a)/a; no extra clipping in this source-export path'
    meta['status'] = 'complete'
    write_json(root / 'manifest.json', meta)
if __name__ == '__main__':
    main()
