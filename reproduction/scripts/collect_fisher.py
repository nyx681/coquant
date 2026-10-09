import _bootstrap
import argparse
import hashlib
import torch
from egquant.common import get_model, layers, load_torch, modules, add_experiment_arguments, experiment_from_args, data_path, save_torch, write_json

def main():
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cfg = experiment_from_args(a)
    cache = load_torch(data_path(cfg['paths']['fisher_tokens']))
    if cache['model_id'] != cfg['model_id'] or cache['dataset'] != 'fisher':
        raise ValueError('Fisher calibration token mismatch')
    tokens = cache['tokens']
    if tokens.shape != (cfg['fisher']['samples'], cfg['fisher']['sequence_length']):
        raise ValueError('Fisher calibration shape mismatch')
    token_hash = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if cache['token_sha256'] != token_hash:
        raise ValueError('Calibration checksum mismatch')
    root = data_path(cfg['paths']['fisher'])
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['fisher']['seed'])
    torch.set_num_threads(1)
    model = get_model(cfg).to(a.device)
    model.config.use_cache = False
    model.requires_grad_(False)
    accum, handles = ({}, [])
    for l, layer in enumerate(layers(model, cfg)):
        for name, suffix in modules(cfg).items():
            w = layer.get_submodule(suffix).weight
            w.requires_grad_(True)
            key = (l, name)
            accum[key] = torch.zeros_like(w, dtype=torch.float32)

            def hook(grad, key=key):
                with torch.no_grad():
                    accum[key].add_(grad.float().square())
                return grad
            handles.append(w.register_hook(hook))
    manifest = dict(model_id=cfg['model_id'], status='running', completed_samples=0, samples=len(tokens), sequence_length=tokens.shape[1], token_sha256=token_hash, accumulator_dtype='float32', model_dtype=str(next(model.parameters()).dtype), statistic='mean of per-sequence squared CE gradients; full causal labels')
    write_json(root / 'manifest.json', manifest)
    try:
        for i, block in enumerate(tokens):
            model.zero_grad(set_to_none=True)
            x = block[None].to(a.device)
            loss = model(input_ids=x, labels=x, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite calibration loss')
            loss.backward()
            manifest['completed_samples'] = i + 1
            write_json(root / 'manifest.json', manifest)
            print('Fisher', i + 1, '/', len(tokens), flush=True)
            del loss, x
        for l in range(cfg['layers']):
            chunk = {name: (accum[l, name] / len(tokens)).cpu() for name in modules(cfg)}
            if any((not torch.isfinite(v).all() or (v < 0).any() for v in chunk.values())):
                raise ValueError('Invalid Fisher result')
            save_torch(root / f'layer_{l}.pt', chunk)
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        for h in handles:
            h.remove()
        write_json(root / 'manifest.json', manifest)
if __name__ == '__main__':
    main()
