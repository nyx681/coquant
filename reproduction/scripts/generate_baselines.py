import _bootstrap
import argparse
import gc
import hashlib
from collections import defaultdict
import torch
from egquant.baselines import awq_reference
from egquant.common import checked_chunk, get_model, layers, load_torch, modules, add_experiment_arguments, experiment_from_args, data_path, save_torch, write_json

def tensor_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple((tensor_tree(v, device) for v in value))
    if isinstance(value, list):
        return [tensor_tree(v, device) for v in value]
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    return value

def hidden(output):
    return output[0] if isinstance(output, tuple) else output

@torch.no_grad()
def generate(cfg, method):
    if not torch.cuda.is_available():
        raise RuntimeError('Baseline generation requires CUDA; use CUDA_VISIBLE_DEVICES to select a GPU')
    cache = load_torch(data_path(cfg['paths'][method + '_tokens']))
    if cache['model_id'] != cfg['model_id'] or cache['dataset'] != method:
        raise ValueError('Wrong baseline calibration cache')
    tokens = cache['tokens']
    if tokens.ndim != 2 or not len(tokens) or tokens.shape[1] != cfg['baselines'][method]['sequence_length']:
        raise ValueError('Baseline calibration shape mismatch')
    if cache['token_sha256'] != hashlib.sha256(tokens.numpy().tobytes()).hexdigest():
        raise ValueError('Baseline calibration checksum mismatch')
    if len(tokens) != cfg['baselines'][method]['samples']:
        raise ValueError(f'{method.upper()} calibration sample count mismatch')
    root = data_path(cfg['paths'][method])
    root.mkdir(parents=True, exist_ok=False)
    spec = dict(model_id=cfg['model_id'], method=method, bits=cfg['bits'], group_size=128, calibration=cfg['baselines'][method], token_sha256=cache['token_sha256'], actual_calibration_blocks=len(tokens), status='running', completed_layers=0, archival_match='Not asserted; compare source artifact and token hashes before exact reproduction.')
    write_json(root / 'manifest.json', spec)
    model = get_model(cfg).cuda()
    model.config.use_cache = False
    stack = layers(model, cfg)
    captured = []
    kwargs = {}

    class Captured(Exception):
        pass

    def capture(module, args, kw):
        captured.append(args[0].detach().cpu())
        kwargs.update(tensor_tree({k: v for k, v in kw.items() if k != 'use_cache'}, 'cpu'))
        raise Captured()
    handle = stack[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for block in tokens:
            try:
                model(block[None].cuda(), use_cache=False)
            except Captured:
                pass
    finally:
        handle.remove()
    if len(captured) != len(tokens):
        raise RuntimeError('Failed to capture calibration hidden states')
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    inps = torch.cat(captured).cuda()
    kw = tensor_tree(kwargs, 'cuda')
    all_scales = []
    try:
        for l, layer in enumerate(stack):
            layer.cuda()
            original = checked_chunk(data_path(cfg['paths']['original']) / f'layer_{l}.pt', cfg)
            if any((not torch.equal(layer.get_submodule(s).weight.cpu(), original[n].to(layer.get_submodule(s).weight.dtype)) for n, s in modules(cfg).items())):
                raise ValueError('Original/model mismatch at baseline layer')
            if method == 'awq':
                from egquant.vendor.awq_scale import auto_scale_block722, apply_scale722
                features = defaultdict(list)
                handles = []
                for name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Linear):

                        def record(m, args, out, name=name):
                            features[name].append(args[0].detach().cpu())
                        handles.append(module.register_forward_hook(record))
                outs = hidden(layer(inps, **kw)).detach()
                for h in handles:
                    h.remove()
                features = {k: torch.cat(v, dim=0) for k, v in features.items()}
                scale_list = auto_scale_block722(layer, dict(kw), cfg['bits'], {'zero_point': True, 'q_group_size': 128}, features)
                name_scales = {name: scale for _, names, scale, _ in scale_list for name in names}
                quantized = {}
                for n, suffix in modules(cfg).items():
                    w = original[n].cuda()
                    scale = name_scales.get(suffix)
                    if scale is None:
                        scale = torch.ones(w.shape[1], dtype=w.dtype, device=w.device)
                    quantized[n] = awq_reference(w, scale, cfg['bits']).cpu()
                save_torch(root / f'layer_{l}.pt', quantized)
                prefix = 'model.decoder.layers' if cfg['model_type'] == 'opt' else 'model.layers'
                for prev, names, scale, th in scale_list:
                    all_scales.append((f'{prefix}.{l}.{prev}', tuple((f'{prefix}.{l}.{n}' for n in names)), scale, th))
                apply_scale722(layer, scale_list, input_feat_dict=features)
                del features, quantized
            else:
                from egquant.vendor.gptq import GPTQ
                from egquant.vendor.quant import Quantizer
                plan = [list(modules(cfg))] if cfg['model_type'] == 'opt' else [['k', 'v', 'q'], ['o'], ['up', 'gate'], ['down']]
                records = {}
                for names in plan:
                    qs, handles = ({}, [])
                    for n in names:
                        module = layer.get_submodule(modules(cfg)[n])
                        q = GPTQ(module)
                        q.quantizer = Quantizer()
                        q.quantizer.configure(cfg['bits'], perchannel=True, sym=False, mse=False)
                        qs[n] = q

                        def record(m, args, out, n=n):
                            qs[n].add_batch(args[0].data, out.data)
                        handles.append(module.register_forward_hook(record))
                    for block in inps:
                        layer(block[None], **kw)
                    for h in handles:
                        h.remove()
                    for n, q in qs.items():
                        s = cfg['baselines']['gptq']
                        q.fasterquant(percdamp=s['damping'], groupsize=128, actorder=s['act_order'], static_groups=s['static_groups'])
                        records[n] = dict(scale=q.quantizer.scale.cpu(), zero=q.quantizer.zero.cpu(), maxq=int(q.quantizer.maxq), group_size=128, static_groups=s['static_groups'], g_idx=q.quantizer.gptq_g_idx.cpu())
                        q.free()
                outs = torch.cat([hidden(layer(block[None], **kw)) for block in inps])
                save_torch(root / f'layer_{l}.pt', {n: layer.get_submodule(s).weight.detach().cpu().contiguous() for n, s in modules(cfg).items()})
                save_torch(root / f'params_{l}.pt', records)
            layer.cpu()
            inps = outs
            spec['completed_layers'] = l + 1
            write_json(root / 'manifest.json', spec)
            print(method, l + 1, '/', len(stack), flush=True)
            torch.cuda.empty_cache()
        if method == 'awq':
            save_torch(root / 'scale_cache.pt', dict(scale=all_scales, clip=[]))
        spec['status'] = 'complete'
    except BaseException:
        spec['status'] = 'failed'
        raise
    finally:
        write_json(root / 'manifest.json', spec)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--method', choices=['awq', 'gptq'], required=True)
    args = p.parse_args()
    generate(experiment_from_args(args), args.method)
