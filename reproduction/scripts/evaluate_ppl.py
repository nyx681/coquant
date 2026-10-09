import _bootstrap
import argparse
import hashlib
import json
import numpy as np
import torch
from egquant.common import checked_chunk, get_model, layers, load_torch, modules, output_dir, add_experiment_arguments, experiment_from_args, data_path, sha, write_json
from egquant.evaluation import perplexity
from egquant.reconstruction import reconstruct_into

def main():
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--method', required=True, choices=['AWQ', 'GPTQ', 'EGConv-A', 'EGConv-G', 'EGComp-AG'])
    p.add_argument('--dataset', required=True, choices=['wiki', 'c4'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    cfg = experiment_from_args(a)
    cache = load_torch(data_path(cfg['paths'][a.dataset + '_tokens']))
    if cache['model_id'] != cfg['model_id'] or cache['dataset'] != a.dataset:
        raise ValueError('Token-cache model/dataset mismatch')
    tokens = cache['tokens']
    token_hash = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if token_hash != cache['token_sha256']:
        raise ValueError('Token cache checksum mismatch')
    expected = cfg.get('expected_token_sha256', {}).get(a.dataset)
    if expected and token_hash != expected:
        raise ValueError('Token stream differs from the recorded experiment checksum')
    root = output_dir(cfg, a.method)
    converted = a.method not in ('AWQ', 'GPTQ')
    if converted:
        run = json.loads((root / 'run.json').read_text())
        metrics = json.loads((root / 'metrics.json').read_text())
        if run['status'] != 'complete' or run['spec']['source_parameters'] != cfg:
            raise ValueError('Incomplete or mismatched quantization result')
    model = get_model(cfg).to(a.device)
    total_weights = parameter_bits = 0
    with torch.no_grad():
        for l, layer in enumerate(layers(model, cfg)):
            original_path = data_path(cfg['paths']['original']) / f'layer_{l}.pt'
            original = checked_chunk(original_path, cfg)
            reference = None if converted else checked_chunk(data_path(cfg['paths'][a.method.lower()]) / f'layer_{l}.pt', cfg)
            for name, suffix in modules(cfg).items():
                key = f'{l}.{name}'
                w = original[name]
                module = layer.get_submodule(suffix)
                if not torch.equal(module.weight.detach().cpu(), w.to(module.weight.dtype)):
                    raise ValueError('Original chunk/model differs: ' + key)
                if converted:
                    cp = root / 'centers' / f'{key}.npz'
                    if sha(cp) != metrics[key]['checkpoint_sha256'] or sha(original_path) != metrics[key]['inputs']['original']:
                        raise ValueError('Artifact checksum mismatch: ' + key)
                    with np.load(cp, allow_pickle=False) as z:
                        info = reconstruct_into(module.weight, w, z['centers'], z['scales'], cfg['bits'])
                    parameter_bits += info['parameter_bits']
                else:
                    if reference[name].shape != w.shape:
                        raise ValueError('Reference/model shape mismatch')
                    module.weight.copy_(reference[name].to(module.weight.dtype))
                    parameter_bits += 32 * (w.numel() // 128)
                total_weights += w.numel()
    ppl = perplexity(model, tokens, torch.device(a.device), block_size=2048, batch_size=a.batch_size)
    record = dict(model_id=cfg['model_id'], method=a.method, bits=cfg['bits'], dataset=a.dataset, ppl=ppl, token_sha256=token_hash, tokens=tokens.numel(), block_size=2048, blocks=tokens.numel() // 2048, batch_size=a.batch_size, base_dtype=str(next(model.parameters()).dtype), attention=model.config._attn_implementation, center_dtype='float16' if converted else None, scale_dtype='float16' if converted else 'source artifact', frozen_fp32_assignments=converted, nominal_bits=cfg['bits'] + (16 * 2 ** cfg['bits'] / 2048 + 16 / 128 if converted else 32 / 128), actual_group_accounting_bits=cfg['bits'] + parameter_bits / total_weights, storage_note='Theoretical packed weight storage, not the dense PPL model or file size.')
    write_json(root / f'{a.dataset}_ppl.json', record)
    print(json.dumps(record, indent=2))
if __name__ == '__main__':
    main()
