import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
C4_TOKEN_SHA256 = {'llama1_7b': '8d1720c52a5a7bd18ffa86a06ce2a8f5cdd6ddd6b294e3ef59a88523095d703c', 'llama2_7b': '0bbabc8182738b63a79131b27e033d34acbfdaad9f8c5b8633761c7f8434609d', 'mistral7b': 'ed1de19f13d233e9b95e6e00f948997d456d1cb3bf0fbf491c26ce5242976a01', 'opt67b': '53ccf3b59d1195aae04270e998a2097fbca16187fde42be20e2353afbc38afdd', 'qwen3_8b_base': 'acaefffd06b565906d3de88e58182166e3bd56d556875c9ce32646751e7e0785'}
LLAMA_MODULES = dict(q='self_attn.q_proj', k='self_attn.k_proj', v='self_attn.v_proj', o='self_attn.o_proj', gate='mlp.gate_proj', up='mlp.up_proj', down='mlp.down_proj')
OPT_MODULES = dict(q='self_attn.q_proj', k='self_attn.k_proj', v='self_attn.v_proj', o='self_attn.out_proj', up='fc1', down='fc2')

def data_path(value):
    p = Path(value)
    if p == Path('/to/path') or Path('/to/path') in p.parents:
        raise ValueError('Replace /to/path/... with a real input or output path')
    if '..' in p.parts:
        raise ValueError('Parent traversal is not supported')
    return p if p.is_absolute() else ROOT / p

def add_experiment_arguments(parser):
    parser.add_argument('--model', required=True, choices=['llama1_7b', 'llama2_7b', 'opt67b', 'qwen3_8b_base', 'mistral7b', 'llama1_13b', 'llama2_13b', 'opt13b'])
    parser.add_argument('--gptq-seed', type=int, default=42)
    parser.add_argument('--bits', type=int, choices=[3, 4], required=True)
    parser.add_argument('--model-dir', default='/to/path/model')
    parser.add_argument('--data-dir', default='/to/path/data')
    parser.add_argument('--work-dir', default='/to/path/work')

def experiment_from_args(args):
    model = args.model
    if model not in ('llama1_7b', 'llama2_7b', 'opt67b', 'qwen3_8b_base', 'mistral7b', 'llama1_13b', 'llama2_13b', 'opt13b'):
        raise ValueError('Unsupported model')
    if args.bits not in (3, 4):
        raise ValueError('Unsupported bit width')
    work = Path(args.work_dir) / model
    data = Path(args.data_dir)
    artifact = work / 'artifacts'
    width = 'w' + str(args.bits)
    paths = {'model': args.model_dir, 'original': str(artifact / 'original'), 'fisher': str(artifact / 'fisher'), 'awq': str(artifact / width / 'awq'), 'gptq': str(artifact / width / 'gptq'), 'output': str(work / 'outputs' / width)}
    for name, folder in (('wiki', 'wikitext2_test'), ('c4', 'c4_validation_shard'), ('fisher', 'c4_train'), ('awq', 'pile_validation'), ('gptq', 'pile_validation')):
        paths[name + '_data'] = str(data / folder)
        paths[name + '_tokens'] = str(work / 'tokens' / (name + '.pt'))
    return {'model_id': model, 'model_type': 'opt' if model in ('opt67b', 'opt13b') else 'llama', 'layers': 40 if model.endswith('13b') else 36 if model == 'qwen3_8b_base' else 32, 'bits': args.bits, 'eta': 0.5, 'group_size': 2048, 'scale_group_size': 128, 'attention': 'auto', 'expected_token_sha256': {'c4': C4_TOKEN_SHA256[{'llama1_13b':'llama1_7b','llama2_13b':'llama2_7b','opt13b':'opt67b'}.get(model,model)]}, 'fisher': {'samples': 100, 'sequence_length': 512, 'seed': 0}, 'paths': paths, 'baselines': {'awq': {'seed': 42, 'samples': 128, 'sequence_length': 2048, 'clip': False}, 'gptq': {'seed': getattr(args, 'gptq_seed', 42), 'samples': 128, 'sequence_length': 2048, 'damping': 0.01, 'act_order': True, 'static_groups': True}}}

def modules(cfg):
    return OPT_MODULES if cfg['model_type'] == 'opt' else LLAMA_MODULES

def layers(model, cfg):
    result = model.model.decoder.layers if cfg['model_type'] == 'opt' else model.model.layers
    if len(result) != cfg['layers']:
        raise ValueError('Configuration/model layer count differs')
    return result

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def checked_chunk(path, cfg):
    chunk = load_torch(path)
    if set(chunk) != set(modules(cfg)):
        raise ValueError('Layer chunk does not contain the expected modules')
    for name, w in chunk.items():
        if w.ndim != 2 or w.shape[1] % 128 or (not torch.isfinite(w).all()):
            raise ValueError('Invalid weight matrix: ' + name)
    return chunk

def get_model(cfg):
    from transformers import AutoModelForCausalLM
    options = {}
    if cfg.get('attention', 'auto') != 'auto':
        options['attn_implementation'] = cfg['attention']
    return AutoModelForCausalLM.from_pretrained(data_path(cfg['paths']['model']), torch_dtype='auto', local_files_only=True, **options).eval()

def output_dir(cfg, method):
    return data_path(cfg['paths']['output']) / method
