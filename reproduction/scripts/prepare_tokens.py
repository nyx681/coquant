import _bootstrap
import argparse
import hashlib
import random
import torch
from egquant.common import load_torch, add_experiment_arguments, experiment_from_args, data_path, save_torch

def main():
    p = argparse.ArgumentParser()
    add_experiment_arguments(p)
    p.add_argument('--dataset', required=True, choices=['wiki', 'c4', 'fisher', 'awq', 'gptq'])
    p.add_argument('--from-cache', metavar='/to/path/tokens.pt')
    a = p.parse_args()
    cfg = experiment_from_args(a)
    provenance = {}
    if a.from_cache:
        old = load_torch(data_path(a.from_cache))
        tokens = old if torch.is_tensor(old) else old['tokens']
        provenance['preparation'] = 'imported token stream; legacy path metadata discarded'
    else:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(data_path(cfg['paths']['model']), local_files_only=True, use_fast=False)
        source = data_path(cfg['paths'][a.dataset + '_data'])
        dataset = load_from_disk(source)
        if not hasattr(dataset, 'column_names') or isinstance(dataset.column_names, dict):
            split = {'wiki': 'test', 'c4': 'validation', 'fisher': 'train', 'awq': 'validation', 'gptq': 'validation'}[a.dataset]
            dataset = dataset[split]
        if a.dataset == 'wiki':
            tokens = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt').input_ids
        elif a.dataset == 'c4':

            def tokenize(examples):
                return tokenizer(examples['text'], truncation=False)
            encoded = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
            ids = [token for row in encoded['input_ids'] for token in row]
            tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        elif a.dataset in ('awq', 'gptq'):

            spec = cfg['baselines'][a.dataset]
            dataset = dataset.shuffle(seed=spec['seed'])
            rng = random.Random(spec['seed'])
            blocks = []
            size = spec['sequence_length']
            for item in dataset:
                text = item['text'].strip()
                if not text:
                    continue
                ids = tokenizer(text, return_tensors='pt').input_ids
                if ids.shape[1] < size:
                    continue
                start = rng.randint(0, ids.shape[1] - size)
                blocks.append(ids[:, start:start + size])
                if len(blocks) == spec['samples']:
                    break
            if len(blocks) != spec['samples']:
                raise ValueError(f'Not enough {a.dataset.upper()} calibration documents')
            tokens = torch.cat(blocks)
            provenance.update(selected_documents=len(blocks), sampling_seed=spec['seed'])
        else:
            rng = random.Random(cfg['fisher']['seed'])
            size = cfg['fisher']['sequence_length']
            blocks = []
            for _ in range(cfg['fisher']['samples']):
                for attempt in range(100000):
                    row = rng.randint(0, len(dataset) - 1)
                    ids = tokenizer(dataset[row]['text'], return_tensors='pt').input_ids
                    if ids.shape[1] >= size:
                        break
                else:
                    raise ValueError('No sufficiently long Fisher calibration document')
                start = 0 if ids.shape[1] - size - 1 < 0 else rng.randint(0, ids.shape[1] - size - 1)
                blocks.append(ids[:, start:start + size])
            tokens = torch.cat(blocks)
        preparations = {'wiki': 'join stored documents with two newlines, then tokenize', 'c4': 'tokenize each stored document, then concatenate in original order', 'awq': 'seeded shuffle; sample one interval from each selected long document', 'gptq': 'seeded shuffle; sample one interval from each selected long document', 'fisher': 'seeded document and interval sampling with replacement'}
        provenance.update(preparation=preparations[a.dataset], tokenizer='use_fast=False; default special-token behavior')
    tokens = tokens.to(dtype=torch.long, device='cpu').contiguous()
    if a.dataset in ('wiki', 'c4'):
        tokens = tokens.reshape(1, -1)
    expected_shape = (cfg['fisher']['samples'], cfg['fisher']['sequence_length'])
    if a.dataset == 'fisher' and tuple(tokens.shape) != expected_shape:
        raise ValueError('Fisher token shape mismatch')
    if a.dataset in ('awq', 'gptq'):
        spec = cfg['baselines'][a.dataset]
        if tuple(tokens.shape) != (spec['samples'], spec['sequence_length']):
            raise ValueError(f'{a.dataset.upper()} calibration token shape mismatch')
    token_hash = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    expected = cfg.get('expected_token_sha256', {}).get(a.dataset)
    if expected and token_hash != expected:
        raise ValueError('Token stream differs from the recorded experiment checksum')
    record = dict(model_id=cfg['model_id'], dataset=a.dataset, tokens=tokens, token_sha256=token_hash, **provenance)
    save_torch(data_path(cfg['paths'][a.dataset + '_tokens']), record)
    print(a.dataset, list(tokens.shape), token_hash)
if __name__ == '__main__':
    main()
