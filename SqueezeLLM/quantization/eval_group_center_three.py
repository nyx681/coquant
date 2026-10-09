"""Evaluate exactly three G128 methods with identical WikiText-2 test protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from llama import get_model, awq_compatible_ppl_eval
from squeezellm.datautils import get_awq_ppl_tokens
from squeezellm.fake_quant import apply_squeezellm_fake_quant


def evaluate(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    model = get_model(args.model).eval()
    with torch.no_grad():
        if args.variant == 'gptq_native':
            records = torch.load(args.gptq_params, map_location='cpu', mmap=True)
            modules = dict(model.named_modules())
            if len(records) != 224:
                raise ValueError('Expected 224 LLaMA-7B Linear records')
            for name, record in records.items():
                target = modules[name].weight
                if record['scale'].shape != (target.shape[0], target.shape[1] // 128):
                    raise ValueError(f'Invalid G128 scale shape: {name}')
                if record.get('group_size', 128) != 128 or record.get('maxq') != 15 or target.shape != record['weight'].shape:
                    raise ValueError(f'Invalid W4-G128 record: {name}')
                target.copy_(record['weight'].to(dtype=target.dtype))
            del records
        else:
            state = json.loads((Path(args.root)/'run.json').read_text())
            if state['status'] != 'complete' or state.get('partial', True):
                raise ValueError('Full LUT generation is not complete')
            apply_squeezellm_fake_quant(model, str(Path(args.root)/args.variant),
                                       include_sparse=False, chunk_rows=512)
        tokens, dataset_path = get_awq_ppl_tokens('wikitext', args.model)
        ppl = awq_compatible_ppl_eval(model, tokens, torch.device('cuda:0'),
                                     'wikitext', block_size=2048, batch_size=8)
    result = dict(variant=args.variant, ppl=ppl, dataset='wikitext2_test',
                  dataset_path=dataset_path, model=args.model, group_size=128, bit=4,
                  block_size=2048, batch_size=8, tokens=tokens.numel(),
                  artifact=args.gptq_params if args.variant=='gptq_native' else str(Path(args.root)/args.variant))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.incomplete')
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, destination)
    print(json.dumps(result), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--model', default='/mnt/dfq/pretrained_model/Llama-7b')
    p.add_argument('--gptq_params', default='/mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt')
    p.add_argument('--variant', choices=['gptq_native','awq','selected','refined'])
    p.add_argument('--output')
    p.add_argument('--generator_pid', type=int)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.variant:
        if not args.output: p.error('--output required with --variant')
        evaluate(args)
        return
    output=Path(args.root)/'ppl'
    output.mkdir(exist_ok=True)
    for variant in ('gptq_native','awq','selected'):
        target=output/f'{variant}_wikitext.json'
        if target.exists():
            raise FileExistsError(f'Refusing to silently reuse previous test result: {target}')
        if variant != 'gptq_native':
            while True:
                state=json.loads((Path(args.root)/'run.json').read_text())
                if state.get('status')=='complete': break
                if args.generator_pid:
                    os.kill(args.generator_pid,0)  # Fail if generator has exited before completion.
                print('Waiting for full LUT generation...', flush=True)
                time.sleep(30)
        subprocess.run([sys.executable,__file__,'--root',args.root,'--model',args.model,
                        '--gptq_params',args.gptq_params,'--variant',variant,'--output',str(target)],check=True)
    results=[json.loads((output/f'{v}_wikitext.json').read_text()) for v in ('awq','gptq_native','selected')]
    (output/'three_ppl.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
