import time
import os
import torch
import torch.nn as nn

# 假设这些模块都在当前目录下
from gptq import *
from modelutils import *
from quant import *

# ===== Fix rotary embedding device mismatch (Llama 专用修复) =====
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

_old_forward = LlamaRotaryEmbedding.forward

def _patched_forward(self, hidden_states, position_ids):
    # 强制把 rotary buffer 移到 hidden_states 的 device
    self.inv_freq = self.inv_freq.to(hidden_states.device)
    return _old_forward(self, hidden_states, position_ids)

LlamaRotaryEmbedding.forward = _patched_forward


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    model.config.use_cache = False
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting GPTQ ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = {k: v for k, v in kwargs.items() if k != 'use_cache'}
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs'] or {}

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                # 这里执行 GPTQ 核心算法：计算 Hessian、更新权重、计算最优 Scale/Zero
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                # With --static-groups, fasterquant has concatenated every
                # original contiguous group's scale/zero into [out, K/group].
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = {k: v for k, v in kwargs.items() if k != 'use_cache'}
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs'] or {}

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        # ===== 关键修复：只有在显式指定 nearest 时才运行 RTN =====
        # 之前这里是 if True，导致 GPTQ 权重被覆盖
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)
        # ======================================================

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

# ===== 保存 GPTQ fake-quant 结果：weight + scale + zero + maxq =====
def save_gptq_params(model, quantizers, save_path):
    print(f"Saving GPTQ fake-quant params to {save_path} ...")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_layers = find_layers(model)
    save_dict = {}

    for name, quantizer in quantizers.items():
        if name not in all_layers:
            continue

        layer = all_layers[name]
        group_size = int(getattr(quantizer, "gptq_group_size", -1))
        static_groups = bool(getattr(quantizer, "gptq_static_groups", False))
        scale = quantizer.scale.detach().cpu()
        zero = quantizer.zero.detach().cpu()

        if group_size > 0:
            expected_groups = (layer.weight.shape[1] + group_size - 1) // group_size
            expected_shape = (layer.weight.shape[0], expected_groups)
            if not static_groups:
                raise ValueError(
                    "Grouped GPTQ real-quant export requires --static-groups; "
                    f"layer {name} was quantized with dynamic groups"
                )
            if tuple(scale.shape) != expected_shape or tuple(zero.shape) != expected_shape:
                raise ValueError(
                    f"Layer {name} has incomplete grouped qparams: "
                    f"scale={tuple(scale.shape)}, zero={tuple(zero.shape)}, "
                    f"expected={expected_shape}"
                )

        save_dict[name] = {
            "weight": layer.weight.data.detach().cpu(),
            "scale": scale,
            "zero": zero,
            "maxq": int(quantizer.maxq),
            "group_size": group_size,
            "static_groups": static_groups,
        }

    torch.save(save_dict, save_path)
    print(f"Saved {len(save_dict)} GPTQ layers to {save_path}")

from transformers import AutoTokenizer

def save_fake_quant_hf(model, base_model_dir: str, save_dir: str, quantizers=None, enc=None):
    os.makedirs(save_dir, exist_ok=True)

    # 1) tokenizer：优先用你外面传进来的 enc（你脚本里 enc 通常就是 tokenizer）
    if enc is not None:
        tok = enc
    else:
        tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)

    # 保险：检查 tok 类型
    if not hasattr(tok, "save_pretrained"):
        raise TypeError(f"Tokenizer object invalid: type={type(tok)} value={tok}")

    # 2) CPU 保存
    model_cpu = model.cpu()
    torch.cuda.empty_cache()

    # 3) 修复 generation_config 的 pad_token_id（否则你之前那类报错还会出现）
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = getattr(tok, "eos_token_id", 0)
        if pad_id is None or pad_id < 0:
            pad_id = 0
    model_cpu.generation_config.pad_token_id = int(pad_id)
    model_cpu.config.pad_token_id = int(pad_id)

    # 4) 保存模型
    print(f"[SAVE] saving fake-quant HF model to: {save_dir}")
    try:
        model_cpu.save_pretrained(save_dir, safe_serialization=True)
    except TypeError:
        model_cpu.save_pretrained(save_dir)

    # 5) 保存 tokenizer
    tok.save_pretrained(save_dir)

    # 6) 保存 quantizers（可选）
    if quantizers is not None:
        torch.save(quantizers, os.path.join(save_dir, "gptq_quantizers.pt"))

    print("[SAVE] done.")
    return save_dir


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=42, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save packed quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save-gptq', type=str, default='',
        help='Save GPTQ fake-quant weights and params (weight/scale/zero/maxq) to this path.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    # 设置设备
    DEV = torch.device('cuda:0')

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # 1. 运行 GPTQ 量化
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(f"GPTQ Time: {time.time() - tick:.2f}s")
        
        if args.save_gptq:
            save_gptq_params(model, quantizers, args.save_gptq)

        # ✅ 新增：保存 fake-quant 的 HuggingFace 模型目录
        # hf_save_dir = "/home/zch/project/awq_code/models/GPTQ/LLAMA1"
        # save_fake_quant_hf(model, args.model, hf_save_dir, quantizers=quantizers)
    
    # 3. 评估模型 (此时模型已经是 GPTQ 后的状态)
    datasets = ['wikitext2']
    if args.new_eval:
        datasets = ['wikitext2']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    # 4. (可选) 打包保存为 int 权重
    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
