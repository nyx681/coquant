import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

# ===== Fix rotary embedding device mismatch =====
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
    print('Starting ...')

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
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

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
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    save_dict = {}
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        # if args.nearest:
        if True:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)

                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)

                # 1) fake-quant 权重（float）
                W_fake = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(W.dtype)
                subset[name].weight.data.copy_(W_fake)

                # 2) 存下来：注意 key 要用“全名”，否则不同层同名模块会覆盖
                full_name = f"model.layers.{i}.{name}"
                save_dict[full_name] = {
                    "weight": W_fake.detach().cpu(),                # fake-quant 后的 float 权重
                    "scale":  quantizer.scale.detach().cpu(),       # nearest 算出来的 scale
                    "zero":   quantizer.zero.detach().cpu(),        # nearest 算出来的 zero
                    "maxq":   int(quantizer.maxq),
                }

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps


    torch.save(save_dict, "/home/zch/project/awq_code/models/gptq3bit/gptq_llama2_pile.pt")

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

# def apply_fake_quant(model, quantizers):
#     """
#     将 GPTQ 量化参数应用为 fake-quant（float 权重）
#     """
#     layers = find_layers(model)

#     for name, quantizer in quantizers.items():
#         if name not in layers:
#             continue

#         layer = layers[name]
#         W = layer.weight.data

#         # 量化 → 反量化
#         W_q = quantize(
#             W,
#             quantizer.scale,
#             quantizer.zero,
#             quantizer.maxq
#         )

#         layer.weight.data = W_q.to(W.dtype)

#     return model

# 只保存模型权重
# def apply_fake_quant(model, quantizers):
#     """
#     将 GPTQ 量化参数应用为 fake-quant（float 权重）
#     """
#     layers = find_layers(model)
#     out = {}
#     # n = 0
#     for name, quantizer in quantizers.items():

#         if name not in layers:
#             continue

#         layer = layers[name]
#         W = layer.weight.data


#         device = W.device
#         dtype = W.dtype

#         # ===== 关键修复：把 quantizer 参数搬到 W 所在 device =====
#         scale = quantizer.scale.to(device)
#         zero = quantizer.zero.to(device)
#         maxq = quantizer.maxq  # Python int，不需要 .to()


# def apply_fake_quant(model, quantizers):
#     """
#     将 GPTQ 量化参数应用为 fake-quant（float 权重），
#     并同时保存 weight / scale / zero / maxq
#     """
#     layers = find_layers(model)
#     out = {}

#     for name, quantizer in quantizers.items():
#         if name not in layers:
#             continue

#         layer = layers[name]
#         W = layer.weight.data

#         device = W.device
#         dtype = W.dtype

#         # ===== 确保 quantizer 参数在同一 device =====
#         scale = quantizer.scale.to(device)
#         zero  = quantizer.zero.to(device)
#         maxq  = quantizer.maxq  # int

#         # ===== fake-quant 权重（如果你前面已经算过，这里只是确保 dtype）=====
#         layer.weight.data = W.to(dtype)

#         # ===== 保存完整量化状态 =====
#         out[name] = {
#             "weight": W.cpu(),          # fake-quant 后的 float 权重
#             "scale":  scale.cpu(),      # GPTQ scale
#             "zero":   zero.cpu(),       # GPTQ zero-point
#             "maxq":   maxq,             # 量化整数上限
#         }

#     torch.save(
#         out,
#         "/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_all_s_3.pt"
#     )
#     exit(0)
#     return model

import torch

@torch.no_grad()
def apply_fake_quant(model, quantizers):
    """
    用 GPTQ quantizer 的 (scale/zero/maxq) 对权重做 fake-quant（quantize->dequant，float输出），
    并保存 fake-quant 权重及对应量化参数。
    """
    layers = find_layers(model)
    out = {}

    for name, quantizer in quantizers.items():
        if name not in layers:
            continue

        layer = layers[name]
        W = layer.weight.data
        device = W.device
        dtype = W.dtype

        # ---- 1) 拿到 GPTQ 的量化参数，并放到同 device ----
        scale = quantizer.scale.to(device=device)
        zero  = quantizer.zero.to(device=device)
        maxq  = int(quantizer.maxq)

        # ---- 2) 关键：确保 scale/zero 形状能广播到 W ----
        # 常见情况：
        # - per-channel: scale shape [out_features] 或 [out_features, 1]
        # - per-group:  scale shape [out_features, n_groups] 或 [n_groups] 等（依实现而定）
        # 这里尽量做“最安全的广播”：把 1D 的 per-outchannel 转成 [out, 1]，用于除法/加法
        if scale.dim() == 1 and W.dim() == 2 and scale.numel() == W.size(0):
            scale_ = scale.view(-1, 1)
        else:
            scale_ = scale

        if zero.dim() == 1 and W.dim() == 2 and zero.numel() == W.size(0):
            zero_ = zero.view(-1, 1)
        else:
            zero_ = zero

        # ---- 3) 做 fake-quant：先量化到整数格点，再反量化回 float ----
        # 你项目里 quantize() 通常返回的是“反量化后的 float 权重”
        # （即 fake-quant），所以直接用即可。
        W_fake = quantize(W, scale_, zero_, maxq).to(dtype)

        # ---- 4) 写回模型（可选，但通常你希望模型也变成 fake-quant 状态）----
        layer.weight.data.copy_(W_fake)

        # ---- 5) 保存：保存 fake-quant 权重 + 对应参数（建议存 CPU）----
        out[name] = {
            "weight": W_fake.detach().cpu(),          # fake-quant 后的 float 权重
            "scale":  scale.detach().cpu(),           # 原始 quantizer 的 scale（不强行改形状）
            "zero":   zero.detach().cpu(),            # 原始 quantizer 的 zero
            "maxq":   maxq,
        }

    torch.save(out, "/home/zch/project/awq_code/models/gptq_zch/gptq_llama1_pile42_all_s_1.pt")
    assert 0
    return model


#         layer.weight.data = W.to(dtype)
#         out[name] = W.cpu()
#     torch.save(out, "/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_2.pt")
#     exit(0)
#     return model

# @torch.no_grad()
# def apply_fake_quant(model, quantizers):
#     layers = find_layers(model)
#     out = {}

#     for name, quantizer in quantizers.items():
#         if name not in layers:
#             continue

#         layer = layers[name]
#         W = layer.weight.data
#         device = W.device

#         scale = quantizer.scale.to(device)
#         zero  = quantizer.zero.to(device)
#         maxq  = quantizer.maxq

#         # int code（数值为整数；dtype 仍可为 float）
#         # W_q = torch.clamp(torch.round(W / scale) + zero, 0, maxq)
#         W_q = torch.round(W / scale)

#         # 保存到 CPU，减少文件体积&避免设备依赖
#         out[name] = W_q.cpu()

#         # 如你也需要 scale/zero 一起保存（可选）
#         # out[name + ".scale"] = quantizer.scale.cpu()
#         # out[name + ".zero"]  = quantizer.zero.cpu()

#     torch.save(out, "/home/zch/project/awq_code/models/gptq/gptq_int_codes_no_zero_no_s.pt")
#     assert 0

#     return out_path


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
        help='Save quantized checkpoint under this name.'
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

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )


    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

        # print('保存假量化模型权重...')

        # 将模型移到CPU以节省内存并保存
        # model_cpu = model.cpu()
        # torch.cuda.empty_cache()
        
        # 保存模型和tokenizer
        # print("正在保存模型权重...")
        # model_cpu.save_pretrained("/home/zch/project/awq_code/models/GPTQ/LLAMA2")
        
        

        # 存模型
        # model = apply_fake_quant(model, quantizers)

        # assert 0

        # save_dir = "/home/zch/project/awq_code/models/gptq"
        # import os
        # os.makedirs(save_dir, exist_ok=True)

        # print(f"Saving fake-quantized HF model to {save_dir}")
        # model = model.cpu()
        # torch.cuda.empty_cache()

        # # 保存模型
        # model.save_pretrained(save_dir)

        # # 保存 tokenizer（用 tok 避免名字冲突）
        # from transformers import AutoTokenizer
        # tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        # print("tok type:", type(tok))
        # tok.save_pretrained(save_dir)

        # print("✓ Fake-quant HF model saved successfully")
        # assert 0



    datasets = ['wikitext2']
    if args.new_eval:
        datasets = ['wikitext2']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

