"""把 ``nuq.py`` 生成的 LUT 中间结果打包为可直接推理的 checkpoint。

``nuq.py`` 输出的 labels 暂时是 NumPy int8 数组，并不代表权重已经按 W3/W4
紧凑存储。本脚本会重新加载原始 Hugging Face 模型，将目标 ``nn.Linear``
替换为 ``QuantLinearLUT``，再调用 ``pack2`` 完成两件事：

* 把每个权重替换为其码本编号，并将编号真正压入 3/4-bit 位流 ``qweight``；
* 把每个输出通道的非均匀聚类中心写入 FP32 ``lookup_table``。

dense-only 模式不要传 ``--include_sparse``。此时 checkpoint 中没有
rows/cols/vals 稀疏残差，只包含紧凑编号、码本以及未量化模块的原有参数。
"""

import json
import os
import pickle
import time

import torch
import torch.nn as nn
import transformers
from squeezellm.modelutils import *
from squeezellm.quant import *


@torch.no_grad()
def llama_sequential(model, folder, include_sparse):
    """读取所有层的 LUT，并建立“完整模块名 -> 量化数据”映射。

    返回的 ``quantizers`` 结构为::

        {
            "model.layers.0.self_attn.q_proj": [lut, outliers_or_none],
            ...
        }

    ``lut`` 的结构来自 nuq.py：
    ``lut[output_channel][group] = (centers, labels)``。当前实现每个输出通道
    只有一个 group。dense-only 模式下 ``outliers_or_none`` 恒为 None。

    函数被 ``torch.no_grad`` 包裹，是因为这里只读取离线量化结果和组织数据，
    不应为后续的权重复制构建 autograd 图。
    """
    print("Starting ...")

    # pack.py 当前针对 LLaMA 风格模型，Transformer block 位于 model.layers。
    layers = model.model.layers

    quantizers = {}
    for i in range(len(layers)):
        # 每个文件对应一个 Transformer 层，内部键使用短名称：
        # q/k/v/o/gate/up/down。
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            # lut_layer[module][channel][group] = (centroids, indices)
            # channel 数等于该 Linear 的 out_features；当前 group 数恒为 1。
            lut_layer = pickle.load(f)

        if include_sparse:
            # 只有 Dense-and-Sparse checkpoint 才读取该文件。其内容按下面的
            # outlier_index 顺序保存，而不是使用完整模块名作为字典键。
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                outlier_list_layer = pickle.load(f)

        # nuq.py 分层文件使用短名称，而模型遍历/替换需要 state_dict 风格的完整
        # 模块路径，因此在这里定义一一对应关系。
        sequential_lut = ["q", "k", "v", "o", "gate", "up", "down"]
        sequential_lut_real_name = {
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
            "down": "mlp.down_proj",
        }

        # outliers/l{i}.pkl 是按模块顺序保存的 list，用这个映射找到对应元素。
        outlier_index = {"q": 0, "k": 1, "v": 2, "o": 3, "gate": 4, "up": 5, "down": 6}

        for s in sequential_lut:
            lut = lut_layer[s]
            if include_sparse:
                idx = outlier_index[s]
                outliers = outlier_list_layer[idx]
            else:
                # dense-only 的关键：不加载也不向 QuantLinearLUT 传入稀疏残差。
                outliers = None
            name = sequential_lut_real_name[s]
            # find_layers/make_quant_lut 使用完整路径定位并替换模型中的 nn.Linear。
            quantizers["model.layers.%d.%s" % (i, name)] = [lut, outliers]

    return quantizers
# {
#     "model.layers.0.self_attn.q_proj": [
#         lut_for_layer0_q_proj,
#         None,
#     ],
#     "model.layers.0.self_attn.k_proj": [
#         lut_for_layer0_k_proj,
#         None,
#     ],
#     "model.layers.0.mlp.up_proj": [
#         lut_for_layer0_up_proj,
#         None,
#     ],
#     "model.layers.1.self_attn.q_proj": [
#         lut_for_layer1_q_proj,
#         None,
#     ],
# }

# 替换模型并执行打包
def llama_pack(
    model,
    quantizers,
    wbits,
    include_sparse,
    balanced,
    num_nonzero_per_thread,
):
    """将目标 Linear 替换为 QuantLinearLUT，并写入码本与 bit-packed 编号。

    ``wbits`` 必须和生成 LUT 时 ``nuq.py --bit`` 一致。函数本身不会重新运行
    K-means；它只消费 centers/labels，把 labels 压缩并复制 centers。

    返回 ``(model, sparsedict)``。dense-only 时 sparsedict 为空；稀疏模式下
    它记录每个量化层的非零残差数量，供加载 checkpoint 时创建正确大小的 buffer。
    """
    # 先保存原始 nn.Linear 的引用。make_quant_lut 会原地替换模型模块，之后仍需
    # 从这些原始层读取 bias 等信息传给新的量化层。
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}

    # 根据 quantizers 中列出的完整模块名，将对应 nn.Linear 替换成空的
    # QuantLinearLUT 容器。容器会按 wbits 创建 qweight 和 lookup_table buffer。
    # 将 nn.Linear 替换成 QuantLinearLUT
    make_quant_lut(
        model,
        quantizers,
        wbits,
        include_sparse=include_sparse,
        balanced=balanced,
    )

    # qweight
    # torch.zeros(
    #     (
    #         infeatures // 32 * bits,
    #         outfeatures,
    #     ),
    #     dtype=torch.int32,
    # )
    # infeatures = 4096
    # outfeatures = 4096
    # bits = 4

    # qweight.shape
    # # [4096 // 32 * 4, 4096]
    # # [512, 4096]

    # qweight.dtype
    # # torch.int32

    # lookup_table
    # lookup_table.shape
    # # [4096, 16]

    # 原始层虽然从模型结构中被替换了，但替换前已经被保存到 layers 字典中，所以仍然存在。

    # 替换完成后重新遍历模型，取得所有待填充的 QuantLinearLUT 实例。
    qlayers = find_layers(model, [QuantLinearLUT])
    print("Packing ...")
    sparsedict = {}

    for name in qlayers:
        print(name)
        lookup_table = quantizers[name]
        # pack2 使用 NumPy 在 CPU 上执行位运算；确保原始 Linear 已位于 CPU。
        layers[name].cpu()

        # pack2 会把每行 centers 写入 lookup_table，把 labels 转置后压入 qweight。
        # W4 每个 uint32 保存 8 个编号；W3 每 3 个 uint32 保存 32 个编号。
        # 这是整个真实打包过程的核心。
        qlayers[name].pack2(
            layers[name], # 替换前保存下来的原始浮点层
            lookup_table, # [lut, outliers]
            include_sparse,
            num_nonzero_per_thread=num_nonzero_per_thread,
        )
        if include_sparse:
            # 稀疏 CSR 的 vals 长度必须写入 checkpoint 元数据，否则加载阶段无法
            # 在 state_dict load 之前创建尺寸正确的 rows/cols/vals buffer。
            sparsedict[name] = qlayers[name].vals.shape[-1]

    print("Done.")
    return model, sparsedict


if __name__ == "__main__":
    import argparse

    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    # --model 指向原始 Hugging Face checkpoint；不能传 chunk_models.py 的分层目录。
    parser.add_argument("--model", type=str, help="llama model to load")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Save quantized checkpoint under this name.",
    )

    # --folder 指向 nuq.py 的 output_folder，其中至少应存在 lut 子目录。
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Path to folder containing luts and outliers.",
    )
    parser.add_argument(
        "--include_sparse",
        action="store_true",
        help="Whether loaded checkpoint has sparse matrix.",
    )

    # 以下两个参数只影响 Dense-and-Sparse 的稀疏 CUDA kernel。dense-only 模式
    # 不传 --include_sparse 时，即使保留默认值也不会产生任何作用。
    parser.add_argument(
        '--balanced', action='store_true',
        help='Whether to use balanced sparse kernel.'
    )
    parser.add_argument(
        '--num_nonzero_per_thread', type=int, default=10,
        help='Num nonzeros assigned to each thread.'
    )


    args = parser.parse_args()

    # 在耗时的打包前创建目标目录，避免全部层打包完才因父目录不存在而保存失败。
    save_directory = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_directory, exist_ok=True)

    # 必须加载原始浮点模型：LUT 中只含被量化 Linear 的 centers/labels，模型结构、
    # 未量化参数以及可能存在的 bias 仍需从原始 checkpoint 获得。
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto"
    )
    model.eval()

    # 第一步：读取全部 l*.pkl，并将短模块名转换为模型中的完整路径。
    print("Running llama_sequential")
    tick = time.time()
    quantizers = llama_sequential(
        model=model,
        folder=args.folder,
        include_sparse=args.include_sparse,
    )
    print("llama_sequential Done:", time.time() - tick)

    # 第二步：替换 Linear，复制 FP32 码本，并真正执行 3/4-bit 编号打包。
    print("Running llama_pack")
    tick = time.time()
    model, numvals = llama_pack(
        model=model,
        quantizers=quantizers,
        wbits=args.wbits,
        include_sparse=args.include_sparse,
        balanced=args.balanced,
        num_nonzero_per_thread=args.num_nonzero_per_thread,
    )
    print("llama_pack Done:", time.time() - tick)

    # 此时 state_dict 中量化层已经不再包含原始 dense weight，而是包含 qweight、
    # lookup_table 和可选 bias/稀疏 buffer；其他未量化模块保持原始数据类型。
    model_dict = model.state_dict()

    if args.include_sparse:
        # 将每层稀疏非零数作为额外键合并。llama.py 加载 checkpoint 时会先读取
        # 这些值，据此构造 QuantLinearLUT 的稀疏 buffer，再加载真正的 CSR 数据。
        for k, v in numvals.items():
            model_dict["sparse_threshold." + k] = v

    # 保存的是部署用的真实量化 checkpoint，而非 nuq.py 的中间 LUT 文件。
    torch.save(model_dict, args.save)

    # 同目录记录全局 bit 宽。当前实现是统一精度，所有目标 Linear 共用 wbits。
    data = {"wbits": args.wbits}
    output_fn = os.path.join(save_directory, "quant_config.json")

    # quant_config.json 可供其他加载器快速确认 checkpoint 的权重量化位宽。
    with open(output_fn, "w") as f:
        json.dump(data, f, indent=4)
