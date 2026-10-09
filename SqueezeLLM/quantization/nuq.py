"""为 SqueezeLLM 生成逐输出通道（per-channel）的非均匀量化码本。

这个脚本只负责离线量化分析，输出的 ``lut/l{layer}.pkl`` 仍是便于后续
处理的中间格式；真正把码本编号压成连续 3/4 bit 的步骤在 ``pack.py`` 和
``squeezellm/quant.py::QuantLinearLUT.pack2`` 中完成。

Dense-only（不使用 Dense-and-Sparse）时的核心过程如下：

1. 从分层 checkpoint 中读取一个 Transformer 层的原始权重；
2. 使用 Fisher/梯度平方，或 AWQ 反量化权重与原始权重的逐元素
   绝对误差 ``abs(W - W_awq)`` 作为 K-means 的 sample weight；
3. 对该层每个目标 Linear 的每一个输出通道（即权重矩阵的一行）分别运行
   一维加权 K-means；
4. 保存每行的 K 个聚类中心（码本）以及每个原始权重的聚类编号（code）。

若指定 ``--outlier_config``、正的 ``--sensitivity`` 或
``--importance_mask``，脚本会先把相应权重移到稀疏分量，再对剩余的稠密
分量执行完全相同的非均匀量化。未指定这些参数时不会生成稀疏分量。
"""

import os

# 外层已经使用 multiprocessing.Pool 并行处理不同的权重行。限制每个子进程
# 内部的 OpenMP 线程数可以避免 sklearn/BLAS 再次并行造成 CPU 线程过量订阅。
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import re

import numpy as np
import torch
from sklearn.cluster import KMeans
from squeezellm.model_parse import (
    get_layers_name,
    get_module_names,
    get_sequential,
    parse_model,
)
from squeezellm.outliers import (
    remove_outliers,
    remove_outliers_by_mask,
    remove_outliers_by_threshold,
)
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()

# ``--model``、``--gradient`` 和 ``--awq_model`` 都是由 chunk_models.py
# 按层拆分后的目录，内部文件名形如 layer_0.pt、layer_1.pt。
parser.add_argument("--model", type=str, help="model weights to load", required=True)
parser.add_argument(
    "--model_type", type=str, default=None, help="model type", choices=["llama", "opt"]
)
parser.add_argument(
    "--gradient",
    type=str,
    default=None,
    help=(
        "chunked Fisher/gradient-square weights; required unless --awq_model "
        "selects AWQ-delta K-means weighting"
    ),
)
parser.add_argument(
    "--awq_model",
    type=str,
    default=None,
    help=(
        "chunked AWQ dequantized weights; when set, K-means sample weights "
        "are abs(original_weight - awq_dequantized_weight) instead of Fisher"
    ),
)
parser.add_argument(
    "--secondary_delta_model",
    type=str,
    default=None,
    help=(
        "optional second dequantized-reference chunk directory; its elementwise "
        "delta is compared with the primary --awq_model delta"
    ),
)
parser.add_argument(
    "--secondary_delta_reference_source",
    type=str,
    default="gptq",
    choices=["awq", "gptq"],
    help="identity of the reference supplied through --secondary_delta_model",
)
parser.add_argument(
    "--delta_reference_selection",
    type=str,
    default="minimum",
    choices=[
        "minimum",
        "normalized_confidence",
        "channel_normalized_confidence",
        "soft_modulation",
    ],
    help=(
        "rule for combining two delta references: minimum takes the raw "
        "elementwise minimum; normalized_confidence compares separately "
        "median-normalized element deltas; channel_normalized_confidence "
        "makes one coherent decision per input channel from normalized RMS "
        "relative error; soft_modulation keeps the primary delta and lets "
        "the secondary delta add a bounded channel-level modulation"
    ),
)
parser.add_argument(
    "--delta_selection_margin",
    type=float,
    default=0.7,
    help=(
        "secondary-reference confidence margin for normalized selection; "
        "secondary is used only when its normalized delta is less than "
        "margin times the primary normalized delta (default: 0.7)"
    ),
)
parser.add_argument(
    "--delta_soft_beta",
    type=float,
    default=0.5,
    help=(
        "strength of the positive secondary-minus-primary inverse-delta "
        "modulation for soft_modulation (default: 0.5)"
    ),
)
parser.add_argument(
    "--delta_reference_source",
    type=str,
    default="awq",
    choices=["awq", "gptq"],
    help=(
        "identity of the dequantized reference supplied through --awq_model; "
        "this changes provenance/metadata only, while delta remains abs(W-W_ref)"
    ),
)
parser.add_argument(
    "--awq_delta_weighting",
    type=str,
    default="direct",
    choices=[
        "direct",
        "inverse",
        "inverse_channel_relative",
        "scale_inverse_delta",
        "dual_elementwise_mix",
    ],
    help=(
        "mapping from abs(W-W_awq) to K-means sample weights: direct gives "
        "larger weight to larger error; inverse gives bounded larger weight "
        "to smaller error; inverse_channel_relative first aggregates RMS "
        "relative error per input channel and broadcasts that channel score "
        "to every row; scale_inverse_delta multiplies the searched AWQ channel "
        "scale by an inverse-delta modulation; dual_elementwise_mix combines "
        "the primary and secondary bounded inverse deltas for every element"
    ),
)
parser.add_argument(
    "--awq_scale_cache",
    type=str,
    default=None,
    help="AWQ search-result .pt containing the per-channel scale list",
)
parser.add_argument(
    "--awq_scale_power",
    type=float,
    default=2.0,
    help="power applied to median-normalized AWQ channel scales (default: 2.0)",
)
parser.add_argument(
    "--awq_scale_delta_lambda",
    type=float,
    default=1.0,
    help="inverse-delta modulation strength for scale_inverse_delta (default: 1.0)",
)
parser.add_argument(
    "--awq_delta_gamma",
    type=float,
    default=1.0,
    help="strength of bounded inverse-delta weighting (default: 1.0)",
)
parser.add_argument(
    "--awq_delta_min_weight",
    type=float,
    default=0.01,
    help="lower bound for inverse-delta sample weights (default: 0.01)",
)
parser.add_argument(
    "--delta_mix_alpha",
    type=float,
    default=0.5,
    help=(
        "secondary-delta coefficient for dual_elementwise_mix; 0 is the "
        "primary endpoint and 1 is the secondary endpoint (default: 0.5)"
    ),
)
parser.add_argument(
    "--delta_mix_operator",
    choices=["geometric", "convex"],
    default="geometric",
    help="elementwise AWQ/GPTQ inverse-delta fusion operator",
)
parser.add_argument(
    "--awq_delta_fisher_blend",
    type=float,
    default=1.0,
    help=(
        "blend coefficient alpha for row-normalized Fisher and AWQ-delta "
        "weights: (1-alpha)*Fisher + alpha*AWQ_weight (default: 1.0, "
        "pure AWQ-delta); values below 1 require --gradient"
    ),
)
parser.add_argument(
    "--awq_delta_fisher_combine",
    type=str,
    default="convex",
    choices=["convex", "multiplicative"],
    help=(
        "how Fisher and AWQ-delta weights are combined: convex uses "
        "(1-alpha)*Fisher + alpha*delta; multiplicative uses "
        "Fisher*(1+alpha*delta), preserving Fisher's zero/near-zero ordering"
    ),
)
parser.add_argument(
    "--kmeans_modules",
    type=str,
    default=None,
    help=(
        "optional comma-separated module names to quantize (for example "
        "v,o,gate,up,down); omitted modules must be supplied by a fallback "
        "LUT during fake evaluation"
    ),
)
parser.add_argument(
    "--bit",
    type=int,
    default=3,
    help="bitwidth",
    choices=[3, 4],
)
# 范围采用左闭右开区间，例如 ``--range 0,10`` 只处理第 0 到第 9 层。
parser.add_argument(
    "--range", type=str, default=None, help="range of layers to quantize"
)
# 输出目录的主要结构为 ``lut/l{layer}.pkl``；稀疏模式还会生成
# ``outliers/l{layer}.pkl`` 和 ``outlier_config.json``。
parser.add_argument(
    "--output_folder", type=str, required=None, help="path to dump the output"
)
parser.add_argument(
    "--outlier_config",
    type=str,
    default=None,
    help="path to outlier config, obtained from generate_outlier_config.py",
)
parser.add_argument(
    "--sensitivity", type=float, default=0, help="sensitivity for outlier extraction"
)
parser.add_argument(
    "--importance_mask",
    type=str,
    default=None,
    help=(
        "path to a squeezellm_external_importance_mask_v1 .pt file; selected "
        "weights are stored in the sparse FP32 component"
    ),
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="number of row-wise K-means worker processes (default: CPU count)",
)


def load_importance_mask_artifact(path):
    """读取并校验外部逐权重重要性掩码。

    掩码文件必须是 ``squeezellm_external_importance_mask_v1`` 格式，且
    ``selections``（兼容旧名称 ``masks``）应是以完整 Linear 模块名为键的
    字典。掩码选中的权重会进入 FP32 稀疏分量，不参与稠密码本聚类。

    该功能属于 Dense-and-Sparse 路径；dense-only 运行不会调用此函数。
    """
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError("Importance mask artifact must be a dictionary")
    if artifact.get("format") != "squeezellm_external_importance_mask_v1":
        raise ValueError(
            "Unsupported importance mask format; expected "
            "squeezellm_external_importance_mask_v1"
        )
    selections = artifact.get("selections", artifact.get("masks"))
    if not isinstance(selections, dict) or not selections:
        raise ValueError("Importance mask artifact contains no selections")
    return artifact


def get_layer_importance_masks(artifact, model_type, layer_idx):
    """取得指定 Transformer 层中每个目标 Linear 对应的外部掩码。

    不同 Hugging Face 模型/导出工具可能使用 ``model.layers``、
    ``model.decoder.layers``，或省略最外层 ``model.`` 前缀。这里为每个模块
    构造兼容别名，但返回值统一使用 SqueezeLLM 的短名称（q/k/v/o/...），
    以便与分层权重 checkpoint 的键保持一致。
    """
    selections = artifact.get("selections", artifact.get("masks"))
    module_names = get_module_names(model_type)
    sequential_names = get_sequential(model_type)
    layer_prefix = get_layers_name(model_type)
    result = {}

    for module_name, sequential_name in zip(module_names, sequential_names):
        # 例如 LLaMA 第 3 层 q_proj 的标准名为
        # ``model.layers.3.self_attn.q_proj``。
        full_name = f"{layer_prefix}.{layer_idx}.{sequential_name}"
        aliases = [full_name]
        if full_name.startswith("model.layers."):
            aliases.append(
                "model.decoder.layers." + full_name[len("model.layers."):]
            )
        elif full_name.startswith("model.decoder.layers."):
            aliases.append("model.layers." + full_name[len("model.decoder.layers."):])
        if full_name.startswith("model."):
            aliases.append(full_name[len("model."):])

        selection = next(
            (selections[name] for name in aliases if name in selections), None
        )
        if selection is None:
            raise KeyError(
                f"Importance mask for layer {layer_idx} module {module_name} "
                f"was not found; tried {aliases}"
            )
        result[module_name] = selection

    return result


def build_awq_delta_scores(model_layer, awq_layer, module_names, layer_idx):
    """计算 AWQ 反量化误差对应的逐元素 K-means 权重。

    聚类样本仍是原始权重 ``W``；返回的 ``abs(W - W_awq)`` 只传入
    sklearn ``KMeans.fit(..., sample_weight=...)``。计算使用 FP32，避免
    FP16/BF16 差值过早下溢。
    """
    scores = {}
    for name in module_names:
        if name not in model_layer:
            raise KeyError(f"Original layer {layer_idx} is missing module {name}")
        if name not in awq_layer:
            raise KeyError(f"AWQ layer {layer_idx} is missing module {name}")
        original = model_layer[name]
        awq_weight = awq_layer[name]
        if tuple(original.shape) != tuple(awq_weight.shape):
            raise ValueError(
                f"AWQ weight shape mismatch at layer {layer_idx} module {name}: "
                f"original={tuple(original.shape)}, awq={tuple(awq_weight.shape)}"
            )
        scores[name] = torch.abs(original.float() - awq_weight.float())
    return scores


def select_elementwise_delta_scores(
    primary_scores,
    secondary_scores,
    module_names,
    selection="minimum",
    margin=0.7,
    original_weights=None,
):
    """Combine two delta references, optionally with a confidence gate."""
    supported = {
        "minimum",
        "normalized_confidence",
        "channel_normalized_confidence",
        "soft_modulation",
    }
    if selection not in supported:
        raise ValueError(f"Unsupported delta-reference selection: {selection}")

    def median_normalize(score):
        positive = score[score > 0]
        if positive.numel() == 0:
            return torch.ones_like(score)
        return score / torch.median(positive).clamp_min(torch.finfo(score.dtype).eps)

    selected = {}
    statistics = {}
    for name in module_names:
        if name not in primary_scores or name not in secondary_scores:
            raise KeyError(f"Missing primary/secondary delta scores for module {name}")
        primary = primary_scores[name]
        secondary = secondary_scores[name]
        if tuple(primary.shape) != tuple(secondary.shape):
            raise ValueError(
                f"Delta shape mismatch for module {name}: "
                f"primary={tuple(primary.shape)}, secondary={tuple(secondary.shape)}"
            )
        primary_less = primary < secondary
        secondary_less = secondary < primary
        ties = ~(primary_less | secondary_less)
        selected_secondary_channels = None
        if selection == "minimum":
            secondary_selected = secondary_less
            selected[name] = torch.where(secondary_selected, secondary, primary)
        elif selection == "normalized_confidence":
            primary_normalized = median_normalize(primary)
            secondary_normalized = median_normalize(secondary)
            secondary_selected = secondary_normalized < margin * primary_normalized
            selected[name] = torch.where(secondary_selected, secondary, primary)
        elif selection == "channel_normalized_confidence":
            if original_weights is None or name not in original_weights:
                raise KeyError(
                    "channel_normalized_confidence requires original weight for "
                    f"module {name}"
                )
            original = original_weights[name].float()
            eps = torch.finfo(primary.dtype).eps
            weight_rms = torch.sqrt(torch.mean(original.square(), dim=0)).clamp_min(eps)
            primary_relative = torch.sqrt(torch.mean(primary.square(), dim=0)) / weight_rms
            secondary_relative = torch.sqrt(torch.mean(secondary.square(), dim=0)) / weight_rms
            primary_normalized = median_normalize(primary_relative)
            secondary_normalized = median_normalize(secondary_relative)
            selected_secondary_channels = (
                secondary_normalized < margin * primary_normalized
            )
            secondary_selected = selected_secondary_channels.unsqueeze(0).expand_as(primary)
            selected[name] = torch.where(secondary_selected, secondary, primary)
        else:
            # The actual soft combination is applied after both references are
            # converted to inverse channel-relative K-means weights.
            secondary_selected = torch.zeros_like(primary, dtype=torch.bool)
            selected[name] = primary
        statistics[name] = {
            "primary_smaller": int(primary_less.sum().item()),
            "secondary_smaller": int(secondary_less.sum().item()),
            "ties": int(ties.sum().item()),
            "selected_primary": int((~secondary_selected).sum().item()),
            "selected_secondary": int(secondary_selected.sum().item()),
            "total": int(primary.numel()),
        }
        if selected_secondary_channels is not None:
            statistics[name]["selected_secondary_channels"] = int(
                selected_secondary_channels.sum().item()
            )
            statistics[name]["total_channels"] = int(
                selected_secondary_channels.numel()
            )
    return selected, statistics


def transform_awq_delta_row(
    delta,
    weighting="direct",
    gamma=1.0,
    min_weight=0.01,
):
    """把一行 AWQ delta 转换为 K-means sample weight。

    ``inverse`` 使用每行非零 delta 的中位数 ``tau`` 做尺度：

    ``min_weight + (1-min_weight) * (1 / (1 + delta/tau))**gamma``

    它保证 delta 越小权重越大，同时避免 ``1/(delta+eps)``
    在 delta 接近零时爆炸。
    """
    delta = np.asarray(delta, dtype=np.float32)
    if weighting == "direct":
        return delta
    if weighting != "inverse":
        raise ValueError(f"Unsupported AWQ delta weighting: {weighting}")

    positive_delta = delta[delta > 0]
    if positive_delta.size == 0:
        return np.ones_like(delta)

    tau = float(np.median(positive_delta))
    eps = np.finfo(np.float32).eps
    inverse = np.power(1.0 / (1.0 + delta / (tau + eps)), gamma)
    return min_weight + (1.0 - min_weight) * inverse


def build_inverse_channel_relative_weights(
    original_weight,
    delta,
    gamma=1.0,
    min_weight=0.01,
):
    """Build one AWQ-delta weight per input channel, shared across rows.

    AWQ's scaling decision is channel structured, whereas a row-local delta
    median discards that structure. We measure each input channel's RMS
    quantization error relative to its RMS original weight, then invert the
    relative error with the same bounded transform used by ``inverse``.

    The result still depends only on ``W`` and ``abs(W-W_awq)``; no activation,
    Fisher, or sparse/outlier information is used.
    """
    weight = np.asarray(original_weight, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    if weight.shape != delta.shape or weight.ndim != 2:
        raise ValueError(
            "inverse_channel_relative expects matching two-dimensional "
            f"weight/delta arrays, got {weight.shape} and {delta.shape}"
        )

    error_rms = np.sqrt(np.mean(np.square(delta), axis=0, dtype=np.float64))
    weight_rms = np.sqrt(np.mean(np.square(weight), axis=0, dtype=np.float64))
    eps = np.finfo(np.float32).eps
    relative_error = error_rms / np.maximum(weight_rms, eps)
    positive = relative_error[relative_error > 0]
    if positive.size == 0:
        return np.ones(weight.shape[1], dtype=np.float32)

    tau = float(np.median(positive))
    inverse = np.power(1.0 / (1.0 + relative_error / (tau + eps)), gamma)
    result = min_weight + (1.0 - min_weight) * inverse
    return np.asarray(result, dtype=np.float32)


def load_awq_channel_scales(path):
    """Map an AWQ search cache's scale tensors to ``(layer, short_module)``."""
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("scale"), list):
        raise ValueError(f"AWQ scale cache has no list-valued 'scale' entry: {path}")

    suffix_to_module = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "out_proj": "o",
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
        "fc1": "up",
        "fc2": "down",
    }
    result = {}
    for item in artifact["scale"]:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            raise ValueError(f"Malformed AWQ scale entry in {path}: {item!r}")
        _, target_names, scales = item[:3]
        scales = torch.as_tensor(scales).float().cpu()
        for full_name in target_names:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", full_name)
            module_name = suffix_to_module.get(full_name.rsplit(".", 1)[-1])
            if match is None or module_name is None:
                continue
            key = (int(match.group(1)), module_name)
            if key in result and not torch.equal(result[key], scales):
                raise ValueError(f"Conflicting AWQ scales for layer/module {key}")
            result[key] = scales
    return result


def normalize_score_row(score):
    """Normalize a non-negative score row without changing its K-means optimum."""
    score = np.asarray(score, dtype=np.float32)
    positive = score[score > 0]
    if positive.size == 0:
        return np.ones_like(score)
    scale = float(np.median(positive))
    return score / (scale + np.finfo(np.float32).eps)

# 每一行，都有自己的一张 LUT。
# 一次调用只处理一个 Linear 的一行权重。
def kmeans_fit(row_data):
    """对权重矩阵的一行执行逐元素加权的一维 K-means。

    参数 ``row_data`` 是 ``(weights_np, sample_weight, n_cluster)``：

    * ``weights_np`` 的形状为 ``[in_features, 1]``，每个权重是一个一维样本；
    * ``sample_weight`` 是同一行逐权重的 Fisher/梯度平方，或
      ``abs(W - W_awq)``；
    * ``n_cluster`` 为 ``2**bit``，W3/W4 分别得到 8/16 个聚类中心。

    最小化的目标可写为
    ``sum_j S[i,j] * (W[i,j] - center[label[i,j]])**2``。``S`` 由当前
    运行选择为 Fisher 或 AWQ delta。这些元素仍然会被量化，不会被
    单独保留为浮点权重。

    返回值包含 FP32/NumPy 聚类中心和 int8 聚类编号。int8 只是 LUT 中间文件
    的存储形式；pack 阶段才会把编号真正压缩成 3 或 4 bit。
    """
    # weights_np 权重
    # sample_weight 是每个权重对应的 Fisher 或 AWQ delta
    # n_cluster 几 bit 量化，码本大小。
    weights_np, sample_weight, n_cluster = row_data
    def fit(current_sample_weight):
        return KMeans(
            n_clusters=n_cluster,
            random_state=0,
            n_init="auto",
            max_iter=50,
        ).fit(weights_np, sample_weight=current_sample_weight)

    kmeans = fit(sample_weight)
    # Some Qwen3 rows have Fisher mass on fewer effective values than the
    # requested codebook size after sparse outliers are removed.  sklearn can
    # then leave zero-mass cluster centers as NaN even though the total sample
    # weight is positive.  Fall back only for those degenerate rows, keeping
    # sparse placeholders at zero weight so that they do not fit the codebook.
    if not np.all(np.isfinite(kmeans.cluster_centers_)):
        active = weights_np.reshape(-1) != 0
        fallback_weight = active.astype(np.float32)
        if np.count_nonzero(fallback_weight) < n_cluster:
            fallback_weight = np.ones(weights_np.shape[0], dtype=np.float32)
        kmeans = fit(fallback_weight)
        if not np.all(np.isfinite(kmeans.cluster_centers_)):
            raise ValueError("K-means produced non-finite centers after fallback")
    return (
        kmeans.cluster_centers_.reshape(-1), # 这一行权重的非均匀码本
        kmeans.labels_.astype(np.int8, copy=False), # 每个权重对应的码本编号
    )


if __name__ == "__main__":
    args = parser.parse_args()

    if args.awq_model is None and args.gradient is None:
        parser.error("one of --gradient or --awq_model is required")
    if args.sensitivity > 0 and args.gradient is None:
        parser.error("--sensitivity requires --gradient even when --awq_model is used")
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num_workers must be positive")
    if args.awq_delta_gamma <= 0:
        parser.error("--awq_delta_gamma must be positive")
    if args.awq_scale_power <= 0:
        parser.error("--awq_scale_power must be positive")
    if args.awq_scale_delta_lambda <= 0:
        parser.error("--awq_scale_delta_lambda must be positive")
    if not 0 < args.delta_selection_margin <= 1:
        parser.error("--delta_selection_margin must be in (0, 1]")
    if args.delta_soft_beta <= 0:
        parser.error("--delta_soft_beta must be positive")
    if not 0 <= args.awq_delta_min_weight < 1:
        parser.error("--awq_delta_min_weight must be in [0, 1)")
    if not 0 <= args.delta_mix_alpha <= 1:
        parser.error("--delta_mix_alpha must be in [0, 1]")
    if args.awq_delta_fisher_combine == "convex" and not (
        0 <= args.awq_delta_fisher_blend <= 1
    ):
        parser.error(
            "--awq_delta_fisher_blend must be in [0, 1] for convex combination"
        )
    if (
        args.awq_delta_fisher_combine == "multiplicative"
        and args.awq_delta_fisher_blend <= 0
    ):
        parser.error(
            "--awq_delta_fisher_blend must be positive for multiplicative combination"
        )
    if (
        args.awq_delta_fisher_combine == "convex"
        and args.awq_delta_fisher_blend < 1
        and args.gradient is None
    ):
        parser.error("--awq_delta_fisher_blend < 1 requires --gradient")
    if args.awq_delta_fisher_combine == "multiplicative" and (
        args.awq_model is None or args.gradient is None
    ):
        parser.error(
            "--awq_delta_fisher_combine multiplicative requires both "
            "--awq_model and --gradient"
        )
    if args.awq_model is None and args.awq_delta_weighting != "direct":
        parser.error("--awq_delta_weighting only applies when --awq_model is set")
    if args.secondary_delta_model is not None and args.awq_model is None:
        parser.error("--secondary_delta_model requires the primary --awq_model")
    if (
        args.awq_delta_weighting == "dual_elementwise_mix"
        and args.secondary_delta_model is None
    ):
        parser.error(
            "--awq_delta_weighting dual_elementwise_mix requires "
            "--secondary_delta_model"
        )
    if (
        args.secondary_delta_model is not None
        and args.secondary_delta_reference_source == args.delta_reference_source
    ):
        parser.error(
            "primary and secondary delta-reference sources must be different"
        )
    if args.awq_delta_weighting == "scale_inverse_delta" and args.awq_scale_cache is None:
        parser.error("--awq_delta_weighting scale_inverse_delta requires --awq_scale_cache")
    if args.awq_scale_cache is not None and not os.path.isfile(args.awq_scale_cache):
        parser.error(f"--awq_scale_cache must be an existing file: {args.awq_scale_cache}")

    # 先验证输入再创建输出目录。这样用户误把文档中的
    # /path/to/... 占位符当成真实路径时，不会先尝试在根目录写文件。
    input_directories = {"--model": args.model}
    if args.gradient is not None:
        input_directories["--gradient"] = args.gradient
    if args.awq_model is not None:
        input_directories["--awq_model"] = args.awq_model
    if args.secondary_delta_model is not None:
        input_directories["--secondary_delta_model"] = args.secondary_delta_model
    for option, path in input_directories.items():
        if not os.path.isdir(path):
            parser.error(
                f"{option} must be an existing chunk directory containing "
                f"layer_*.pt files; got: {path}"
            )
        if not os.path.isfile(os.path.join(path, "layer_0.pt")):
            parser.error(f"{option} contains no layer_0.pt: {path}")

    if args.output_folder is None:
        parser.error("--output_folder is required")

    # 内建 sensitivity 和外部 importance mask 都用于定义“敏感权重”，同时使用
    # 会让权重归属产生歧义，因此显式禁止。幅值 outlier 配置仍可与外部掩码叠加。
    if args.importance_mask is not None and args.sensitivity > 0:
        parser.error(
            "--importance_mask cannot be combined with --sensitivity; an "
            "external sensitive mask may be combined with --outlier_config"
        )

    # 若未显式传入模型类型，则按项目原有规则从模型标识中推断。
    model_type = args.model_type or parse_model(args.model)
    module_names = get_module_names(model_type)
    if args.kmeans_modules is None:
        kmeans_module_names = list(module_names)
    else:
        kmeans_module_names = [
            name.strip() for name in args.kmeans_modules.split(",") if name.strip()
        ]
        if not kmeans_module_names:
            parser.error("--kmeans_modules must contain at least one module name")
        if len(kmeans_module_names) != len(set(kmeans_module_names)):
            parser.error("--kmeans_modules contains duplicate module names")
        unknown_modules = set(kmeans_module_names).difference(module_names)
        if unknown_modules:
            parser.error(
                "--kmeans_modules contains unsupported names "
                f"{sorted(unknown_modules)}; supported modules are {module_names}"
            )
        # Preserve the model's canonical module order regardless of CLI order.
        selected_modules = set(kmeans_module_names)
        kmeans_module_names = [
            name for name in module_names if name in selected_modules
        ]
    delta_source = args.delta_reference_source
    secondary_delta_source = args.secondary_delta_reference_source
    if args.awq_delta_weighting == "dual_elementwise_mix":
        delta_source_label = (
            f"{delta_source}_{secondary_delta_source}_elementwise_"
            f"{args.delta_mix_operator}_alpha{args.delta_mix_alpha:g}"
        )
    elif args.secondary_delta_model is not None:
        delta_source_label = (
            f"{delta_source}_{secondary_delta_source}_elementwise_"
            f"{args.delta_reference_selection}"
        )
    else:
        delta_source_label = delta_source
    if args.awq_model is None:
        kmeans_weight_source = "fisher"
    elif args.awq_delta_fisher_combine == "multiplicative":
        kmeans_weight_source = f"fisher_{delta_source_label}_delta_modulation"
    elif args.awq_delta_fisher_blend < 1:
        kmeans_weight_source = f"fisher_{delta_source_label}_delta_blend"
    elif args.awq_delta_weighting == "scale_inverse_delta":
        kmeans_weight_source = f"{delta_source_label}_scale_inverse_delta_modulation"
    elif args.awq_delta_weighting == "inverse_channel_relative":
        kmeans_weight_source = f"{delta_source_label}_inverse_channel_relative_delta"
    elif args.awq_delta_weighting == "inverse":
        kmeans_weight_source = f"{delta_source_label}_inverse_delta"
    elif args.awq_delta_weighting == "dual_elementwise_mix":
        kmeans_weight_source = f"{delta_source_label}_inverse_delta"
    else:
        kmeans_weight_source = f"{delta_source_label}_delta"
    print(f"K-means sample-weight source: {kmeans_weight_source}")

    awq_channel_scales = (
        load_awq_channel_scales(args.awq_scale_cache)
        if args.awq_delta_weighting == "scale_inverse_delta"
        else None
    )

    # dense-only 模式下 importance_artifact 为 None，不加载任何稀疏掩码。
    importance_artifact = (
        load_importance_mask_artifact(args.importance_mask)
        if args.importance_mask is not None
        else None
    )

    # 三类参数中任意一个生效都会开启 Dense-and-Sparse 路径。若三者均未设置，
    # is_outlier_mode=False，后续整张权重矩阵直接进入 K-means，即 dense-only。
    is_outlier_mode = (
        args.outlier_config is not None
        or args.sensitivity > 0
        or importance_artifact is not None
    )

    if args.outlier_config is not None:
        # outlier_threshold 是整个配置的统计信息；outlier_config 则按层记录每个
        # Linear 的上下阈值，实际提取权重时使用后者。
        with open(args.outlier_config, "r") as f:
            outlier_config = json.load(f)
        outlier_threshold = outlier_config["outlier_threshold"]
        outlier_config = outlier_config["outlier_config"]
    else:
        outlier_threshold = 0
        outlier_config = None

    # 无论是否启用稀疏分量，稠密部分的码本都写入 lut 子目录。
    lut_folder = f"{args.output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    # 无论 dense-only 还是 D+S 都记录 K-means 权重来源，避免在后续
    # PPL 对比时把 Fisher 码本和 AWQ-delta 码本混淆。
    with open(f"{args.output_folder}/kmeans_config.json", "w") as f:
        json.dump(
            {
                "format": "squeezellm_kmeans_config_v1",
                "bit": args.bit,
                "weight_source": kmeans_weight_source,
                "kmeans_modules": kmeans_module_names,
                "model_chunks": os.path.abspath(args.model),
                "gradient_chunks": (
                    os.path.abspath(args.gradient) if args.gradient is not None else None
                ),
                "awq_dequantized_chunks": (
                    os.path.abspath(args.awq_model)
                    if args.awq_model is not None and delta_source == "awq"
                    else None
                ),
                "quantized_reference_chunks": (
                    os.path.abspath(args.awq_model)
                    if args.awq_model is not None
                    else None
                ),
                "delta_reference_source": (
                    delta_source if args.awq_model is not None else None
                ),
                "secondary_quantized_reference_chunks": (
                    os.path.abspath(args.secondary_delta_model)
                    if args.secondary_delta_model is not None
                    else None
                ),
                "secondary_delta_reference_source": (
                    secondary_delta_source
                    if args.secondary_delta_model is not None
                    else None
                ),
                "delta_reference_selection": (
                    args.delta_reference_selection
                    if args.secondary_delta_model is not None
                    else None
                ),
                "delta_selection_margin": (
                    args.delta_selection_margin
                    if args.secondary_delta_model is not None
                    and args.delta_reference_selection
                    in {"normalized_confidence", "channel_normalized_confidence"}
                    else None
                ),
                "delta_soft_beta": (
                    args.delta_soft_beta
                    if args.secondary_delta_model is not None
                    and args.delta_reference_selection == "soft_modulation"
                    else None
                ),
                "awq_delta_weighting": (
                    args.awq_delta_weighting
                    if args.awq_model is not None
                    else None
                ),
                "awq_delta_gamma": (
                    args.awq_delta_gamma if args.awq_model is not None else None
                ),
                "awq_delta_min_weight": (
                    args.awq_delta_min_weight
                    if args.awq_model is not None
                    else None
                ),
                "delta_mix_alpha": (
                    args.delta_mix_alpha
                    if args.awq_delta_weighting == "dual_elementwise_mix"
                    else None
                ),
                "delta_mix_operator": (
                    args.delta_mix_operator
                    if args.awq_delta_weighting == "dual_elementwise_mix"
                    else None
                ),
                "awq_scale_cache": (
                    os.path.abspath(args.awq_scale_cache)
                    if args.awq_scale_cache is not None
                    else None
                ),
                "awq_scale_power": (
                    args.awq_scale_power
                    if args.awq_delta_weighting == "scale_inverse_delta"
                    else None
                ),
                "awq_scale_delta_lambda": (
                    args.awq_scale_delta_lambda
                    if args.awq_delta_weighting == "scale_inverse_delta"
                    else None
                ),
                "awq_delta_fisher_blend": (
                    args.awq_delta_fisher_blend
                    if args.awq_model is not None
                    else None
                ),
                "awq_delta_fisher_combine": (
                    args.awq_delta_fisher_combine
                    if args.awq_model is not None
                    else None
                ),
                "objective": (
                    "sum(normalized(Fisher) * "
                    f"(1+alpha*normalized({delta_source.upper()}_delta_weight)) * "
                    "(W-center[label])^2)"
                    if args.awq_model is not None
                    and args.awq_delta_fisher_combine == "multiplicative"
                    else
                    "sum(((1-alpha)*normalized(Fisher) + "
                    f"alpha*normalized({delta_source.upper()}_delta_weight)) * "
                    "(W-center[label])^2)"
                    if args.awq_model is not None
                    and args.awq_delta_fisher_blend < 1
                    else "sum(normalized(AWQ_scale)^scale_power * "
                    "(1+lambda*normalized(bounded_inverse(relative_delta))) * "
                    "(W-center[label])^2)"
                    if args.awq_model is not None
                    and args.awq_delta_weighting == "scale_inverse_delta"
                    else "sum(bounded_inverse(channel_RMS(abs(W-W_"
                    f"{delta_source}))/channel_RMS(W)) * (W-center[label])^2)"
                    if args.awq_model is not None
                    and args.awq_delta_weighting == "inverse_channel_relative"
                    else f"sum(bounded_inverse(abs(W-W_{delta_source})) * (W-center[label])^2)"
                    if args.awq_model is not None
                    and args.awq_delta_weighting == "inverse"
                    else (
                        f"sum(abs(W-W_{delta_source}) * (W-center[label])^2)"
                        if args.awq_model is not None
                        else "sum(Fisher * (W-center[label])^2)"
                    )
                ),
            },
            f,
            indent=4,
        )

    if is_outlier_mode:
        # 只有 Dense-and-Sparse 路径才创建 outliers 子目录并记录稀疏选择配置。
        outlier_folder = f"{args.output_folder}/outliers"
        print(
            f"Running outlier mode with outlier folder threshold {outlier_threshold} "
            f"and sensitivity {args.sensitivity}, importance mask {args.importance_mask}"
        )
        if not os.path.exists(outlier_folder):
            os.makedirs(outlier_folder)

        # 保存产生本批 outliers 的配置，便于复现实验和核对 checkpoint 来源。
        with open(f"{args.output_folder}/outlier_config.json", "w") as f:
            json.dump(
                {
                    "outlier_threshold": outlier_threshold,
                    "sensitivity": args.sensitivity,
                    "outlier_config": outlier_config,
                    "importance_mask": args.importance_mask,
                    "importance_selection": (
                        importance_artifact.get("selection")
                        if importance_artifact is not None
                        else None
                    ),
                },
                f,
                indent=4,
            )

    if args.range:
        # Python range 的结束下标不包含在内："2,5" -> [2, 3, 4]。
        ranges = args.range.split(",")
        ranges = [int(r) for r in ranges]
        ran = list(range(ranges[0], ranges[1]))
    else:
        # 默认假设模型分块目录中只有 layer_*.pt 文件，因此直接用文件数作为层数。
        # 若目录里混有其他文件，这个计数会偏大；使用时不要向该目录添加杂项文件。
        nlayers = len([f for f in os.listdir(args.model)])
        ran = list(range(nlayers))

    print(f"Quantizing layers {ran}")

    # 每个 K-means 任务只处理一行权重。外层进程池并行这些任务，而每个进程的
    # OMP_NUM_THREADS=1，从而让并行粒度保持可控。
    pool = Pool(args.num_workers or os.cpu_count())
    delta_selection_by_layer = {}

    for l in ran:
        # 当前 for 已经直接遍历 ran，这个判断通常恒为 False；保留它是为了不改变
        # 原有控制流，并兼容未来 ran 的构造方式可能发生变化。
        if ran is not None and l not in ran:
            print(f"Skipping layer {l}")
            continue

        lut_file_name = f"{lut_folder}/l{l}.pkl"

        if is_outlier_mode:
            outlier_file_name = f"{outlier_folder}/l{l}.pkl"
        else:
            outlier_file_name = None

        # 支持断点续跑。dense-only 只要求 LUT 文件完整存在；稀疏模式必须同时有
        # LUT 和 outlier 文件，避免上次运行只写出一半结果时误跳过该层。
        layer_outputs_exist = (
            os.path.isfile(lut_file_name)
            and os.path.getsize(lut_file_name) > 0
            and (
                not is_outlier_mode
                or (
                    os.path.isfile(outlier_file_name)
                    and os.path.getsize(outlier_file_name) > 0
                )
            )
        )
        if layer_outputs_exist:
            print(
                f"Skipping layer {l}, LUT"
                + (" and sparse output" if is_outlier_mode else "")
                + " already exist"
            )
            continue

        print(f"Quantizing layer {l}")

        # layer 文件都是字典，键为 q/k/v/o/gate/up/down 等短模块名；
        # 所有张量的形状都应为 [out_features, in_features]。
        gradient_layer = None
        if args.gradient is not None:
            gradient_path = os.path.join(args.gradient, f"layer_{l}.pt")
            # {
            #     "q":    fisher_of_q_proj,
            #     "k":    fisher_of_k_proj,
            #     ...
            # }
            if not os.path.isfile(gradient_path):
                raise FileNotFoundError(
                    f"Needs chunked gradient file at {gradient_path}"
                )
            gradient_layer = torch.load(gradient_path, map_location="cpu")

        model_layer_path = os.path.join(args.model, f"layer_{l}.pt")
        if not os.path.isfile(model_layer_path):
            raise FileNotFoundError(
                f"Needs chunked model weight file at {model_layer_path}"
            )
        model_layer = torch.load(model_layer_path, map_location="cpu")

        secondary_score_for_weighting = None
        if args.awq_model is not None:
            awq_layer_path = os.path.join(args.awq_model, f"layer_{l}.pt")
            if not os.path.isfile(awq_layer_path):
                raise FileNotFoundError(
                    f"Needs chunked AWQ dequantized weight file at {awq_layer_path}"
                )
            # {
            #     "q": abs(original_q - awq_dequantized_q),
            #     ...
            # }
            awq_layer = torch.load(awq_layer_path, map_location="cpu")
            primary_score_layer = build_awq_delta_scores(
                model_layer, awq_layer, kmeans_module_names, l
            )
            del awq_layer
            if args.secondary_delta_model is not None:
                secondary_layer_path = os.path.join(
                    args.secondary_delta_model, f"layer_{l}.pt"
                )
                if not os.path.isfile(secondary_layer_path):
                    raise FileNotFoundError(
                        "Needs chunked secondary dequantized weight file at "
                        f"{secondary_layer_path}"
                    )
                secondary_layer = torch.load(
                    secondary_layer_path, map_location="cpu"
                )
                secondary_score_layer = build_awq_delta_scores(
                    model_layer, secondary_layer, kmeans_module_names, l
                )
                if args.awq_delta_weighting == "dual_elementwise_mix":
                    score_layer = primary_score_layer
                    secondary_score_for_weighting = secondary_score_layer
                else:
                    score_layer, layer_selection_stats = select_elementwise_delta_scores(
                        primary_score_layer,
                        secondary_score_layer,
                        kmeans_module_names,
                        selection=args.delta_reference_selection,
                        margin=args.delta_selection_margin,
                        original_weights=model_layer,
                    )
                    if args.delta_reference_selection == "soft_modulation":
                        secondary_score_for_weighting = secondary_score_layer
                    delta_selection_by_layer[str(l)] = layer_selection_stats
                    primary_count = sum(
                        item["selected_primary"]
                        for item in layer_selection_stats.values()
                    )
                    secondary_count = sum(
                        item["selected_secondary"]
                        for item in layer_selection_stats.values()
                    )
                    print(
                        f"Layer {l} delta combination "
                        f"({args.delta_reference_selection}): "
                        f"{delta_source}={primary_count}, "
                        f"{secondary_delta_source}={secondary_count}"
                    )
                del secondary_layer
                if secondary_score_for_weighting is None:
                    del secondary_score_layer
            else:
                score_layer = primary_score_layer
        else:
            score_layer = gradient_layer

        if is_outlier_mode:
            # 此分支先原地移除被选中的权重，并把原值收集到 outliers；后续 K-means
            # 只拟合留下的稠密部分。dense-only 不进入整个分支。
            if importance_artifact is not None:
                print(f"Removing weights selected by external mask for layer {l}")
                importance_masks_layer = get_layer_importance_masks(
                    importance_artifact, model_type, l
                )
                outliers = remove_outliers_by_mask(
                    model=model_layer,
                    importance_masks=importance_masks_layer,
                )
                if outlier_config is not None:
                    print(
                        f"Adding fixed magnitude outliers for layer {l} after "
                        "external sensitive-mask extraction"
                    )
                    outliers = remove_outliers_by_threshold(
                        model=model_layer,
                        outlier_config=outlier_config[l],
                        outlier_weights=outliers,
                    )
            else:
                print(
                    f"Removing outliers outlier={outlier_threshold}, "
                    f"sensitivity={args.sensitivity}"
                )
                outlier_config_layer = (
                    outlier_config[l] if outlier_config is not None else None
                )
                outliers = remove_outliers(
                    model=model_layer,
                    sensitivity=args.sensitivity,
                    outlier_config=outlier_config_layer,
                    gradients=gradient_layer,
                )

        config_per_layer = {}

        # 按固定模块顺序构建任务。每一个 task 对应一个 Linear 的一个输出通道。
        # 任务顺序稍后必须原样复用，才能把并行结果正确放回相应模块和权重行。
        kmeans_tasks = []
        for name in kmeans_module_names:
            # 显式转为 float32 再交给 NumPy/sklearn，避免原模型为 FP16/BF16 时
            # 聚类数值精度不足或 NumPy 不支持 BF16。
            if name not in score_layer:
                raise KeyError(
                    f"{kmeans_weight_source} scores for layer {l} are missing "
                    f"module {name}"
                )
            g = score_layer[name].float().numpy()
            secondary_g = (
                secondary_score_for_weighting[name].float().numpy()
                if args.awq_delta_weighting == "dual_elementwise_mix"
                else None
            )
            module_weight = model_layer[name]
            _weights_np = module_weight.float().numpy()
            channel_sample_weight = None
            if (
                args.awq_model is not None
                and args.awq_delta_weighting == "inverse_channel_relative"
            ):
                channel_sample_weight = build_inverse_channel_relative_weights(
                    _weights_np,
                    g,
                    gamma=args.awq_delta_gamma,
                    min_weight=args.awq_delta_min_weight,
                )
            elif (
                args.awq_model is not None
                and args.awq_delta_weighting == "scale_inverse_delta"
            ):
                key = (l, name)
                if key not in awq_channel_scales:
                    raise KeyError(
                        f"AWQ scale cache is missing layer {l} module {name}"
                    )
                scale = awq_channel_scales[key].numpy()
                if scale.shape != (module_weight.shape[1],):
                    raise ValueError(
                        f"AWQ scale shape mismatch at layer {l} module {name}: "
                        f"expected {(module_weight.shape[1],)}, got {scale.shape}"
                    )
                scale_weight = np.power(
                    normalize_score_row(scale), args.awq_scale_power
                )
                inverse_delta = build_inverse_channel_relative_weights(
                    _weights_np,
                    g,
                    gamma=args.awq_delta_gamma,
                    min_weight=args.awq_delta_min_weight,
                )
                delta_modulation = normalize_score_row(inverse_delta)
                channel_sample_weight = scale_weight * (
                    1.0 + args.awq_scale_delta_lambda * delta_modulation
                )
                if (
                    args.secondary_delta_model is not None
                    and args.delta_reference_selection == "soft_modulation"
                ):
                    secondary_g = (
                        secondary_score_for_weighting[name].float().numpy()
                    )
                    secondary_inverse = build_inverse_channel_relative_weights(
                        _weights_np,
                        secondary_g,
                        gamma=args.awq_delta_gamma,
                        min_weight=args.awq_delta_min_weight,
                    )
                    secondary_modulation = normalize_score_row(secondary_inverse)
                    positive_gain = np.maximum(
                        secondary_modulation - delta_modulation, 0.0
                    )
                    channel_sample_weight = channel_sample_weight * (
                        1.0 + args.delta_soft_beta * positive_gain
                    )

            # b-bit 编号最多表达 2**b 个值，即 W3 使用 8 个中心、W4 使用 16 个。
            n_cluster = 2**args.bit

            for i in range(module_weight.shape[0]):
                # Linear.weight 的第 i 行对应第 i 个输出通道。当前实现每行只有一组
                # 码本（group 数恒为 1），而不是把输入维度再切成多个 group。
                weights_np_temp = _weights_np[i, :]
                weights_np = weights_np_temp.reshape(-1, 1)

                # 稀疏路径会把已提取权重的位置置零；将这些位置的聚类权重也
                # 清零，可防止占位零参与码本拟合。dense-only 中通常所有权重非零，
                # 所以 sample_weight 基本就等于原始 score。
                # dense-only 不应用 W!=0 过滤：原始权重恰好为零的元素
                # 仍是合法样本，特别是它的 AWQ delta 可能非零。
                weight_mask = (
                    weights_np_temp != 0
                    if is_outlier_mode
                    else np.ones_like(weights_np_temp, dtype=bool)
                )
                raw_score = g[i, :]
                if args.awq_model is not None:
                    if args.awq_delta_weighting == "dual_elementwise_mix":
                        primary_weight = transform_awq_delta_row(
                            raw_score,
                            weighting="inverse",
                            gamma=args.awq_delta_gamma,
                            min_weight=args.awq_delta_min_weight,
                        )
                        secondary_weight = transform_awq_delta_row(
                            secondary_g[i, :],
                            weighting="inverse",
                            gamma=args.awq_delta_gamma,
                            min_weight=args.awq_delta_min_weight,
                        )
                        alpha = args.delta_mix_alpha
                        if args.delta_mix_operator == "geometric":
                            eps = np.finfo(np.float32).eps
                            sample_weight = np.exp(
                                (1.0 - alpha) * np.log(primary_weight + eps)
                                + alpha * np.log(secondary_weight + eps)
                            )
                        else:
                            sample_weight = (
                                (1.0 - alpha) * primary_weight
                                + alpha * secondary_weight
                            )
                    elif channel_sample_weight is not None:
                        sample_weight = channel_sample_weight
                    else:
                        sample_weight = transform_awq_delta_row(
                            raw_score,
                            weighting=args.awq_delta_weighting,
                            gamma=args.awq_delta_gamma,
                            min_weight=args.awq_delta_min_weight,
                        )
                    if (
                        args.awq_delta_fisher_combine == "multiplicative"
                        or args.awq_delta_fisher_blend < 1
                    ):
                        fisher_score = gradient_layer[name][i, :].float().numpy()
                        alpha = args.awq_delta_fisher_blend
                        fisher_weight = normalize_score_row(fisher_score)
                        delta_weight = normalize_score_row(sample_weight)
                        if args.awq_delta_fisher_combine == "multiplicative":
                            sample_weight = fisher_weight * (
                                1.0 + alpha * delta_weight
                            )
                        else:
                            sample_weight = (
                                (1.0 - alpha) * fisher_weight
                                + alpha * delta_weight
                            )
                else:
                    sample_weight = raw_score
                sample_weight = sample_weight * weight_mask

                # sklearn 不接受总权重为零的加权聚类。遇到全零 score 行时退化为
                # 普通（所有样本等权）K-means，确保仍能生成完整码本。
                if not np.all(np.isfinite(sample_weight)):
                    raise ValueError(
                        f"Non-finite {kmeans_weight_source} sample weights at "
                        f"layer {l} module {name} row {i}"
                    )
                if np.sum(sample_weight, dtype=np.float64) <= 0:
                    sample_weight = np.ones_like(sample_weight)
                kmeans_tasks.append((weights_np, sample_weight, n_cluster))

        # imap 保留输入任务顺序，因此结果可以使用 task_idx 顺序还原；tqdm 仅显示
        # 当前 Transformer 层所有输出通道的聚类进度。
        kmeans_results = list(tqdm(pool.imap(kmeans_fit, kmeans_tasks), total=len(kmeans_tasks)))

        # 将扁平的并行结果还原成 pack.py 所需的层级结构：
        # config_per_layer[module][output_channel][group]
        #     = (centers[K], labels[in_features])
        # 当前 group 数为 1，所以每个 (centers, labels) 外面仍包一层 list。
        task_idx = 0
        for name in kmeans_module_names:
            config_per_row = []
            module_weight = model_layer[name]
            for i in range(module_weight.shape[0]):
                centers, labels = kmeans_results[task_idx]
                task_idx += 1
                config_per_row.append([(centers, labels)])
            config_per_layer[name] = config_per_row


        # LUT 中间文件同时保存码本中心和未压缩的编号。真正的 bit packing 不在
        # 这里发生，而是在 pack.py 调用 QuantLinearLUT.pack2 时完成。
        with open(lut_file_name, "wb") as f:
            print(f"Saving layer lut to {lut_folder}/l{l}.pkl")
            pickle.dump(config_per_layer, f)

        if is_outlier_mode:
            with open(outlier_file_name, "wb") as f:
                print(f"Saving layer outliers to {outlier_folder}/l{l}.pkl")
                # remove_outliers 返回值的第 0 项是按模块排列的稀疏权重列表；转为
                # PyTorch sparse tensor 可避免在中间文件中保存大量显式零值。
                pickle.dump([x.to_sparse() for x in outliers[0]], f)

    # 显式关闭并等待所有 worker，保证退出前所有 K-means 任务和文件写入均完成。
    pool.close()
    pool.join()

    if args.secondary_delta_model is not None and delta_selection_by_layer:
        totals = {
            "primary_smaller": 0,
            "secondary_smaller": 0,
            "ties": 0,
            "selected_primary": 0,
            "selected_secondary": 0,
            "total": 0,
        }
        for layer_stats in delta_selection_by_layer.values():
            for module_stats in layer_stats.values():
                for key in totals:
                    totals[key] += module_stats[key]
        with open(
            f"{args.output_folder}/delta_selection_stats.json", "w"
        ) as f:
            json.dump(
                {
                    "selection": args.delta_reference_selection,
                    "selection_margin": args.delta_selection_margin,
                    "soft_beta": args.delta_soft_beta,
                    "tie_policy": "primary/default",
                    "primary_source": delta_source,
                    "primary_chunks": os.path.abspath(args.awq_model),
                    "secondary_source": secondary_delta_source,
                    "secondary_chunks": os.path.abspath(
                        args.secondary_delta_model
                    ),
                    "totals": totals,
                    "layers": delta_selection_by_layer,
                },
                f,
                indent=4,
            )

# nuq.py 完成了“量化数值的确定”，但没有完成“真实低比特打包”。
# 假设一个模型层只包含 q，而且 q 只有两行
# layer_lut = {
#     "q": [
#         [
#             (
#                 # 第 0 行的码本
#                 np.array([-0.5, -0.1, 0.1, 0.5]),
#                 # 第 0 行每个权重的码本编号
#                 np.array([0, 1, 1, 2, 3], dtype=np.int8),
#             )
#         ],
#         [
#             (
#                 # 第 1 行的码本
#                 np.array([-0.8, -0.2, 0.3, 0.9]),

#                 # 第 1 行每个权重的码本编号
#                 np.array([1, 0, 2, 2, 3], dtype=np.int8),
#             )
#         ],
#     ]
# }

# 每个模块的数据结构是：
# lut_layer[module][output_channel][group]
#     = (centers, labels)
