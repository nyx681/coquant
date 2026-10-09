import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name
import torch.nn.functional as F

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]

# 全局变量：统计 max_dot_idx 的取值次数
max_dot_idx_stats = {
    'count_0': 0,  # max_dot_idx == 0 的次数
    'count_1': 0,  # max_dot_idx == 1 的次数
    'count_2': 0,  # max_dot_idx == 2 的次数
    'total': 0    # 总调用次数
}

def get_max_dot_idx_stats():
    """获取 max_dot_idx 的统计信息"""
    stats = max_dot_idx_stats.copy()
    if stats['total'] > 0:
        stats['percent_0'] = stats['count_0'] / stats['total'] * 100
        stats['percent_1'] = stats['count_1'] / stats['total'] * 100
        stats['percent_2'] = stats['count_2'] / stats['total'] * 100
    else:
        stats['percent_0'] = 0.0
        stats['percent_1'] = 0.0
        stats['percent_2'] = 0.0
    return stats

def print_max_dot_idx_stats():
    """打印 max_dot_idx 的统计信息"""
    stats = get_max_dot_idx_stats()
    print(f"max_dot_idx Statistics:")
    print(f"  Total calls: {stats['total']}")
    print(f"  count_0 (w1): {stats['count_0']} ({stats['percent_0']:.2f}%)")
    print(f"  count_1 (w2): {stats['count_1']} ({stats['percent_1']:.2f}%)")
    print(f"  count_2 (w3): {stats['count_2']} ({stats['percent_2']:.2f}%)")

def reset_max_dot_idx_stats():
    """重置 max_dot_idx 的统计信息"""
    global max_dot_idx_stats
    max_dot_idx_stats = {
        'count_0': 0,
        'count_1': 0,
        'count_2': 0,
        'total': 0
    }


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)

def top_sum_threhold(fractional_parts):
    row_sums = fractional_parts.sum(dim=1)  # [N]

    # 2. 转为 int 下标（比如四舍五入）
    indices = row_sums.round().long()  # [N]

    # 3. 对每行排序
    sorted_parts, _ = torch.sort(fractional_parts, dim=1, descending=True)  # [N, D]

    # 4. 取每行第 sum 位的元素（注意下标不能越界）
    D = fractional_parts.size(1)
    indices = torch.clamp(indices, 0, D-1)  # 防止越界

    # 5. 用 gather 取出每行对应下标的元素
    thresholds = sorted_parts.gather(1, indices.unsqueeze(1))  # [N, 1]

    return thresholds

def top_sum_threhold_balance(fractional_parts,t = 0.5):
    row_sums = fractional_parts.sum(dim=1)  # [N]
    original_sum = (fractional_parts > 0.5).sum(dim=1)

    # 2. 转为 int 下标（比如四舍五入）
    #print(f"t: {t}")
    row_sums = t*row_sums + (1-t)*original_sum
    indices = row_sums.round().long()  # [N]

    # 3. 对每行排序
    sorted_parts, _ = torch.sort(fractional_parts, dim=1, descending=True)  # [N, D]

    # 4. 取每行第 sum 位的元素（注意下标不能越界）
    D = fractional_parts.size(1)
    indices = torch.clamp(indices, 0, D-1)  # 防止越界

    # 5. 用 gather 取出每行对应下标的元素
    thresholds = sorted_parts.gather(1, indices.unsqueeze(1))  # [N, 1]

    return thresholds


def top_sum_threhold_balance_value(fractional_parts,t = 0.5):
    row_sums = fractional_parts.sum(dim=1)  # [N]
    #original_sum = (fractional_parts > 0.5).sum(dim=1)

    # 2. 转为 int 下标（比如四舍五入）
    #print(f"t: {t}")
    #row_sums = t*row_sums + (1-t)*original_sum
    indices = row_sums.round().long()  # [N]

    # 3. 对每行排序
    sorted_parts, _ = torch.sort(fractional_parts, dim=1, descending=True)  # [N, D]

    # 4. 取每行第 sum 位的元素（注意下标不能越界）
    D = fractional_parts.size(1)
    indices = torch.clamp(indices, 0, D-1)  # 防止越界

    # 5. 用 gather 取出每行对应下标的元素
    thresholds = sorted_parts.gather(1, indices.unsqueeze(1))  # [N, 1]

    thresholds = t* thresholds + (1-t)*0.5

    return thresholds


def top_sum_number(fractional_parts):
    row_sums = fractional_parts.sum(dim=1)  # [N]
    #original_sum = (fractional_parts > 0.5).sum(dim=1)

    # 2. 转为 int 下标（比如四舍五入）
    #print(f"t: {t}")
    #row_sums = t*row_sums + (1-t)*original_sum
    indices = row_sums.round().long()  # [N]

    # 3. 对每行排序
    sorted_parts, sorted_indices = torch.sort(fractional_parts, dim=1, descending=True)  # [N, D]

    _, sorted_indices_2 = torch.sort(sorted_indices, dim=1, descending=False)
    test_restored = torch.gather(sorted_parts, 1, sorted_indices_2)
    # print(f"test_restored: {test_restored.shape}")
    # print(f"fractional_parts: {fractional_parts.shape}")
    assert torch.all(test_restored == fractional_parts)
    
    # 4. 取每行第 sum 位的元素（注意下标不能越界）
    D = fractional_parts.size(1)
    indices = torch.clamp(indices, 0, D-1)  # 防止越界

    col_indices = torch.arange(D, device=sorted_parts.device)  # [0, 1, 2, ..., D-1]
    indices_expanded = indices.unsqueeze(1)
    mask = col_indices <= indices_expanded
    result = mask.half()
    final_result = torch.gather(result, 1, sorted_indices_2)

    # 5. 用 gather 取出每行对应下标的元素
    #thresholds = sorted_parts.gather(1, indices.unsqueeze(1))  # [N, 1]

    # thresholds = t* thresholds + (1-t)*0.5

    return final_result

def median_threshold_number(fractional_parts):
    """
    对 fractional_parts 的每一行，中位数以上的值变成1，中位数以下的值变成0
    """
    median_per_row = torch.median(fractional_parts, dim=-1, keepdim=True)[0]  # shape: (num_rows, 1)
    mask = (fractional_parts >= median_per_row).half()  # shape: (num_rows, q_group_size)，中位数以上为1，以下为0，使用half与top_sum_number保持一致
    return mask

def count_and_entropy(tensor):
    # 将tensor展平
    flat_tensor = tensor.view(-1)
    
    # 获取唯一值和每个值的个数
    unique_values, counts = torch.unique(flat_tensor, return_counts=True)
    
    # 计算概率
    total = counts.sum()
    probs = counts.float() / total
    
    # 计算熵 (避免log(0))
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
    
    # print("唯一值:", unique_values)
    # print("每个值的个数:", counts)
    # print("概率:", probs)
    # print("熵:", entropy.item())
    
    return entropy, unique_values, counts
# core quantization method (simulated quantization)

def row_entropy_vectorized(tensor):
    """
    向量化计算每行的熵
    """
    N, D = tensor.shape
    device = tensor.device

    # 1. 统计全局 unique values
    unique_values = torch.unique(tensor)
    K = unique_values.numel()
    print(f"unique_values: {unique_values.shape}")
    print(f"K: {K}")
    # 2. 对每行统计每个 unique value 的 count
    # [N, D] -> [N, K, D]，每个位置和 unique_values 比较
    eq = tensor.unsqueeze(1) == unique_values.view(1, -1, 1)  # [N, K, D]
    counts = eq.sum(dim=2)  # [N, K]，每行每个 unique value 的 count

    # 3. 计算概率
    probs = counts.float() / D  # [N, K]

    # 4. 计算熵
    # 避免 log(0)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)  # [N]

    del eq, counts, unique_values  # 删除变量
    gc.collect()    # Python垃圾回收
    torch.cuda.empty_cache() 

    return entropy


def output_cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

def output_dot_similarity(a, b):
    return torch.sum(a * b, dim=-1)
def output_dot_similarity_abs(a, b):
    return torch.sum(a.abs() * b.abs(), dim=-1)
def output_mse_similarity(a, b):
    """
    每行做差取绝对值然后求和
    """
    return -torch.sum((a - b).abs(), dim=-1)

def output_rel_err_similarity(a, b):
    """
    逐行计算相对误差相似度
    对于每一行，计算 (a - b) / (b.abs() + eps) 的绝对值，然后求和
    返回负值，因为误差越小越好（相似度越高）
    """
    eps = 1e-8
    rel_err_per_row = ((a - b) / (b.abs() + eps)).abs().sum(dim=-1)  # shape: (num_rows,)
    return -rel_err_per_row  # 取负值，因为误差越小越好

def output_weighted_dot_similarity(a, b):
    """
    逐行计算加权点积相似度
    使用 b 的绝对值作为权重，大权重的影响更大
    公式：sum(a * b * |b|) 对每行求和
    """
    return torch.sum(a * b * b.abs(), dim=-1)

def output_normalized_dot_similarity(a, b):
    """
    逐行计算归一化点积相似度
    先归一化每行，再计算点积，避免数值问题
    公式：sum((a / ||a||) * (b / ||b||))，如果范数为0则返回0
    """
    eps = 1e-8
    a_norm = torch.norm(a, p=2, dim=-1, keepdim=True)  # shape: (num_rows, 1)
    b_norm = torch.norm(b, p=2, dim=-1, keepdim=True)  # shape: (num_rows, 1)
    a_normalized = a / (a_norm + eps)
    b_normalized = b / (b_norm + eps)
    return torch.sum(a_normalized * b_normalized, dim=-1)

def output_weighted_l2_similarity(a, b):
    """
    逐行计算加权L2距离（取负值）
    使用 b 的绝对值作为权重
    公式：-sum(|a - b| * |b|) 对每行求和
    """
    return -torch.sum((a - b).abs() * b.abs(), dim=-1)

def output_importance_weighted_similarity(a, b):
    """
    逐行计算重要性加权的相似度
    结合点积和权重重要性
    公式：sum(a * b) * ||b||，即点积乘以b的L2范数
    """
    dot_per_row = torch.sum(a * b, dim=-1)  # shape: (num_rows,)
    b_norm_per_row = torch.norm(b, p=2, dim=-1)  # shape: (num_rows,)
    return dot_per_row * b_norm_per_row

def select_rows_by_max_similarity(sim_list, w_list):
    """
    根据每行的最大相似度值，从多个权重矩阵中选择对应的行
    
    参数:
        sim_list: 相似度向量列表，每个元素shape为 (num_rows,)
        w_list: 权重矩阵列表，每个元素shape为 (num_rows, q_group_size)
    
    返回:
        selected_w: 根据最大相似度选择的权重矩阵，shape为 (num_rows, q_group_size)
    """
    # 将相似度值堆叠，找到每行相似度最大的索引
    sim_stack = torch.stack(sim_list, dim=0)  # shape: (num_sims, num_rows)
    max_idx = sim_stack.argmax(dim=0)  # shape: (num_rows,)，值为 0, 1, 2, ...
    
    # 根据索引选择对应的 w
    w_stack = torch.stack(w_list, dim=0)  # shape: (num_weights, num_rows, q_group_size)
    num_rows = w_list[0].shape[0]
    row_indices = torch.arange(num_rows, device=w_list[0].device)  # shape: (num_rows,)
    selected_w = w_stack[max_idx, row_indices]  # shape: (num_rows, q_group_size)
    
    return selected_w

def apply_scale_to_weights_dict(weights_dict, scales_list):
    """
    直接对权重字典应用 scales
    
    参数:
        weights_dict: dict, 层名 -> 权重张量的字典（如 gptq_weights）
        scales_list: list, AWQ scales 列表，格式为 [(prev_op_name, layer_names, scales, th_t), ...]
    
    返回:
        无（直接修改 weights_dict）
    
    注意：
        根据 scale_fc_fc 的逻辑：fc2.weight.mul_(scales.view(1, -1))
        scales.view(1, -1) 的形状是 (1, out_features)
        当与 (out_features, in_features) 的权重相乘时，会广播到每一行
        所以应该使用 scales.view(1, -1) 或 scales.view(-1, 1)，取决于 scales 的形状
    """
    for prev_op_name, layer_names, scales, th_t in scales_list:
        if scales is None:
            continue
        
        # 注意：scale_fc_fc 会同时修改前一层（fc1）和当前层（fc2）
        # - fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))  # 前一层后几行除以 scales
        # - fc2.weight.mul_(scales.view(1, -1))  # 当前层乘以 scales
        
        # 如果 prev_op_name 也在 weights_dict 中，也需要处理前一层
        # 这通常发生在 prev_op 是 Linear 层的情况（scale_fc_fc）
        if prev_op_name in weights_dict:
            prev_weight = weights_dict[prev_op_name]
            if len(scales.shape) == 1:
                # 根据 scale_fc_fc：fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
                # 前一层权重的后 scales.size(0) 行需要除以 scales
                if prev_weight.shape[0] >= scales.shape[0]:
                    # 只对后 scales.size(0) 行进行除法操作
                    prev_weight[-scales.shape[0]:, :] = prev_weight[-scales.shape[0]:, :] / scales.view(-1, 1)
                    weights_dict[prev_op_name] = prev_weight
        
        # 对于每个目标层（fc2），应用 scales
        for layer_name in layer_names:
            if layer_name in weights_dict:
                weight = weights_dict[layer_name]
                
                # 根据 scale_fc_fc 的逻辑：fc2.weight.mul_(scales.view(1, -1))
                # scales.view(1, -1) 的形状是 (1, out_features)
                # 当与 (out_features, in_features) 的权重相乘时，会广播到每一行
                # 每一行（第 i 行）会乘以 scales[i]
                if len(scales.shape) == 1:
                    # 检查 scales 是否对应 out_features（行数）
                    if weight.shape[0] == scales.shape[0]:
                        # scales 对应 out_features，使用 view(1, -1) 来精确匹配 scale_fc_fc 的逻辑
                        # 这会对每一行（out_features）乘以对应的 scale
                        weights_dict[layer_name] = weight * scales.view(1, -1)
                    elif weight.shape[1] == scales.shape[0]:
                        # scales 对应 in_features（列数），对每一列乘以对应的 scale
                        weights_dict[layer_name] = weight * scales.view(-1, 1)
                    else:
                        print(f"Warning: scales shape {scales.shape} doesn't match weight shape {weight.shape} for {layer_name}")
                        print(f"  prev_op: {prev_op_name}, layer: {layer_name}")
                        print(f"  weight shape: {weight.shape}, scales shape: {scales.shape}")
                else:
                    print(f"Warning: scales shape {scales.shape} is not 1D for {layer_name}")

def check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point=True, q_group_size=-1):
    """
    检查 gptq_w 是否属于当前 n_bit 的量化范畴
    
    参数:
        gptq_w: GPTQ权重矩阵，shape应与w相同
        w: 原始权重矩阵，用于计算scales和zeros
        n_bit: 量化位数
        zero_point: 是否使用zero_point量化
        q_group_size: 量化组大小，-1表示不使用分组
    
    返回:
        dict: 包含以下键值
            - is_compatible: bool, 是否完全兼容
            - out_of_range_ratio: float, 超出范围的比例
            - min_q_int: int, 量化后的最小整数值
            - max_q_int: int, 量化后的最大整数值
            - expected_min: int, 期望的最小值
            - expected_max: int, 期望的最大值
    """
    #print('start check_gptq_w_bit_compatibility')
    org_w_shape = w.shape
    assert gptq_w.shape == org_w_shape, f"gptq_w shape {gptq_w.shape} != w shape {w.shape}"
    
    # 如果需要分组，先reshape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w_reshaped = w.reshape(-1, q_group_size)
        gptq_w_reshaped = gptq_w.reshape(-1, q_group_size)
    else:
        w_reshaped = w.reshape(-1, w.shape[-1])
        gptq_w_reshaped = gptq_w.reshape(-1, gptq_w.shape[-1])
    
    assert w_reshaped.dim() == 2
    
    # 计算scales和zeros
    if zero_point:
        max_val = w_reshaped.amax(dim=1, keepdim=True)
        min_val = w_reshaped.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = w_reshaped.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0
    
    # 使用scales和zeros来量化gptq_w
    quantized_float_gptq = gptq_w_reshaped / scales
    quantized_int_gptq = torch.round(quantized_float_gptq) - zeros
    
    # 检查是否在范围内
    in_range = (quantized_int_gptq >= min_int) & (quantized_int_gptq <= max_int)
    all_in_range = in_range.all()
    
    # 计算超出范围的比例
    out_of_range_ratio = (~in_range).float().mean().item()
    min_q_int = quantized_int_gptq.min().item()
    max_q_int = quantized_int_gptq.max().item()
    
    result = {
        'is_compatible': all_in_range,
        'out_of_range_ratio': out_of_range_ratio,
        'min_q_int': min_q_int,
        'max_q_int': max_q_int,
        'expected_min': min_int,
        'expected_max': max_int,
    }
    
    return result

def output_l2_norm_similarity(a, b):
    """
    每行做差取绝对值然后求和
    """
    return -torch.norm(a - b, p='fro', dim=-1)

def output_dot_div_mse_similarity(a, b):
    """
    返回每一行的 dot 除以 mse
    dot = sum(a * b)
    mse = sum(|a - b|)
    """
    dot = torch.sum(a * b, dim=-1)  # shape: (num_rows,)
    mse = torch.sum((a - b).abs(), dim=-1)  # shape: (num_rows,)
    # 避免除以0，添加一个小的epsilon
    epsilon = 1e-8
    return dot / (mse + epsilon)

def weighted_cosine_similarity(a, b):
    """
    计算归一化的加权点积相似度 (Normalized Weighted Dot Product Similarity)
    
    数学公式:
        weighted_dot_similarity(a, b) = Σᵢ (aᵢ × bᵢ × |bᵢ|) / sqrt(||a|| × ||b||)
    
    其中:
        - a * b: 逐元素乘积 (标准点积的元素)
        - b.abs(): b的绝对值, 作为权重因子
        - ||a|| 和 ||b||: a和b在最后一个维度上的L2范数
        - 对最后一个维度求和后除以norm乘积的开方进行归一化
    
    特点:
        - 相比标准点积 Σ(aᵢ * bᵢ), 这里用 |bᵢ| 作为权重
        - |bᵢ| 越大, 该维度对相似度的贡献越大
        - 类似cosine_similarity, 通过除以norm乘积的开方进行归一化
        - 用于量化过程中的相似度比较
    """
    weighted_dot = torch.sum(a * b * b.abs(), dim=-1)
    a_norm = torch.norm(a, dim=-1)
    b_norm = torch.norm(b, dim=-1)
    eps = 1e-8
    return weighted_dot / torch.sqrt(a_norm * b_norm + eps)

def weighted_dot_similarity(a, b):
    """
    计算归一化的加权点积相似度 (Normalized Weighted Dot Product Similarity)
    
    数学公式:
        weighted_dot_similarity(a, b) = Σᵢ (aᵢ × bᵢ × |bᵢ|)
    
    其中:
        - a * b: 逐元素乘积 (标准点积的元素)
        - b.abs(): b的绝对值, 作为权重因子
        - ||a|| 和 ||b||: a和b在最后一个维度上的L2范数
        - 对最后一个维度求和后除以norm乘积的开方进行归一化
    
    特点:
        - 相比标准点积 Σ(aᵢ * bᵢ), 这里用 |bᵢ| 作为权重
        - |bᵢ| 越大, 该维度对相似度的贡献越大
        - 类似cosine_similarity, 通过除以norm乘积的开方进行归一化
        - 用于量化过程中的相似度比较
    """
    weighted_dot = torch.sum(a * b * b.abs(), dim=-1)
    
    return weighted_dot #/ torch.sqrt(a_norm * b_norm + eps)


def output_e_distance(a, b):
    a= a.sum(dim=-1)
    b= b.sum(dim=-1)
    a = torch.exp(a)
    b = torch.exp(b)
    return torch.abs(a-b)

def output_e_distance_abs(a, b):
    a = torch.abs(a)
    b = torch.abs(b)
    a= a.sum(dim=-1)
    b= b.sum(dim=-1)
    a = torch.exp(a)
    b = torch.exp(b)
    return torch.abs(a-b)
def output_e_distance_square(a, b):
    a = a.pow(2)
    b = b.pow(2)
    a= a.sum(dim=-1)
    b= b.sum(dim=-1)
    a = torch.exp(a)
    b = torch.exp(b)
    return torch.abs(a-b)

# def pseudo_quantize_tensor722(
#     w, W_int, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
# ):
#     org_w_shape = w.shape
#     if q_group_size > 0:
#         assert org_w_shape[-1] % q_group_size == 0
#         w = w.reshape(-1, q_group_size)
#         current_w_shape = w.shape
#     assert w.dim() == 2
#     if zero_point:
#         max_val = w.amax(dim=1, keepdim=True)
#         min_val = w.amin(dim=1, keepdim=True)
#         max_int = 2**n_bit - 1
#         min_int = 0
#         scales = (max_val - min_val).clamp(min=1e-5) / max_int
#         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
#     else:  # we actually never used this
#         assert min_val is None
#         max_val = w.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)
#         max_int = 2 ** (n_bit - 1) - 1
#         min_int = -(2 ** (n_bit - 1))
#         scales = max_val / max_int
#         zeros = 0

#     assert torch.isnan(scales).sum() == 0
#     assert torch.isnan(w).sum() == 0

#     # 计算量化前的浮点值
#     quantized_float = w / scales
    
#     # 1. 向下取整做差得到小数部分
#     floor_values = torch.floor(quantized_float)
#     fractional_parts = quantized_float - floor_values
    
#     # 2. 计算小数部分每个channel的平均值
#     # 对于每一行(channel)计算小数部分的平均值作为阈值
#     # channel_thresholds = torch.mean(fractional_parts, dim=1, keepdim=True)
#     # channel_thresholds,_ = torch.median(fractional_parts, dim=1, keepdim=True)
#     # channel_thresholds = top_sum_threhold(fractional_parts)
    

#     # channel_thresholds = torch.max(channel_thresholds, torch.ones_like(channel_thresholds) * 0.5)
    
#     # 3. 使用自适应阈值进行四舍五入操作
#     # 如果小数部分大于该channel的平均值，则向上取整，否则向下取整
#     # t = 0.6
#     # channel_thresholds = t*channel_thresholds + (1-t)*torch.ones_like(channel_thresholds) * 0.5

#     #channel_thresholds = top_sum_threhold_balance_value(fractional_parts,t = th_t)
#     #new_fractional_parts = fractional_parts - (channel_thresholds-0.5)
#     #quantized_int = torch.round(new_fractional_parts+floor_values)
    

#     # quantized_int_entropy, _,_ = count_and_entropy(quantized_int)
#     # original_quantized_int_entropy, _,_ = count_and_entropy(original_quantized_int)

#     # original_entropy,_,_ = count_and_entropy(quantized_float)
    
#     # print(f"entropy: {quantized_int_entropy}", f'original_int: {original_quantized_int_entropy}')
#     # if quantized_int_entropy > original_quantized_int_entropy:
#     #     print('use original quantized int')
#     #     quantized_int = original_quantized_int

#     # quantized_int_entropy = row_entropy_vectorized(quantized_int)
#     # original_quantized_int_entropy = row_entropy_vectorized(original_quantized_int)

#     print(f"search_flag: {search_flag}")
#     if search_flag == True:
#         quantized_int = torch.round(quantized_float)
#     else:
        
#         quantized_int = top_sum_number(fractional_parts) + floor_values 
#         original_quantized_int = torch.round(quantized_float)

#         # print('quantized_int',quantized_int)
#         # print('original_quantized_int',original_quantized_int)
#         # print('gptq', W_int)
#         # W_int = W_int.reshape(-1, q_group_size)
#         # candidates = [quantized_int, original_quantized_int, W_int]
        
#         # quantized_int_norm = torch.norm(quantized_int, p=2, dim=1)
#         # original_quantized_int_norm = torch.norm(original_quantized_int, p=2, dim=1)

#         # mask = (quantized_int_norm > original_quantized_int_norm).unsqueeze(1)  # [N, 1]
        
#         # quantized_int_std = torch.std(quantized_int, dim=1)
#         # original_quantized_int_std = torch.std(original_quantized_int, dim=1)

#         # mask = (quantized_int_std > original_quantized_int_std).unsqueeze(1)  # [N, 1]
#         """  dot similarity  """
#         # print('real dot similarity')
#         # quantized_int_cos = output_dot_similarity(quantized_int,quantized_float)
#         # original_quantized_int_cos = output_dot_similarity(original_quantized_int,quantized_float)
#         # mask = (quantized_int_cos > original_quantized_int_cos).unsqueeze(1)  # [N, 1]

     
#         """  wdot similarity  """
#         # 1111111111111111111111111
#         # print('real dot similarity')

#         # cos_list = []
#         # for i in candidates:
#         #     print(i.shape)
#         #     i = i.reshape(-1, q_group_size)
#         #     k_cos = weighted_dot_similarity(i, quantized_float)
#         #     cos_list.append(k_cos)

#         # # print(cos_list)

#         # stacked = torch.stack(cos_list, dim=0)      # [K, N]
#         # idx = stacked.argmax(dim=0)                 # [N]  每个位置的赢家编号 0..K-1

#         # # 统计各个候选被选中的个数
#         # total = idx.numel()
#         # count_0 = (idx == 0).sum().item()
#         # count_1 = (idx == 1).sum().item()
#         # count_2 = (idx == 2).sum().item()

#         # # print(f"total: {total}, count_0: {count_0}, count_1: {count_1}, count_2: {count_2}")

#         # idx = idx.unsqueeze(1)

#         # idx1d = idx.squeeze(-1).long()  # 确保是 long

#         # # candidates: [A, B, C] 每个都是 [N, M]
#         # stacked = torch.stack(candidates, dim=0)  # [K=3, N, M]

#         # N = stacked.shape[1]
#         # rows = torch.arange(N, device=stacked.device)
#         # quantized_int = stacked[idx1d, rows, :]
#         # quantized_int = W_int
#         # 1111111111111111111111111


#         # q_int: [N, M]，第 r 行来自 candidates[idx[r]]
        
#         # print('final quantized int',quantized_int)
        
        

#         # quantized_int_cos = weighted_dot_similarity(quantized_int,quantized_float)
#         # original_quantized_int_cos = weighted_dot_similarity(original_quantized_int,quantized_float)
        
#         # mask = (quantized_int_cos > original_quantized_int_cos).unsqueeze(1)  # [N, 1]
#         # print('mask',mask.shape)

#         # assert 0


#         """  cosine similarity  """
#         # print('real cosine similarity')
#         # quantized_int_cos = output_cosine_similarity(quantized_int,quantized_float)
#         # original_quantized_int_cos = output_cosine_similarity(original_quantized_int,quantized_float)
#         # mask = (quantized_int_cos > original_quantized_int_cos).unsqueeze(1)  # [N, 1]

#         # """  使用加权余弦相似度  """
#         # print('real weighted cosine similarity')
#         # quantized_int_cos = weighted_cosine_similarity(quantized_int,quantized_float)
#         # original_quantized_int_cos = weighted_cosine_similarity(original_quantized_int,quantized_float)
#         # mask = (quantized_int_cos > original_quantized_int_cos).unsqueeze(1)  # [N, 1]


#         #quantized_int_e_distance = output_e_distance(quantized_int,quantized_float)
#         #original_quantized_int_e_distance = output_e_distance(original_quantized_int,quantized_float)

#         #mask = (quantized_int_e_distance < original_quantized_int_e_distance).unsqueeze(1)  # [N, 1]

#         # 111111111111111
#         # quantized_int = torch.where(mask, quantized_int, original_quantized_int)
#         # 111111111111111

#         # quantized_int_entropy, _,_ = count_and_entropy(quantized_int)
#         # quantized_int_entropy, _,_ = count_and_entropy(quantized_int)
#         # original_quantized_int_entropy, _,_ = count_and_entropy(original_quantized_int)

        
        
#         # print(f"entropy: {quantized_int_entropy}", f'original_int: {original_quantized_int_entropy}')
#         # if quantized_int_entropy < original_quantized_int_entropy:
#         #     print('use original quantized int')
#         #     quantized_int = original_quantized_int


#     # print(f"quantized_int_entropy: {quantized_int_entropy.shape}", f'original_int: {original_quantized_int_entropy.shape}')
#     # print(f"quantized_int: {quantized_int.shape}", f'original_int: {original_quantized_int.shape}')
    

#     # 先得到每行是否选 quantized_int（True）还是 original_quantized_int（False）
#     # mask = (quantized_int_entropy < original_quantized_int_entropy).unsqueeze(1)  # [N, 1]

#     # # 按行选择
#     # #result = torch.where(mask, quantized_int, original_quantized_int)  # [N, D]
#     # quantized_int = torch.where(mask, quantized_int, original_quantized_int)



        
#     # quantized_int = torch.where(
#     #     fractional_parts > channel_thresholds,
#     #     torch.ceil(quantized_float),
#     #     torch.floor(quantized_float)
#     # )

#     #if apply to zeros，如果不希望apply，把下面三行注释掉即可
#     #print('apply to zeros')
    
#     tmp_quantized_int = quantized_int.reshape(current_w_shape)
#     min_val = tmp_quantized_int.amin(dim=1, keepdim=True)
#     zeros = (-min_val).clamp_(min_int, max_int)
#     w1 = (torch.clamp(quantized_int + zeros, min_int, max_int) - zeros) * scales


#     tmp_quantized_int = original_quantized_int.reshape(current_w_shape)
#     min_val = tmp_quantized_int.amin(dim=1, keepdim=True)
#     zeros = (-min_val).clamp_(min_int, max_int)
#     w2 = (torch.clamp(original_quantized_int + zeros, min_int, max_int) - zeros) * scales

#     w3 = W_int


#     # zeros = min_val.clamp_(min_int, max_int)

#     # if inplace:
#     #     assert 0
#     #     ((w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
#     # else:
#     #     w = (torch.clamp(quantized_int + zeros, min_int, max_int) - zeros) * scales
    

#     # if inplace:
#     #     (
#     #         (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
#     #     ).mul_(scales)
#     # else:
#     #     w = (
#     #         torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
#     #     ) * scales


#     assert torch.isnan(w).sum() == 0

#     w1 = w1.reshape(org_w_shape)
#     w2 = w2.reshape(org_w_shape)
#     w3 = w3.reshape(org_w_shape)

#     w = w.reshape(org_w_shape)


#     candidates = [w1, w2, w3]
#     # candidates = [w1, w2]

#     # candidates = [w1, w3]
#     # candidates = [w2, w3]
#     cos_list = []
#     for i in candidates:
#         print(i.shape)
#         # k_cos = weighted_dot_similarity(i, w)
#         # k_cos = weighted_cosine_similarity(i, w)
#         k_cos = output_dot_similarity(i, w)
#         cos_list.append(k_cos)

#     # print(cos_list)

#     stacked = torch.stack(cos_list, dim=0)      # [K, N]
#     idx = stacked.argmax(dim=0)                 # [N]  每个位置的赢家编号 0..K-1

#     # 统计各个候选被选中的个数
#     # total = idx.numel()
#     # count_0 = (idx == 0).sum().item()
#     # count_1 = (idx == 1).sum().item()
#     # count_2 = (idx == 2).sum().item()

#     # print(f"total: {total}, count_0: {count_0}, count_1: {count_1}, count_2: {count_2}")

#     idx = idx.unsqueeze(1)

#     idx1d = idx.squeeze(-1).long()  # 确保是 long

#     # candidates: [A, B, C] 每个都是 [N, M]
#     stacked = torch.stack(candidates, dim=0)  # [K=3, N, M]

#     N = stacked.shape[1]
#     rows = torch.arange(N, device=stacked.device)
#     w = stacked[idx1d, rows, :]
    
#     # print(W_int.shape)

#     # gptq基线
#     # w = W_int.reshape(org_w_shape)
#     # print('w', w)
#     # w = W_int
#     # print('w', w)
    
#     return w
    
#     # print(w.shape)

#     # if get_scale_zp:
#     #     print('get scale zp')
#     #     return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
#     # else:
#     #     return w
def pseudo_quantize_tensor722(
    w, gptq_w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
):
    org_w_shape = w.shape
    check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point, q_group_size)
    assert gptq_w.shape == org_w_shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        gptq_w = gptq_w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    quantized_float = w / scales
    floor_values = torch.floor(quantized_float)
    fractional_parts = quantized_float - floor_values
    def back_quantize(quantized_int,scales,zeros,min_int,max_int):
        
        w = (
                torch.clamp(quantized_int + zeros, min_int, max_int) - zeros
            ) * scales
        assert torch.isnan(w).sum() == 0
        return w
    

    #print(f"search_flag: {search_flag}")
    if search_flag == True:
        quantized_int = torch.round(quantized_float)
        original_quantized_int = quantized_int
        median_quantized_int = quantized_int
        w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        w2 = w1
        w3 = w1
        base_w = w1
    else:
        
        top_sum_quantized_int = top_sum_number(fractional_parts) + floor_values 
        original_quantized_int = torch.round(quantized_float)
        base_quantized_int = torch.floor(quantized_float)
        median_quantized_int = median_threshold_number(fractional_parts) + floor_values
        tmp_sim1 = output_dot_similarity(original_quantized_int, quantized_float)
        tmp_sim2 = output_dot_similarity(top_sum_quantized_int, quantized_float)
        combined_quantized_int = select_rows_by_max_similarity([tmp_sim1, tmp_sim2], [original_quantized_int, top_sum_quantized_int])
        w1 = gptq_w
        
        w2 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        base_w = back_quantize(base_quantized_int,scales,zeros,min_int,max_int)
        #w3 = back_quantize(combined_quantized_int,scales,zeros,min_int,max_int)
        w3 = back_quantize(top_sum_quantized_int,scales,zeros,min_int,max_int)
        #w3 = gptq_w #back_quantize(median_quantized_int,scales,zeros,min_int,max_int)
    #w1 = w3
    #w1=w2
    w3=w2
    # w3=w1
    # w2=w1
    # 保存原始的 w（在 reshape 之前）用于相似度计算
    w_original = w.clone()
    
    # 确保所有张量的数据类型一致，避免类型不匹配错误
    w1 = w1.to(w_original.dtype)
    w2 = w2.to(w_original.dtype)
    w3 = w3.to(w_original.dtype)
    base_w = base_w.to(w_original.dtype)
    w1 = w1.reshape(org_w_shape)
    w2 = w2.reshape(org_w_shape)
    w3 = w3.reshape(org_w_shape)
    base_w = base_w.reshape(org_w_shape)
    w_original = w_original.reshape(org_w_shape)
    
    # 计算 w1, w2, w3 分别与 w_original 的逐行相似度
    # 建议尝试的方法（按推荐顺序）：
    # 1. output_importance_weighted_similarity - 结合点积和权重重要性，通常效果最好
    # 2. output_weighted_dot_similarity - 加权点积，大权重影响更大
    # 3. output_normalized_dot_similarity - 归一化点积，数值稳定
    # 4. output_dot_similarity - 简单点积，最稳定
    # 5. output_weighted_l2_similarity - 加权L2距离
    # 6. output_cosine_similarity - 余弦相似度
    # 7. output_rel_err_similarity - 相对误差（可能不稳定）
    
    # 当前使用的方法（可以切换尝试）
    # sim1 = output_importance_weighted_similarity(w1, w_original)  # shape: (num_rows,)
    # sim2 = output_importance_weighted_similarity(w2, w_original)  # shape: (num_rows,)
    # sim3 = output_importance_weighted_similarity(w3, w_original)  # shape: (num_rows,)
    
    # 备选方法（取消注释以尝试）：
    # sim1 = output_weighted_dot_similarity(w1, w_original)
    # sim2 = output_weighted_dot_similarity(w2, w_original)
    # sim3 = output_weighted_dot_similarity(w3, w_original)
    
    # sim1 = output_normalized_dot_similarity(w1, w_original)
    # sim2 = output_normalized_dot_similarity(w2, w_original)
    # sim3 = output_normalized_dot_similarity(w3, w_original)
    
    sim1 = output_mse_similarity(w1, w_original)
    sim2 = output_mse_similarity(w2, w_original)
    sim3 = output_mse_similarity(w3, w_original)
    # sim1 = output_dot_similarity(w1-base_w, w_original-base_w)
    # sim2 = output_dot_similarity(w2-base_w, w_original-base_w)
    # sim3 = output_dot_similarity(w3-base_w, w_original-base_w)
    #print(f"sim1: {sim1.mean().item():.6f}, sim2: {sim2.mean().item():.6f}, sim3: {sim3.mean().item():.6f}")
    # 根据每行的最大相似度值选择对应的权重矩阵行
    #f_new_w = select_rows_by_max_similarity([sim1, sim2, sim3], [w1, w2, w3])
    
    # 将三个相似度值堆叠，用于后续统计
    sim_stack = torch.stack([sim1, sim2, sim3], dim=0)  # shape: (3, num_rows)
    
    # 新分支：计算矩阵整体的内积
    # dot1 = torch.sum(w1 * w_original)  # w1 与 w_original 的整体内积
    # dot2 = torch.sum(w2 * w_original)  # w2 与 w_original 的整体内积
    # dot3 = torch.sum(w3 * w_original)  # w3 与 w_original 的整体内积
    
    # 方法1: 加权Frobenius范数（考虑权重的重要性）
    # 使用 w_original 的绝对值作为权重，大权重的影响更大
    # weight_mask = w_original.abs()
    # weighted_fro1 = -torch.norm((w1 - w_original) * weight_mask, p='fro')
    # weighted_fro2 = -torch.norm((w2 - w_original) * weight_mask, p='fro')
    # weighted_fro3 = -torch.norm((w3 - w_original) * weight_mask, p='fro')
    
    # # 方法2: 相对误差（百分比误差）
    # # 对于大权重，绝对误差可能很大但相对误差小；对于小权重则相反
    # rel_err1 = -torch.mean(((w1 - w_original) / (w_original.abs() + 1e-8)).abs())
    # rel_err2 = -torch.mean(((w2 - w_original) / (w_original.abs() + 1e-8)).abs())
    # rel_err3 = -torch.mean(((w3 - w_original) / (w_original.abs() + 1e-8)).abs())
    
    # # # 方法3: 逐行的加权点积（考虑每行的重要性）
    # # # 使用 w_original 每行的L2范数作为权重
    # row_weights = torch.norm(w_original, p=2, dim=-1)  # shape: (num_rows,)
    # row_dot1 = torch.sum((w1 * w_original).sum(dim=-1) * row_weights)
    # row_dot2 = torch.sum((w2 * w_original).sum(dim=-1) * row_weights)
    # row_dot3 = torch.sum((w3 * w_original).sum(dim=-1) * row_weights)
    
    # # # 方法4: 组合多个指标（加权平均）
    # # # 归一化各个指标到 [0, 1] 范围，然后组合
    # scores1 = torch.stack([weighted_fro1, rel_err1, row_dot1])
    # scores2 = torch.stack([weighted_fro2, rel_err2, row_dot2])
    # scores3 = torch.stack([weighted_fro3, rel_err3, row_dot3])
    
    # # # # 归一化（可选：使用softmax或min-max归一化）
    # # # all_scores = torch.stack([scores1, scores2, scores3], dim=0)  # (3, 3)
    # # # # 使用简单的加权平均，可以根据效果调整权重
    # weights = torch.tensor([0.0, 0.0, 1.0], device=w1.device)  # 可调整的权重
    # combined_score1 = (scores1 * weights).sum()
    # combined_score2 = (scores2 * weights).sum()
    # combined_score3 = (scores3 * weights).sum()
    
    # # # 根据组合分数选择

    # 统计每行所有等于最大值的索引（如果某行有多个相同的最大值，都统计）
    max_values = sim_stack.max(dim=0)[0]  # shape: (num_rows,)，每行的最大值
    is_max_0 = (sim_stack[0] == max_values).float()  # shape: (num_rows,)，sim1是否等于最大值
    is_max_1 = (sim_stack[1] == max_values).float()  # shape: (num_rows,)，sim2是否等于最大值
    is_max_2 = (sim_stack[2] == max_values).float()  # shape: (num_rows,)，sim3是否等于最大值
    
    # 计算每行的重要性权重（使用 w_original 每行的L2范数）
    row_weights = torch.norm(w_original, p=2, dim=-1)  # shape: (num_rows,)
    
    row_weights = torch.ones_like(row_weights)
    # 使用行重要性权重进行加权统计
    count_0 = (is_max_0 * row_weights).sum().float()  # 加权统计sim1等于最大值的行
    count_1 = (is_max_1 * row_weights).sum().float()  # 加权统计sim2等于最大值的行
    count_2 = (is_max_2 * row_weights).sum().float()  # 加权统计sim3等于最大值的行
    dot_values = torch.stack([count_0, count_1, count_2])  # shape: (3,)
    #print(f"dot_values (counts in max_idx): [{count_0.item():.0f}, {count_1.item():.0f}, {count_2.item():.0f}], max_idx={dot_values.argmax().item()}")
  
    max_dot_idx = dot_values.argmax().item()  # 标量值：0, 1, 或 2
    
    # 更新全局统计
    max_dot_idx_stats['total'] += 1
    if max_dot_idx == 0:
        max_dot_idx_stats['count_0'] += 1
        new_w = w1
        #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    elif max_dot_idx == 1:
        max_dot_idx_stats['count_1'] += 1
        #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=1")
        new_w = w2
        #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    else:
        max_dot_idx_stats['count_2'] += 1
        #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=0")
        #print(f"dot_values (counts in max_idx): [{count_0.item():.0f}, {count_1.item():.0f}, {count_2.item():.0f}], max_idx={dot_values.argmax().item()}")
        #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
        new_w = w3
    
    #new_w = gptq_w.clone()
    #new_w = w1
    #new_w = f_new_w
    w = new_w.reshape(org_w_shape)
    

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w



def pseudo_quantize_tensor722_backup0116(
    w, gptq_w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
):
    org_w_shape = w.shape
    check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point, q_group_size)
    assert gptq_w.shape == org_w_shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        gptq_w = gptq_w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    quantized_float = w / scales
    floor_values = torch.floor(quantized_float)
    fractional_parts = quantized_float - floor_values
    def back_quantize(quantized_int,scales,zeros,min_int,max_int):
        
        w = (
                torch.clamp(quantized_int + zeros, min_int, max_int) - zeros
            ) * scales
        assert torch.isnan(w).sum() == 0
        return w
    

    #print(f"search_flag: {search_flag}")
    if search_flag == True:
        quantized_int = torch.round(quantized_float)
        original_quantized_int = quantized_int
        median_quantized_int = quantized_int
        w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        w2 = w1
        w3 = w1
        base_w = w1
    else:
        
        quantized_int = top_sum_number(fractional_parts) + floor_values 
        original_quantized_int = torch.round(quantized_float)
        base_quantized_int = torch.floor(quantized_float)
        median_quantized_int = median_threshold_number(fractional_parts) + floor_values

        w1 = gptq_w
        w3 = back_quantize(quantized_int,scales,zeros,min_int,max_int)
        w2 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        base_w = back_quantize(base_quantized_int,scales,zeros,min_int,max_int)
        #w3 = gptq_w #back_quantize(median_quantized_int,scales,zeros,min_int,max_int)
    #w1 = w3
    # w1=w2
    # w3=w2
    # w3=w1
    # w2=w1
    # 保存原始的 w（在 reshape 之前）用于相似度计算
    w_original = w.clone()
    
    # 确保所有张量的数据类型一致，避免类型不匹配错误
    w1 = w1.to(w_original.dtype)
    w2 = w2.to(w_original.dtype)
    w3 = w3.to(w_original.dtype)
    base_w = base_w.to(w_original.dtype)
    w1 = w1.reshape(org_w_shape)
    w2 = w2.reshape(org_w_shape)
    w3 = w3.reshape(org_w_shape)
    base_w = base_w.reshape(org_w_shape)
    w_original = w_original.reshape(org_w_shape)
    
    # 计算 w1, w2, w3 分别与 w_original 的逐行相似度
    # 建议尝试的方法（按推荐顺序）：
    # 1. output_importance_weighted_similarity - 结合点积和权重重要性，通常效果最好
    # 2. output_weighted_dot_similarity - 加权点积，大权重影响更大
    # 3. output_normalized_dot_similarity - 归一化点积，数值稳定
    # 4. output_dot_similarity - 简单点积，最稳定
    # 5. output_weighted_l2_similarity - 加权L2距离
    # 6. output_cosine_similarity - 余弦相似度
    # 7. output_rel_err_similarity - 相对误差（可能不稳定）
    
    # 当前使用的方法（可以切换尝试）
    # sim1 = output_importance_weighted_similarity(w1, w_original)  # shape: (num_rows,)
    # sim2 = output_importance_weighted_similarity(w2, w_original)  # shape: (num_rows,)
    # sim3 = output_importance_weighted_similarity(w3, w_original)  # shape: (num_rows,)
    
    # 备选方法（取消注释以尝试）：
    # sim1 = output_weighted_dot_similarity(w1, w_original)
    # sim2 = output_weighted_dot_similarity(w2, w_original)
    # sim3 = output_weighted_dot_similarity(w3, w_original)
    
    # sim1 = output_normalized_dot_similarity(w1, w_original)
    # sim2 = output_normalized_dot_similarity(w2, w_original)
    # sim3 = output_normalized_dot_similarity(w3, w_original)
    
    sim1 = output_dot_similarity(w1, w_original)
    sim2 = output_dot_similarity(w2, w_original)
    sim3 = output_dot_similarity(w3, w_original)
    # sim1 = output_dot_similarity(w1-base_w, w_original-base_w)
    # sim2 = output_dot_similarity(w2-base_w, w_original-base_w)
    # sim3 = output_dot_similarity(w3-base_w, w_original-base_w)
    #print(f"sim1: {sim1.mean().item():.6f}, sim2: {sim2.mean().item():.6f}, sim3: {sim3.mean().item():.6f}")
    # 根据每行的最大相似度值选择对应的权重矩阵行
    f_new_w = select_rows_by_max_similarity([sim1, sim2, sim3], [w1, w2, w3])
    
    # 将三个相似度值堆叠，用于后续统计
    sim_stack = torch.stack([sim1, sim2, sim3], dim=0)  # shape: (3, num_rows)
    
    # 新分支：计算矩阵整体的内积
    # dot1 = torch.sum(w1 * w_original)  # w1 与 w_original 的整体内积
    # dot2 = torch.sum(w2 * w_original)  # w2 与 w_original 的整体内积
    # dot3 = torch.sum(w3 * w_original)  # w3 与 w_original 的整体内积
    
    # 方法1: 加权Frobenius范数（考虑权重的重要性）
    # 使用 w_original 的绝对值作为权重，大权重的影响更大
    # weight_mask = w_original.abs()
    # weighted_fro1 = -torch.norm((w1 - w_original) * weight_mask, p='fro')
    # weighted_fro2 = -torch.norm((w2 - w_original) * weight_mask, p='fro')
    # weighted_fro3 = -torch.norm((w3 - w_original) * weight_mask, p='fro')
    
    # # 方法2: 相对误差（百分比误差）
    # # 对于大权重，绝对误差可能很大但相对误差小；对于小权重则相反
    # rel_err1 = -torch.mean(((w1 - w_original) / (w_original.abs() + 1e-8)).abs())
    # rel_err2 = -torch.mean(((w2 - w_original) / (w_original.abs() + 1e-8)).abs())
    # rel_err3 = -torch.mean(((w3 - w_original) / (w_original.abs() + 1e-8)).abs())
    
    # # # 方法3: 逐行的加权点积（考虑每行的重要性）
    # # # 使用 w_original 每行的L2范数作为权重
    # row_weights = torch.norm(w_original, p=2, dim=-1)  # shape: (num_rows,)
    # row_dot1 = torch.sum((w1 * w_original).sum(dim=-1) * row_weights)
    # row_dot2 = torch.sum((w2 * w_original).sum(dim=-1) * row_weights)
    # row_dot3 = torch.sum((w3 * w_original).sum(dim=-1) * row_weights)
    
    # # # 方法4: 组合多个指标（加权平均）
    # # # 归一化各个指标到 [0, 1] 范围，然后组合
    # scores1 = torch.stack([weighted_fro1, rel_err1, row_dot1])
    # scores2 = torch.stack([weighted_fro2, rel_err2, row_dot2])
    # scores3 = torch.stack([weighted_fro3, rel_err3, row_dot3])
    
    # # # # 归一化（可选：使用softmax或min-max归一化）
    # # # all_scores = torch.stack([scores1, scores2, scores3], dim=0)  # (3, 3)
    # # # # 使用简单的加权平均，可以根据效果调整权重
    # weights = torch.tensor([0.0, 0.0, 1.0], device=w1.device)  # 可调整的权重
    # combined_score1 = (scores1 * weights).sum()
    # combined_score2 = (scores2 * weights).sum()
    # combined_score3 = (scores3 * weights).sum()
    
    # # # 根据组合分数选择

    # 统计每行所有等于最大值的索引（如果某行有多个相同的最大值，都统计）
    max_values = sim_stack.max(dim=0)[0]  # shape: (num_rows,)，每行的最大值
    is_max_0 = (sim_stack[0] == max_values).float()  # shape: (num_rows,)，sim1是否等于最大值
    is_max_1 = (sim_stack[1] == max_values).float()  # shape: (num_rows,)，sim2是否等于最大值
    is_max_2 = (sim_stack[2] == max_values).float()  # shape: (num_rows,)，sim3是否等于最大值
    
    # 计算每行的重要性权重（使用 w_original 每行的L2范数）
    row_weights = torch.norm(w_original, p=2, dim=-1)  # shape: (num_rows,)
    
    row_weights = torch.ones_like(row_weights)
    # 使用行重要性权重进行加权统计
    count_0 = (is_max_0 * row_weights).sum().float()  # 加权统计sim1等于最大值的行
    count_1 = (is_max_1 * row_weights).sum().float()  # 加权统计sim2等于最大值的行
    count_2 = (is_max_2 * row_weights).sum().float()  # 加权统计sim3等于最大值的行
    dot_values = torch.stack([count_0, count_1, count_2])  # shape: (3,)
    #print(f"dot_values (counts in max_idx): [{count_0.item():.0f}, {count_1.item():.0f}, {count_2.item():.0f}], max_idx={dot_values.argmax().item()}")
  
    max_dot_idx = dot_values.argmax().item()  # 标量值：0, 1, 或 2
    
    # 更新全局统计
    max_dot_idx_stats['total'] += 1
    if max_dot_idx == 0:
        max_dot_idx_stats['count_0'] += 1
        new_w = w1
        #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    elif max_dot_idx == 1:
        max_dot_idx_stats['count_1'] += 1
        #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=1")
        new_w = w2
        print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    else:
        max_dot_idx_stats['count_2'] += 1
        #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=0")
        #print(f"dot_values (counts in max_idx): [{count_0.item():.0f}, {count_1.item():.0f}, {count_2.item():.0f}], max_idx={dot_values.argmax().item()}")
        print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
        new_w = w3
    
    #new_w = gptq_w.clone()
    #new_w = w1
    #new_w = f_new_w
    w = new_w.reshape(org_w_shape)
    

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w



def pseudo_quantize_tensor722_old(
    w, W_int, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        current_w_shape = w.shape
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # 计算量化前的浮点值
    quantized_float = w / scales
    
    # 1. 向下取整做差得到小数部分
    floor_values = torch.floor(quantized_float)
    fractional_parts = quantized_float - floor_values

    print(f"search_flag: {search_flag}")
    if search_flag == True:
        quantized_int = torch.round(quantized_float)
    else:
        
        quantized_int = top_sum_number(fractional_parts) + floor_values 
        original_quantized_int = torch.round(quantized_float)
    
    # tmp_quantized_int = quantized_int.reshape(current_w_shape)
    # min_val = tmp_quantized_int.amin(dim=1, keepdim=True)
    # zeros = (-min_val).clamp_(min_int, max_int)
    w1 = (torch.clamp(quantized_int + zeros, min_int, max_int) - zeros) * scales



    # tmp_quantized_int = original_quantized_int.reshape(current_w_shape)
    # min_val = tmp_quantized_int.amin(dim=1, keepdim=True)
    # zeros = (-min_val).clamp_(min_int, max_int)
    w2 = (torch.clamp(original_quantized_int + zeros, min_int, max_int) - zeros) * scales

    w3 = W_int

    assert torch.isnan(w).sum() == 0
    w0 = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    w0 = w0.reshape(org_w_shape)
    w1 = w1.reshape(org_w_shape)
    w2 = w2.reshape(org_w_shape)
    # w3 = w3.reshape(org_w_shape)
    w = w.reshape(org_w_shape)

    # candidates = [w1, w2, w3]
    candidates = [w1, w2]

    cos_list = []
    for i in candidates:
        print(i.shape)
        # k_cos = weighted_dot_similarity(i, w)
        # k_cos = weighted_cosine_similarity(i, w)
        # k_cos = output_cosine_similarity(i, w)
        k_cos = output_dot_similarity(i, w)
        cos_list.append(k_cos)

    stacked = torch.stack(cos_list, dim=0)      # [K, N]
    idx = stacked.argmax(dim=0)                 # [N]  每个位置的赢家编号 0..K-1


    idx = idx.unsqueeze(1)

    idx1d = idx.squeeze(-1).long()  # 确保是 long

    # candidates: [A, B, C] 每个都是 [N, M]
    stacked = torch.stack(candidates, dim=0)  # [K=3, N, M]

    N = stacked.shape[1]
    rows = torch.arange(N, device=stacked.device)
    w = stacked[idx1d, rows, :]
    
    
    return w




@torch.no_grad()
def pseudo_quantize_model_weight722(
    model,
    w_bit,
    model_path,
    q_config,
    th_t_dict={},
    awq_results=None
):
    from .pre_quant import get_blocks, get_named_linears

    # 导入gptq模型权重
    # 如果 awq_results 中已经有 scaled_gptq_weights，直接使用（已经在 entry.py 中通过重新 apply_awq722 得到）
    if awq_results is not None and "scaled_gptq_weights" in awq_results:
        gptq_weights = awq_results["scaled_gptq_weights"]
        print(f"Using pre-scaled gptq_weights from awq_results")
    else:
        # 否则加载原始的 gptq_weights
        
        if 'Llama-7b' in model_path:
            gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_1.pt", map_location="cpu")
        elif 'llama-2-7b' in model_path:
            gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_2.pt", map_location="cpu")
        elif 'Meta-Llama-3-8B' in model_path:
            gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_3.pt", map_location="cpu")
        else:
            gptq_weights = {}
        
        # 如果提供了 awq_results，对 gptq_weights 也应用 scales（旧方法，保留作为后备）
        # if awq_results is not None and "scale" in awq_results:
        #     scales_list = awq_results["scale"]
        #     apply_scale_to_weights_dict(gptq_weights, scales_list)
        #     print(f"Applied scales to gptq_weights (fallback method)")
    
    layers = get_blocks(model)
    tmp_i = 0
    
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        if i < len(layers)/2:
            search_flag = False
        else:
            search_flag = False
        for n, m in named_linears.items():
            m.cuda()

            full_name = f"model.layers.{i}.{n}"

            if full_name not in gptq_weights:
                print("找不到:", full_name)
                # 这里要 continue，否则还会 KeyError
                m.cpu()
                continue

            W_int = gptq_weights[full_name].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )

            # print('pseudo_quantize_model_weight722')
            # print(n)
            m.weight.data = pseudo_quantize_tensor722(
                m.weight.data, W_int, n_bit=w_bit,th_t=1, **q_config, search_flag=search_flag
            )
            tmp_i += 1
            m.cpu()
    print(f"tmp_i: {tmp_i}")
    print_max_dot_idx_stats()

@torch.no_grad()
def pseudo_quantize_model_weight722_single_layer(
    model,
    w_bit,
    q_config,
    layer_index,
    th_t_dict={},  
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    tmp_i = 0
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        if i != layer_index:
            continue
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            print('pseudo_quantize_model_weight722_single_layer')
            m.weight.data = pseudo_quantize_tensor722(
                m.weight.data, n_bit=w_bit,th_t=th_t_dict[tmp_i], **q_config
            )
            tmp_i += 1
            m.cpu()
    print(f"tmp_i: {tmp_i}")



@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
