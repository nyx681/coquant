import os
import time
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


def output_cosine_similarity(a, b,dim=-1):
    return F.cosine_similarity(a, b, dim=dim)

def output_dot_similarity(a, b,dim=-1):
    return torch.sum(a * b, dim=dim)
def dot_sim(a, b,c):
    """
    返回元素相乘的矩阵，并使用c的绝对值作为权重
    参数:
        a, b: 输入张量
        c: 权重张量
    返回:
        a * b * c.abs(): 逐元素相乘的结果
    """
    return a * b * c.abs()

def mse_sim(a, b,c):
    """
    返回元素相乘的矩阵，并使用c的绝对值作为权重
    参数:
        a, b: 输入张量
        c: 权重张量
    返回:
        a * b * c.abs(): 逐元素相乘的结果
    """
    return -(a - b).abs() * c.abs()

def output_dot_similarity_abs(a, b,dim=-1):
    return torch.sum(a.abs() * b.abs(), dim=dim)
def output_mse_similarity(a, b,dim=-1):
    """
    每行做差取绝对值然后求和
    """
    return -torch.sum((a - b).abs(), dim=dim)

def output_rel_err_similarity(a, b,dim=-1):
    """
    逐行计算相对误差相似度
    对于每一行，计算 (a - b) / (b.abs() + eps) 的绝对值，然后求和
    返回负值，因为误差越小越好（相似度越高）
    """
    eps = 1e-8
    rel_err_per_row = ((a - b) / (b.abs() + eps)).abs().sum(dim=dim)  # shape: (num_rows,)
    return -rel_err_per_row  # 取负值，因为误差越小越好

def output_weighted_dot_similarity(a, b,dim=-1):
    """
    逐行计算加权点积相似度
    使用 b 的绝对值作为权重，大权重的影响更大
    公式：sum(a * b * |b|) 对每行求和
    """
    return torch.sum(a * b * b.abs(), dim=dim)

def output_normalized_dot_similarity(a, b,dim=-1):
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
    return torch.sum(a_normalized * b_normalized, dim=dim)

def output_weighted_l2_similarity(a, b,dim=-1):
    """
    逐行计算加权L2距离（取负值）
    使用 b 的绝对值作为权重
    公式：-sum(|a - b| * |b|) 对每行求和
    """
    return -torch.sum((a - b).abs() * b.abs(), dim=dim)

def output_importance_weighted_similarity(a, b,dim=-1):
    """
    逐行计算重要性加权的相似度
    结合点积和权重重要性
    公式：sum(a * b) * ||b||，即点积乘以b的L2范数
    """
    dot_per_row = torch.sum(a * b, dim=dim)  # shape: (num_rows,)
    b_norm_per_row = torch.norm(b, p=2, dim=dim)  # shape: (num_rows,)
    return dot_per_row * b_norm_per_row

def compute_eigenvectors(matrix, symmetric=False, return_eigenvalues=False):
    """
    计算矩阵的特征向量
    
    参数:
        matrix: torch.Tensor, 输入矩阵，形状为 (n, n) 或 (..., n, n)
        symmetric: bool, 如果为 True，假设矩阵是对称的（或Hermitian），使用更稳定的 eigh 方法
                   如果为 False，使用 eig 方法（可能返回复数特征值）
        return_eigenvalues: bool, 如果为 True，同时返回特征值
    
    返回:
        如果 return_eigenvalues=False:
            eigenvectors: torch.Tensor, 特征向量，形状为 (n, n) 或 (..., n, n)
                          每一列是一个特征向量
        如果 return_eigenvalues=True:
            (eigenvalues, eigenvectors): tuple
                eigenvalues: torch.Tensor, 特征值，形状为 (n,) 或 (..., n)
                eigenvectors: torch.Tensor, 特征向量，形状为 (n, n) 或 (..., n, n)
    
    示例:
        >>> matrix = torch.randn(4, 4)
        >>> matrix = matrix @ matrix.T  # 创建对称矩阵
        >>> eigenvectors = compute_eigenvectors(matrix, symmetric=True)
        >>> eigenvalues, eigenvectors = compute_eigenvectors(matrix, symmetric=True, return_eigenvalues=True)
    """
    if symmetric:
        # 对于对称矩阵，使用 eigh（更稳定，返回实数）
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    else:
        # 对于一般矩阵，使用 eig（可能返回复数）
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        # 如果矩阵是实矩阵，特征值可能是复数，需要处理
        if eigenvalues.is_complex():
            # 对于实矩阵，复数特征值通常成对出现，取实部
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
    
    if return_eigenvalues:
        return eigenvalues, eigenvectors
    else:
        return eigenvectors

# import torch
from typing import Dict, Optional

@torch.no_grad()
def principal_angles_weighted_sin2_energy_k(
    W: torch.Tensor,
    W_q: torch.Tensor,
    energy: float = 0.90,
    k_max: Optional[int] = None,
    full_matrices: bool = False,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute principal angles between the TOP-k right-singular subspaces of W and W_q,
    where k is chosen by an energy threshold on W's singular values, and return a
    weighted sin^2 aggregation.

    Assumptions:
        W, W_q have shape (out_features, in_features).

    Definitions:
        W = U diag(S) V^T,  W_q = U_q diag(S_q) V_q^T
        Vk:  top-k right singular vectors of W  (in_features, k)
        Vk_q: top-k right singular vectors of W_q (in_features, k)
        cos(theta_i) are the singular values of Vk^T Vk_q

    Weighted sin^2 aggregation (data-free):
        weights w_i = S_i(W)^2 / sum_{j<=k} S_j(W)^2
        score = sum_{i<=k} w_i * sin^2(theta_i)

    Args:
        W, W_q: (out_features, in_features) tensors on same device/dtype.
        energy: energy threshold in (0,1], e.g., 0.90 or 0.95. Chooses smallest k s.t.
                sum_{i<=k} S_i^2 / sum_i S_i^2 >= energy.
        k_max: optional cap on k (e.g., 128). If None, no cap other than rank.
        full_matrices: passed to torch.linalg.svd (False recommended).
        eps: numerical epsilon for stability.

    Returns:
        dict with:
            - k_eff: effective k used (scalar int tensor)
            - energy_achieved: achieved energy ratio (scalar)
            - angles_rad / angles_deg: (k_eff,) sorted ascending
            - cos: (k_eff,) corresponding cos(theta_i)
            - sin2: (k_eff,) sin^2(theta_i)
            - weights: (k_eff,) normalized weights from W's singular values
            - weighted_sin2: scalar, sum_i weights_i * sin^2(theta_i)  (lower is better)
            - unweighted_sum_sin2: scalar, sum_i sin^2(theta_i)
            - fro_sin: scalar, ||sin(Theta)||_F = sqrt(sum_i sin^2(theta_i))
    """
    assert W.ndim == 2 and W_q.ndim == 2, "W and W_q must be 2D tensors."
    assert W.shape == W_q.shape, "W and W_q must have identical shapes."
    assert 0.0 < energy <= 1.0, "energy must be in (0, 1]."

    # Save original dtype and convert to float32 for SVD (SVD doesn't support float16)
    original_dtype = W.dtype
    W_f32 = W.float()
    W_q_f32 = W_q.float()

    out_f, in_f = W_f32.shape
    r = min(out_f, in_f)

    # --- SVD of W (need singular values S and right vectors Vh) ---
    U, S, Vh = torch.linalg.svd(W_f32, full_matrices=full_matrices)  # Vh is V^T
    # S shape: (r,) sorted descending

    # Choose k by energy on S^2
    S2 = S[:r].pow(2)
    total = S2.sum().clamp_min(eps)
    cum = torch.cumsum(S2, dim=0) / total
    k_energy = int(torch.searchsorted(cum, torch.tensor(energy, device=W.device, dtype=W_f32.dtype), right=False).item()) + 1
    k_eff = min(k_energy, r)
    if k_max is not None:
        k_eff = min(k_eff, int(k_max))
    if k_eff <= 0:
        raise ValueError("Effective k ended up <= 0; check inputs/energy/k_max.")

    energy_achieved = (cum[k_eff - 1]).detach()

    # Top-k right singular vectors of W: Vk = Vh[:k,:].T => (in_f, k_eff)
    Vk = Vh[:k_eff, :].T

    # --- SVD of W_q (need right vectors for top-k) ---
    Uq, Sq, Vhq = torch.linalg.svd(W_q_f32, full_matrices=full_matrices)
    Vk_q = Vhq[:k_eff, :].T

    # Core kxk matrix and its singular values => cos(theta_i)
    M = Vk.T @ Vk_q  # (k_eff, k_eff)
    _, cos_vals, _ = torch.linalg.svd(M, full_matrices=False)  # cos_vals sorted desc

    # Clamp for numerical safety; theoretically in [0,1]
    cos_vals = cos_vals.clamp(0.0, 1.0)

    angles_rad = torch.acos(cos_vals)  # in [0, pi/2]
    angles_deg = angles_rad * (180.0 / torch.pi)

    sin2 = 1.0 - cos_vals.pow(2)
    unweighted_sum_sin2 = sin2.sum()
    fro_sin = torch.sqrt(unweighted_sum_sin2.clamp_min(0.0))

    # Weights from W's singular values (top-k) using energy (sigma^2)
    w_raw = S2[:k_eff]
    weights = w_raw / w_raw.sum().clamp_min(eps)

    # Weighted sin^2 aggregation: lower is better
    weighted_sin2 = (weights * sin2).sum()

    # Sort angles ascending (small -> large) and permute aligned arrays for readability
    angles_rad, idx = torch.sort(angles_rad)
    angles_deg = angles_deg[idx]
    cos_vals = cos_vals[idx]
    sin2 = sin2[idx]
    # NOTE: weights correspond to W's singular directions (ordered by S, not by these angles).
    # For a clean "weighted by importance" score, we should keep weights aligned with the same i
    # indexing as cos_vals from SVD(M). Those cos_vals are also ordered by "best alignment" not
    # by singular-direction index. So we keep weights unsorted for scoring (already used).
    # We return weights in original top-k order.
    # If you want a per-angle "importance", you'd need an explicit matching, which is not canonical.

    return {
        "k_eff": torch.tensor(k_eff, device=W.device),
        "energy_achieved": energy_achieved,
        "angles_rad": angles_rad,
        "angles_deg": angles_deg,
        "cos": cos_vals,
        "sin2": sin2,
        "weights": weights,  # in top-k singular-value order of W (not angle-sorted)
        "weighted_sin2": weighted_sin2,
        "unweighted_sum_sin2": unweighted_sum_sin2,
        "fro_sin": fro_sin,
    }


# -------------------------
# Example usage:
# result = principal_angles_weighted_sin2_energy_k(W, W_q, energy=0.90, k_max=128)
# print("k_eff:", int(result["k_eff"].item()), "energy:", float(result["energy_achieved"].item()))
# print("weighted_sin2:", float(result["weighted_sin2"].item()))
# print("first 10 angles (deg):", result["angles_deg"][:10].cpu().tolist())


def compute_svd(matrix, full_matrices=True):
    """
    计算矩阵的奇异值分解 (SVD)
    
    对矩阵 A 进行 SVD 分解：A = U @ diag(S) @ V^T
    其中：
    - U: 左奇异向量矩阵
    - S: 奇异值向量（按降序排列）
    - V^T: 右奇异向量矩阵的转置
    
    参数:
        matrix: torch.Tensor, 输入矩阵，形状为 (m, n) 或 (..., m, n)
        full_matrices: bool, 如果为 True，返回完整的 U 和 V^T 矩阵
                      如果为 False，只返回必要的部分（更节省内存）
                      - full_matrices=True: U 形状为 (m, m), V^T 形状为 (n, n)
                      - full_matrices=False: U 形状为 (m, min(m,n)), V^T 形状为 (min(m,n), n)
    
    返回:
        (U, S, V_T): tuple
            U: torch.Tensor, 左奇异向量矩阵，形状为 (m, m) 或 (m, min(m,n))
            S: torch.Tensor, 奇异值向量（按降序排列），形状为 (min(m,n),)
            V_T: torch.Tensor, 右奇异向量矩阵的转置，形状为 (n, n) 或 (min(m,n), n)
    
    示例:
        >>> matrix = torch.randn(5, 3)  # 形状为 (5, 3)
        >>> U, S, V_T = compute_svd(matrix, full_matrices=True)
        >>> # 验证: matrix ≈ U @ diag(S) @ V^T
        >>> reconstructed = U @ torch.diag(S[:3]) @ V_T
        >>> print(torch.allclose(matrix, reconstructed))  # 应该为 True
        
        >>> # 只保留前 k 个最大的奇异值（低秩近似）
        >>> k = 2
        >>> U_k = U[:, :k]  # (5, 2)
        >>> S_k = S[:k]  # (2,)
        >>> V_T_k = V_T[:k, :]  # (2, 3)
        >>> low_rank_approx = U_k @ torch.diag(S_k) @ V_T_k
    """
    U, S, V_T = torch.linalg.svd(matrix, full_matrices=full_matrices)
    
    return U, S, V_T

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

def select_rows_by_max_similarity0124(sim_list, w_list):
    """
    根据每行的最大相似度值，从多个权重矩阵中选择对应的行
    
    参数:
        sim_list: 相似度向量列表，每个元素shape为 (num_rows,)
        w_list: 权重矩阵列表，每个元素shape为 (num_rows, q_group_size) 或展平后的向量
    
    返回:
        selected_w: 根据最大相似度选择的权重矩阵，shape与w_list[0]相同
        masked_w_list: 掩码后的权重矩阵列表，每个元素与对应的w_list元素形状相同，
                       只有被选中的位置保留原值，其他位置设为0
    """
    # 将相似度值堆叠，找到每行相似度最大的索引
    sim_stack = torch.stack(sim_list, dim=0)  # shape: (num_sims, num_rows)
    max_idx = sim_stack.argmax(dim=0)  # shape: (num_rows,)，值为 0, 1, 2, ...
    
    # 根据索引选择对应的 w
    w_stack = torch.stack(w_list, dim=0)  # shape: (num_weights, num_rows, ...)
    num_rows = w_list[0].shape[0]
    row_indices = torch.arange(num_rows, device=w_list[0].device)  # shape: (num_rows,)
    selected_w = w_stack[max_idx, row_indices]  # shape: (num_rows, ...)
    
    # 创建掩码后的权重矩阵列表
    masked_w_list = []
    mask_list = []
    for i, w in enumerate(w_list):
        # 创建掩码：如果该位置被选中（max_idx == i），保留原值；否则设为0
        # Keep the mask in the candidate dtype so masking FP16 weights does
        # not silently allocate a full FP32 matrix during real quantization.
        mask = (max_idx == i).to(dtype=w.dtype)  # shape: (num_rows,)
        # 如果w是多维的，需要扩展mask的维度以匹配w的形状
        if w.dim() > 1:
            # w可能是 (num_rows, ...) 的形状，需要扩展mask
            mask = mask.view(-1, *([1] * (w.dim() - 1)))  # shape: (num_rows, 1, ...)
            mask = mask.expand_as(w)  # shape: 与w相同
        else:
            # w是展平后的向量，mask已经是 (num_rows,) 的形状
            pass
        masked_w = w * mask  # 只有被选中的位置保留原值，其他为0
        masked_w_list.append(masked_w)
        mask_list.append(mask)
    return selected_w, masked_w_list, mask_list



def select_column_by_max_similarity(sim_list, w_list):
    """
    根据每列的最大相似度值，从多个权重矩阵中选择对应的列
    
    参数:
        sim_list: 相似度向量列表，每个元素shape为 (num_cols,)
        w_list: 权重矩阵列表，每个元素shape为 (num_rows, num_cols)
    
    返回:
        selected_w: 根据最大相似度选择的权重矩阵，shape为 (num_rows, num_cols)
    """
    # 将相似度值堆叠，找到每列相似度最大的索引
    sim_stack = torch.stack(sim_list, dim=0)  # shape: (num_sims, num_cols)
    max_idx = sim_stack.argmax(dim=0)  # shape: (num_cols,)，值为 0, 1, 2, ...
    
    # 根据索引选择对应的列
    # 对于每一列，从对应的权重矩阵中选择该列
    num_cols = w_list[0].shape[1]
    num_rows = w_list[0].shape[0]
    selected_w = torch.zeros_like(w_list[0])  # shape: (num_rows, num_cols)
    
    # 对于每个候选权重矩阵，选择其对应的列
    for i, w in enumerate(w_list):
        mask = (max_idx == i)  # shape: (num_cols,)，布尔掩码，标记哪些列应该从 w_list[i] 选择
        selected_w[:, mask] = w[:, mask]
    
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

def output_l2_norm_similarity(a, b,dim=-1):
    """
    每行做差取绝对值然后求和
    """
    return -torch.norm(a - b, p='fro', dim=-1)

def output_dot_div_mse_similarity(a, b,dim=-1):
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

def weighted_cosine_similarity(a, b,dim=-1):
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

def weighted_dot_similarity(a, b,dim=-1):
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


def output_e_distance(a, b,dim=-1):
    a= a.sum(dim=dim)
    b= b.sum(dim=dim)
    a = torch.exp(a)
    b = torch.exp(b)
    return torch.abs(a-b)

def output_e_distance_abs(a, b,dim=-1):
    a = torch.abs(a)
    b = torch.abs(b)
    a= a.sum(dim=dim)
    b= b.sum(dim=dim)
    a = torch.exp(a)
    b = torch.exp(b)
    return torch.abs(a-b)
def output_e_distance_square(a, b,dim=-1):
    a = a.pow(2)
    b = b.pow(2)
    a= a.sum(dim=dim)
    b= b.sum(dim=dim)
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

# def rtn_quantize_group_asym(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
#     """
#     Group-wise asymmetric RTN quantization for Linear weights.

#     Args:
#         weight: Tensor of shape [out_features, in_features]
#         bits:   Quantization bits (e.g., 4 or 8)
#         group_size: group size along in_features dimension

#     Returns:
#         q:      int tensor, shape [out_features, in_features_pad]
#         scales: float tensor, shape [out_features, num_groups]
#         zeros:  int tensor, shape [out_features, num_groups]
#     """
#     assert weight.dim() == 2, "Expect weight shape [out_features, in_features]"
#     out_features, in_features = weight.shape

#     qmin = 0
#     qmax = 2**bits - 1  # e.g., 4bit -> 15

#     # pad to make in_features divisible by group_size
#     pad = (group_size - (in_features % group_size)) % group_size
#     if pad != 0:
#         weight = torch.cat([weight, weight.new_zeros(out_features, pad)], dim=1)
#     in_features_pad = weight.shape[1]
#     num_groups = in_features_pad // group_size

#     # [out, num_groups, group_size]
#     w = weight.view(out_features, num_groups, group_size)

#     w_min = w.amin(dim=2, keepdim=True)  # [out, num_groups, 1]
#     w_max = w.amax(dim=2, keepdim=True)  # [out, num_groups, 1]

#     # scale = (max - min) / (qmax - qmin)
#     scales = (w_max - w_min) / float(qmax - qmin)

#     # avoid division by zero (all values equal in a group)
#     scales = torch.clamp(scales, min=1e-8)

#     # zero-point: z = round(qmin - min/scale) = round(-min/scale) since qmin=0
#     zeros = torch.round(qmin - (w_min / scales)).clamp(qmin, qmax)

#     # quantize: q = round(w/scale + z)
#     q = torch.round(w / scales + zeros).clamp(qmin, qmax)

#     # return shapes:
#     # q: [out, in_pad], scales/zeros: [out, num_groups]
#     q = q.view(out_features, in_features_pad).to(torch.int16)  # int8/int16均可；很多实现用 int8 存 0..15
#     scales = scales.squeeze(-1)
#     zeros = zeros.squeeze(-1).to(torch.int16)

#     return q, scales, zeros

import torch

def rtn_quant_dequant_group_asym(weight: torch.Tensor, bits: int = 4, group_size: int = 128,
                                 return_params: bool = True):
    """
    Group-wise asymmetric RTN: quantize then dequantize.

    Returns:
        w_hat:  float tensor, [out_features, in_features_pad]
        scales: float tensor, [out_features, num_groups]          (if return_params)
        zeros:  int tensor,   [out_features, num_groups]          (if return_params)
    """
    assert weight.dim() == 2
    out_features, in_features = weight.shape

    qmin = 0
    qmax = 2**bits - 1

    pad = (group_size - (in_features % group_size)) % group_size
    if pad != 0:
        weight = torch.cat([weight, weight.new_zeros(out_features, pad)], dim=1)
    in_features_pad = weight.shape[1]
    num_groups = in_features_pad // group_size

    w = weight.view(out_features, num_groups, group_size)

    w_min = w.amin(dim=2, keepdim=True)
    w_max = w.amax(dim=2, keepdim=True)

    scales = (w_max - w_min) / float(qmax - qmin)
    scales = torch.clamp(scales, min=1e-8)

    zeros = torch.round(qmin - (w_min / scales)).clamp(qmin, qmax)

    q = torch.round(w / scales + zeros).clamp(qmin, qmax)

    w_hat = (q - zeros) * scales
    w_hat = w_hat.view(out_features, in_features_pad)

    if not return_params:
        return w_hat

    return w_hat, scales.squeeze(-1), zeros.squeeze(-1)


def pseudo_quantize_tensor722(
    o_w, current_awq_x_s=None, gptq_w=None, gptq_scales=None, gptq_zeros=None, n_bit=8, quant_type='awq', zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False, th_t=1, search_flag=False
):
    if current_awq_x_s is None or current_awq_x_s.numel() != o_w.shape[-1]:
        current_awq_x_s = torch.ones(o_w.shape[-1], device=o_w.device, dtype=o_w.dtype)

    def _awq_only_quant(weight):
        org_shape = weight.shape
        scale_x = current_awq_x_s.view(1, -1).to(device=weight.device, dtype=weight.dtype)
        scaled_w = weight * scale_x
        if q_group_size > 0:
            assert org_shape[-1] % q_group_size == 0
            q_w = scaled_w.reshape(-1, q_group_size)
        else:
            q_w = scaled_w.reshape(scaled_w.shape[0], -1)
        if zero_point:
            max_val = q_w.amax(dim=1, keepdim=True)
            min_val = q_w.amin(dim=1, keepdim=True)
            max_int = 2 ** n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            q_dequant = (torch.clamp(torch.round(q_w / scales) + zeros, min_int, max_int) - zeros) * scales
        else:
            max_val = q_w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            scales = max_val / max_int
            zeros = torch.zeros_like(scales)
            q_dequant = torch.clamp(torch.round(q_w / scales), min_int, max_int) * scales
        out = q_dequant.reshape(org_shape) / scale_x
        if get_scale_zp:
            return out, scales.view(org_shape[0], -1), zeros.view(org_shape[0], -1)
        return out

    if gptq_w is None or gptq_scales is None or gptq_zeros is None:
        return _awq_only_quant(o_w)

    print("gptq_w.shape:", gptq_w.shape)
    print("gptq_scales.shape:", gptq_scales.shape)
    print("gptq_zeros.shape:", gptq_zeros.shape)

    ooo = gptq_w.clone()
    
    def back_quantize(quantized_int,scales,zeros,min_int=None,max_int=None):
        if min_int is None and max_int is None:
            w = (quantized_int - zeros) * scales
        else:
            w = (torch.clamp(quantized_int + zeros, min_int, max_int) - zeros) * scales
        assert torch.isnan(w).sum() == 0
        return w
    
    org_w_shape = o_w.shape
    org_w = o_w.clone()
    awq_w = o_w.mul_(current_awq_x_s.view(1, -1))
    # print("awq_w.shape:",awq_w.shape)
    # sys.exit()
    
    #check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point, q_group_size)
    assert gptq_w.shape == org_w_shape
    # gptq_scale_zeros = gptq_scales * gptq_zeros
    # gptq_w_int = torch.round(
    #                 (gptq_w + gptq_scale_zeros)
    #                 / gptq_scales
    #             ).to(torch.int)[:, None]

    # gptq_int = torch.round(gptq_w/gptq_scales)
    # gptq_int_zeros = gptq_int + gptq_zeros
    # gptq_scale_zeros = gptq_scales * gptq_zeros
    #gptq_w = gptq_int * gptq_scales#back_quantize(gptq_w_int,gptq_scales,gptq_zeros,min_int=None,max_int=None)
    # print('gptq_int.min(),gptq_int.max():',gptq_int.min(),gptq_int.max())
    # print('gptq_int_zeros.min(),gptq_int_zeros.max():',gptq_int_zeros.min(),gptq_int_zeros.max())
    # #assert gptq_int_zeros.min() >= 0 and gptq_int_zeros.max() <= (2**n_bit - 1)
    #print('gptq_zero.min(),gptq_int.max():',gptq_int.min(),gptq_int.max())

    #print('gptq_w.min(),gptq_w.max():',gptq_w.min(),gptq_w.max())
    

    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        awq_w = awq_w.reshape(-1, q_group_size)
        #gptq_w = gptq_w.reshape(-1, q_group_size)
    assert awq_w.dim() == 2
    if zero_point:
        max_val = awq_w.amax(dim=1, keepdim=True)
        med_val = awq_w.median(dim=1, keepdim=True)[0]
        avg_val = awq_w.mean(dim=1, keepdim=True)
        min_val = awq_w.amin(dim=1, keepdim=True)
        q25_val = torch.quantile(awq_w.float(), 0.25, dim=1, keepdim=True)
        q75_val = torch.quantile(awq_w.float(), 0.75, dim=1, keepdim=True)
        q33_val = torch.quantile(awq_w.float(), 0.33, dim=1, keepdim=True)
        q66_val = torch.quantile(awq_w.float(), 0.66, dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        pos_scales = (max_val ).clamp(min=1e-5) / max_int
        neg_scales = (-min_val).clamp(min=1e-5) / max_int

        med_scales = (med_val-min_val).clamp(min=1e-5) / max_int
        up_med_scales = (max_val-med_val).clamp(min=1e-5) / max_int

        avg_scales = (avg_val-min_val).clamp(min=1e-5) / max_int
        up_avg_scales = (max_val-avg_val).clamp(min=1e-5) / max_int

        scales_0033 = ( torch.max(torch.max(q33_val-min_val, -min_val), q33_val) ).clamp(min=1e-5) / max_int
        scales_3366 = ( torch.max(torch.max(q66_val-q33_val, -min_val), q66_val) ).clamp(min=1e-5) / max_int
        scales_6611 = ( torch.max(torch.max(max_val-q66_val, -min_val), max_val) ).clamp(min=1e-5) / max_int
        zeros_0033 = (-torch.round(min_val / scales_0033)).clamp_(min_int, max_int)
        #assert (-torch.round(min_val / scales_0033) >= min_int).all() and (-torch.round(min_val / scales_0033) <= max_int).all(), f"min_val: {min_val}, scales_0033: {scales_0033}, zeros_0033: {zeros_0033}"
        zeros_3366 = (-torch.round(q33_val / scales_3366)).clamp_(min_int, max_int)
        #assert (-torch.round(q33_val / scales_3366) >= min_int).all() and (-torch.round(q33_val / scales_3366) <= max_int).all(),f"q33_val: {q33_val}, scales_3366: {scales_3366}, zeros_3366: {zeros_3366}"
        zeros_6611 = (-torch.round(q66_val / scales_6611)).clamp_(min_int, max_int)
        #assert (-torch.round(q66_val / scales_6611) >= min_int).all() and (-torch.round(q66_val / scales_6611) <= max_int).all(),f"q66_val: {q66_val}, scales_6611: {scales_6611}, zeros_6611: {zeros_6611}"

        med_zeros = (-torch.round(min_val / med_scales)).clamp_(min_int, max_int)

        pos_zeros = (-torch.round(0 / pos_scales)).clamp_(min_int, max_int)
        neg_zeros = (-torch.round(min_val / neg_scales)).clamp_(min_int, max_int)
        up_med_zeros = (-torch.round(med_val / up_med_scales)).clamp_(min_int, max_int)

        up_avg_zeros = (-torch.round(avg_val / up_avg_scales)).clamp_(min_int, max_int)
        avg_zeros = (-torch.round(min_val / avg_scales)).clamp_(min_int, max_int)

        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = awq_w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(awq_w).sum() == 0

    quantized_float = awq_w / scales
    floor_values = torch.floor(quantized_float)
    fractional_parts = quantized_float - floor_values
    

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
        #w2 = gptq_w

        w2 = back_quantize(torch.round(awq_w/med_scales),med_scales,med_zeros,min_int,max_int)

        w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        
        w1 = back_quantize(torch.round(awq_w/up_med_scales),up_med_scales,up_med_zeros,min_int,max_int)

        w1 = back_quantize(torch.round(awq_w/up_avg_scales),up_avg_scales,up_avg_zeros,min_int,max_int)

        base_w = back_quantize(base_quantized_int,scales,zeros,min_int,max_int)
        #w3 = back_quantize(combined_quantized_int,scales,zeros,min_int,max_int)
        #w1 = w2
        #w3 = w1 #back_quantize(top_sum_quantized_int,scales,zeros,min_int,max_int)
        #w1 = w2
        
        w3 = back_quantize(torch.round(awq_w/avg_scales),avg_scales,avg_zeros,min_int,max_int)

        w2 = w3
        #w1 = w
        #w3 = gptq_w #back_quantize(median_quantized_int,scales,zeros,min_int,max_int)
        w1 = back_quantize(torch.round(awq_w/scales_0033),scales_0033,zeros_0033,min_int,max_int)
        w2 = back_quantize(torch.round(awq_w/scales_3366),scales_3366,zeros_3366,min_int,max_int)
        w3 = back_quantize(torch.round(awq_w/scales_6611),scales_6611,zeros_6611,min_int,max_int)

        w1 = back_quantize(torch.round(awq_w/scales),scales,zeros,min_int,max_int)
        w2 = back_quantize(torch.round(awq_w/pos_scales),pos_scales,pos_zeros,min_int,max_int)
        w3 = back_quantize(torch.round(awq_w/med_scales),med_scales,med_zeros,min_int,max_int)
        #w3 = back_quantize(torch.round(w/neg_scales),neg_scales,neg_zeros,min_int,max_int)
        #w1 = w2

        # w2 = gptq_w
        # w3 = w2
        w2 = gptq_w
        w3 = w2
        
    # 确保所有张量的数据类型一致，避免类型不匹配错误
    w1 = w1.to(org_w.dtype)
    w2 = w2.to(org_w.dtype)
    w3 = w3.to(org_w.dtype)
    
    w1 = w1.reshape(org_w_shape)

    w1 = w1 / current_awq_x_s
    w2 = w2.reshape(org_w_shape)
    w3 = w3.reshape(org_w_shape)
    
    w4, scales_4, zeros_4 = rtn_quant_dequant_group_asym(org_w.clone(), bits=n_bit, group_size=q_group_size)
    
    sim1 = mse_sim(w1, org_w,torch.ones_like(org_w)).view(-1)
    sim2 = mse_sim(w2, org_w,torch.ones_like(org_w)).view(-1)
    sim3 = mse_sim(w3, org_w,torch.ones_like(org_w)).view(-1)
    sim4 = mse_sim(w4, org_w,torch.ones_like(org_w)).view(-1)
    
    sim_stack = torch.stack([sim1, sim2, sim4], dim=0)  # shape: (3, num_elements)

    # awq + gptq
    # sim_stack = torch.stack([sim1, sim2], dim=0)
    # awq + rtn
    # sim_stack = torch.stack([sim1, sim4], dim=0)
    # rtn + gptq
    # sim_stack = torch.stack([sim2, sim4], dim=0)


    max_idx = sim_stack.argmax(dim=0)  # shape: (num_elements,)，值为 0, 1, 2
    
    # 统计每个来源的数量
    count_w1 = (max_idx == 0).sum().item()
    count_w2 = (max_idx == 1).sum().item()
    count_w3 = (max_idx == 2).sum().item()
    count_w4 = (max_idx == 3).sum().item()
    total_elements = max_idx.numel()
    
    # 打印统计信息
    print(f"[select_rows_by_max_similarity] Total elements: {total_elements}, "
          f"from w1: {count_w1}, from w2: {count_w2}, from w3: {count_w3}, from w4: {count_w4}")
    
    f_new_w = select_rows_by_max_similarity([sim1, sim2, sim4], [w1.view(-1), w2.view(-1), w4.view(-1)])

    # awq + gptq
    # f_new_w = select_rows_by_max_similarity([sim1, sim2], [w1.view(-1), w2.view(-1)])
    # awq + rtn
    # f_new_w = select_rows_by_max_similarity([sim1, sim4], [w1.view(-1), w4.view(-1)])
    # rtn + gptq
    # f_new_w = select_rows_by_max_similarity([sim2, sim4], [w2.view(-1), w4.view(-1)])
    
    f_new_w = f_new_w.reshape(org_w.shape)

    #     new_w = w3
    
    #new_w = gptq_w.clone()
    #new_w = w1
    new_w = f_new_w
    w = new_w.reshape(org_w_shape)
    

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        if quant_type == 'awq':
            print('awq selected')
            return w1
        elif quant_type == 'gptq':
            print('gptq selected')
            return w2
        elif quant_type == 'rtn':
            print('rtn selected')
            return w4
        else:
            print('all selected')
            return w
        # w1 awq w2 gptq w4 rtn w co
        # return w1
        # return w2
        # return w4
        # return w

# def pseudo_quantize_tensor722(
#     o_w, current_awq_x_s, gptq_w,gptq_scales, gptq_zeros, n_bit=8, quant_type = 'awq', zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
# ):
#     print("gptq_w.shape:",gptq_w.shape)
#     print("gptq_scales.shape:",gptq_scales.shape)
#     print("gptq_zeros.shape:",gptq_zeros.shape)

#     ooo = gptq_w.clone()
    
#     def back_quantize(quantized_int,scales,zeros,min_int=None,max_int=None):
#         if min_int is None and max_int is None:
#             w = (quantized_int - zeros) * scales
#         else:
#             w = (torch.clamp(quantized_int + zeros, min_int, max_int) - zeros) * scales
#         assert torch.isnan(w).sum() == 0
#         return w
    
#     org_w_shape = o_w.shape
#     org_w = o_w.clone()
#     awq_w = o_w.mul_(current_awq_x_s.view(1, -1))
#     #check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point, q_group_size)
#     assert gptq_w.shape == org_w_shape
    
#     if q_group_size > 0:
#         assert org_w_shape[-1] % q_group_size == 0
#         awq_w = awq_w.reshape(-1, q_group_size)
#         #gptq_w = gptq_w.reshape(-1, q_group_size)
#     assert awq_w.dim() == 2
#     if zero_point:
#         max_val = awq_w.amax(dim=1, keepdim=True)
#         med_val = awq_w.median(dim=1, keepdim=True)[0]
#         avg_val = awq_w.mean(dim=1, keepdim=True)
#         min_val = awq_w.amin(dim=1, keepdim=True)
#         q25_val = torch.quantile(awq_w.float(), 0.25, dim=1, keepdim=True)
#         q75_val = torch.quantile(awq_w.float(), 0.75, dim=1, keepdim=True)
#         q33_val = torch.quantile(awq_w.float(), 0.33, dim=1, keepdim=True)
#         q66_val = torch.quantile(awq_w.float(), 0.66, dim=1, keepdim=True)
#         max_int = 2**n_bit - 1
#         min_int = 0
#         scales = (max_val - min_val).clamp(min=1e-5) / max_int
#         pos_scales = (max_val ).clamp(min=1e-5) / max_int
#         neg_scales = (-min_val).clamp(min=1e-5) / max_int

#         med_scales = (med_val-min_val).clamp(min=1e-5) / max_int
#         up_med_scales = (max_val-med_val).clamp(min=1e-5) / max_int

#         avg_scales = (avg_val-min_val).clamp(min=1e-5) / max_int
#         up_avg_scales = (max_val-avg_val).clamp(min=1e-5) / max_int

#         scales_0033 = ( torch.max(torch.max(q33_val-min_val, -min_val), q33_val) ).clamp(min=1e-5) / max_int
#         scales_3366 = ( torch.max(torch.max(q66_val-q33_val, -min_val), q66_val) ).clamp(min=1e-5) / max_int
#         scales_6611 = ( torch.max(torch.max(max_val-q66_val, -min_val), max_val) ).clamp(min=1e-5) / max_int
#         zeros_0033 = (-torch.round(min_val / scales_0033)).clamp_(min_int, max_int)
#         #assert (-torch.round(min_val / scales_0033) >= min_int).all() and (-torch.round(min_val / scales_0033) <= max_int).all(), f"min_val: {min_val}, scales_0033: {scales_0033}, zeros_0033: {zeros_0033}"
#         zeros_3366 = (-torch.round(q33_val / scales_3366)).clamp_(min_int, max_int)
#         #assert (-torch.round(q33_val / scales_3366) >= min_int).all() and (-torch.round(q33_val / scales_3366) <= max_int).all(),f"q33_val: {q33_val}, scales_3366: {scales_3366}, zeros_3366: {zeros_3366}"
#         zeros_6611 = (-torch.round(q66_val / scales_6611)).clamp_(min_int, max_int)
#         #assert (-torch.round(q66_val / scales_6611) >= min_int).all() and (-torch.round(q66_val / scales_6611) <= max_int).all(),f"q66_val: {q66_val}, scales_6611: {scales_6611}, zeros_6611: {zeros_6611}"

#         med_zeros = (-torch.round(min_val / med_scales)).clamp_(min_int, max_int)

#         pos_zeros = (-torch.round(0 / pos_scales)).clamp_(min_int, max_int)
#         neg_zeros = (-torch.round(min_val / neg_scales)).clamp_(min_int, max_int)

#         up_med_zeros = (-torch.round(med_val / up_med_scales)).clamp_(min_int, max_int)

#         up_avg_zeros = (-torch.round(avg_val / up_avg_scales)).clamp_(min_int, max_int)
#         avg_zeros = (-torch.round(min_val / avg_scales)).clamp_(min_int, max_int)

#         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
#     else:  # we actually never used this
#         assert min_val is None
#         max_val = awq_w.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)
#         max_int = 2 ** (n_bit - 1) - 1
#         min_int = -(2 ** (n_bit - 1))
#         scales = max_val / max_int
#         zeros = 0

#     assert torch.isnan(scales).sum() == 0
#     assert torch.isnan(awq_w).sum() == 0

#     quantized_float = awq_w / scales
#     floor_values = torch.floor(quantized_float)
#     fractional_parts = quantized_float - floor_values
    

#     #print(f"search_flag: {search_flag}")
#     if search_flag == True:
#         quantized_int = torch.round(quantized_float)
#         original_quantized_int = quantized_int
#         median_quantized_int = quantized_int
#         w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
#         w2 = w1
#         w3 = w1
#         base_w = w1
#     else:
        
#         top_sum_quantized_int = top_sum_number(fractional_parts) + floor_values 
#         original_quantized_int = torch.round(quantized_float)
#         base_quantized_int = torch.floor(quantized_float)
#         median_quantized_int = median_threshold_number(fractional_parts) + floor_values
#         tmp_sim1 = output_dot_similarity(original_quantized_int, quantized_float)
#         tmp_sim2 = output_dot_similarity(top_sum_quantized_int, quantized_float)
#         combined_quantized_int = select_rows_by_max_similarity([tmp_sim1, tmp_sim2], [original_quantized_int, top_sum_quantized_int])
#         #w2 = gptq_w

#         w2 = back_quantize(torch.round(awq_w/med_scales),med_scales,med_zeros,min_int,max_int)

#         # w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        
#         w1 = back_quantize(torch.round(awq_w/up_med_scales),up_med_scales,up_med_zeros,min_int,max_int)

#         # w1 = back_quantize(torch.round(awq_w/up_avg_scales),up_avg_scales,up_avg_zeros,min_int,max_int)

#         #w3 = back_quantize(combined_quantized_int,scales,zeros,min_int,max_int)
#         #w1 = w2
#         #w3 = w1 #back_quantize(top_sum_quantized_int,scales,zeros,min_int,max_int)
#         #w1 = w2
        
#         # w3 = back_quantize(torch.round(awq_w/avg_scales),avg_scales,avg_zeros,min_int,max_int)

#         # w2 = w3


#         # w1 = back_quantize(torch.round(awq_w/scales_0033),scales_0033,zeros_0033,min_int,max_int)
#         # w2 = back_quantize(torch.round(awq_w/scales_3366),scales_3366,zeros_3366,min_int,max_int)
#         # w3 = back_quantize(torch.round(awq_w/scales_6611),scales_6611,zeros_6611,min_int,max_int)

#         # w1 = back_quantize(torch.round(awq_w/scales),scales,zeros,min_int,max_int)
#         # w2 = back_quantize(torch.round(awq_w/pos_scales),pos_scales,pos_zeros,min_int,max_int)
#         # w3 = back_quantize(torch.round(awq_w/med_scales),med_scales,med_zeros,min_int,max_int)



#         # w2 = gptq_w
#         # w3 = w2
        
#     # 确保所有张量的数据类型一致，避免类型不匹配错误
#     w1 = w1.to(org_w.dtype)
#     w2 = w2.to(org_w.dtype)
    
#     w1 = w1.reshape(org_w_shape)
#     w2 = w2.reshape(org_w_shape)

#     w1 = w1 / current_awq_x_s
#     w2 = w2 / current_awq_x_s

    
#     sim1 = mse_sim(w1, org_w,torch.ones_like(org_w)).view(-1)
#     sim2 = mse_sim(w2, org_w,torch.ones_like(org_w)).view(-1)
    
#     sim_stack = torch.stack([sim1, sim2], dim=0)

#     max_idx = sim_stack.argmax(dim=0)  # shape: (num_elements,)，值为 0, 1, 2
    
#     # 统计每个来源的数量
#     count_w1 = (max_idx == 0).sum().item()
#     count_w2 = (max_idx == 1).sum().item()
#     total_elements = count_w1 + count_w2
    
#     # 打印统计信息
#     print(f"[select_rows_by_max_similarity] Total elements: {total_elements}, "
#           f"from w1: {count_w1}, from w2: {count_w2}")
    
#     f_new_w = select_rows_by_max_similarity([sim1, sim2], [w1.view(-1), w2.view(-1)])
    
#     f_new_w = f_new_w.reshape(org_w.shape)

#     new_w = f_new_w
#     w = new_w.reshape(org_w_shape)
    

#     if get_scale_zp:
#         return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
#     else:
#         return w

def pseudo_quantize_tensor0124(
    o_w, current_awq_x_s, gptq_w, gptq_unscaled_wxxx, gptq_scales, gptq_zeros, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,th_t=1,search_flag=False
):
    #assert (gptq_w == gptq_unscaled_w).all(),"gptq_w and gptq_unscaled_w are not the same"
    org_w_shape = o_w.shape
    org_w = o_w.clone()
    #check_gptq_w_bit_compatibility(gptq_w, w, n_bit, zero_point, q_group_size)
    assert gptq_w.shape == org_w_shape
    awq_w = o_w.mul_(current_awq_x_s)


    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        awq_w = awq_w.reshape(-1, q_group_size)
        # gptq_w = gptq_w.reshape(-1, q_group_size)
        # gptq_unscaled_w = gptq_unscaled_w.reshape(-1, q_group_size)
    assert awq_w.dim() == 2
    if zero_point:
        max_val = awq_w.amax(dim=1, keepdim=True)
        min_val = awq_w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = awq_w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(awq_w).sum() == 0

    quantized_float = awq_w / scales
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
        w2 = gptq_w
        
        w1 = back_quantize(original_quantized_int,scales,zeros,min_int,max_int)
        #base_w = back_quantize(base_quantized_int,scales,zeros,min_int,max_int)
        #w3 = back_quantize(combined_quantized_int,scales,zeros,min_int,max_int)
        #w1 = w2
        #w3 = w1#back_quantize(top_sum_quantized_int,scales,zeros,min_int,max_int)
        w3 = w2
        #w3 = gptq_w #back_quantize(median_quantized_int,scales,zeros,min_int,max_int)
   
    
    
    # 确保所有张量的数据类型和设备一致，避免类型不匹配错误
    w1 = w1.to(org_w.device, dtype=org_w.dtype)
    w2 = w2.to(org_w.device, dtype=org_w.dtype)
    #w3 = w3.to(w_original.device, dtype=w_original.dtype)
    #base_w = base_w.to(w_original.device, dtype=w_original.dtype)
    w1 = w1.reshape(org_w_shape)
    w2 = w2.reshape(org_w_shape)
    #w3 = w3.reshape(org_w_shape)
   
    
    

    # original_awq_w = w1
    # original_gptq_w = w2
    
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
    
    # 计算两个矩阵差值的Frobenius norm，然后广播成 (num_rows,) 的形状
    # 计算 Frobenius norm（标量）
    
    # row_weights 与 w_original 进行矩阵乘法，得到一个向量
    # row_weights: (num_rows,), w_original: (num_rows, num_cols)
    # 结果: (num_cols,)
    # 方案1: 每列的和乘以 norm（保留符号，同时与 norm 相关）
    #col_weights = torch.sum(w_original, dim=0) * torch.norm(w_original, p=2, dim=0)
    
    # 方案2: 每列的均值乘以 norm（归一化版本）
    #col_weights = torch.norm(w_original, p=2, dim=0)  # shape: (num_cols,)
    
    # 方案3: 每列的主方向（符号）乘以 norm
    # col_sum = torch.sum(w_original, dim=0)
    # col_norm = torch.norm(w_original, p=2, dim=0)
    # col_weights = torch.sign(col_sum) * col_norm
    
    # 方案4: 每列的加权和，权重与 norm 相关
    # col_norm = torch.norm(w_original, p=2, dim=0)
    # col_weights = torch.sum(w_original * col_norm.unsqueeze(0), dim=0)
    #col_weights = torch.amax(w_original, dim=0)
    #col_weights = 1.0 / (col_weights + 1e-8)  # 取倒数，添加小值避免除以零
    # #weighted_w_original = row_weights @ w_original  # shape: (num_cols,) 或 (..., num_cols)
    # row_weights = torch.norm(w_original, p=2, dim=-1) 
    # fro_norm1 = -torch.norm(w1@col_weights - w_original@col_weights, p='fro')
    # fro_norm2 = -torch.norm(w2@col_weights - w_original@col_weights, p='fro')
    # fro_norm3 = -torch.norm(w3@col_weights - w_original@col_weights, p='fro')
    # #print(f"fro_norm1: {fro_norm1.item():.6f}, fro_norm2: {fro_norm2.item():.6f}, fro_norm3: {fro_norm3.item():.6f}")
    # # 获取 num_rows（从 w_original 的第一个维度，用于后续统计）
    # # 由于 Frobenius norm 是标量，我们需要将其广播成与后续代码期望的形状一致
    # # 后续代码期望 sim1, sim2, sim3 是 (num_rows,) 的形状
    #num_rows = w_original.shape[0] if w_original.dim() >= 1 else 1
    
    # # 将标量广播成 (num_rows,) 的形状
    # # 使用 expand 或 repeat 来广播，但需要先创建一个1D张量
    # sim1 = fro_norm1.unsqueeze(0).expand(num_rows)
    # sim2 = fro_norm2.unsqueeze(0).expand(num_rows)
    # sim3 = fro_norm3.unsqueeze(0).expand(num_rows)

    # sim1 = -(w1@col_weights - w_original@col_weights).abs()
    # sim2 = -(w2@col_weights - w_original@col_weights).abs()
    # sim3 = -(w3@col_weights - w_original@col_weights).abs()

    # sim1 = -principal_angles_weighted_sin2_energy_k(w_original, w1)['weighted_sin2']
    # sim2 = -principal_angles_weighted_sin2_energy_k(w_original, w2)['weighted_sin2']
    # sim3 = -principal_angles_weighted_sin2_energy_k(w_original, w3)['weighted_sin2']

    # row_weights = torch.norm(w_original, p=2, dim=-1) #* torch.norm(col_weights, p=2)  # shape: (num_rows,)
    # #row_weights = row_weights.unsqueeze(1).expand(-1, w_original.shape[1]) # shape: (num_rows, num_cols)
    # row_weights = torch.ones_like(row_weights)
   
    # sim1 = -torch.norm((w1@col_weights - w_original@col_weights)*row_weights, p='fro')
    # sim2 = -torch.norm((w2@col_weights - w_original@col_weights)*row_weights, p='fro')
    # sim3 = -torch.norm((w3@col_weights - w_original@col_weights)*row_weights, p='fro')

    # sim1 = sim1.unsqueeze(0).expand(num_rows)
    # sim2 = sim2.unsqueeze(0).expand(num_rows)
    # sim3 = sim3.unsqueeze(0).expand(num_rows)
    
    # # sim1 = output_dot_similarity(w1-base_w, w_original-base_w)
    # # sim2 = output_dot_similarity(w2-base_w, w_original-base_w)
    # # sim3 = output_dot_similarity(w3-base_w, w_original-base_w)
    # #print(f"sim1: {sim1.mean().item():.6f}, sim2: {sim2.mean().item():.6f}, sim3: {sim3.mean().item():.6f}")
    # # 根据每行的最大相似度值选择对应的权重矩阵行
    # # sim1 = output_dot_similarity(w1, w_original,dim=0)
    # # sim2 = output_dot_similarity(w2, w_original,dim=0)
    # # sim3 = output_dot_similarity(w3, w_original,dim=0)
    # col_weights = torch.norm(o_w, p=2, dim=0)
    # col_weights = col_weights.unsqueeze(0).expand(w_original.shape[0], -1)  # shape: (num_rows, num_cols)
    # row_weights = row_weights.unsqueeze(1).expand(-1, w_original.shape[1]) # shape: (num_rows, num_cols)
    sim1 = mse_sim(w1/current_awq_x_s, org_w,torch.ones_like(org_w)).view(-1)
    sim2 = mse_sim(w2, org_w,torch.ones_like(org_w)).view(-1)
    #sim3 = mse_sim(w3, w_original,torch.ones_like(w_original)).view(-1)
    # f_new_w = select_column_by_max_similarity([sim1, sim2, sim3], [w1, w2, w3])
    f_new_w, masked_w_list, mask_list = select_rows_by_max_similarity0124([sim1, sim2], [w1.view(-1), w2.view(-1)])
    f_new_w = f_new_w.reshape(org_w.shape)
    # masked_w_list[0] 和 masked_w_list[1] 分别是掩码后的 w1 和 w2（展平后的）
    # 如果需要，可以 reshape 回原始形状：
    masked_w1 = masked_w_list[0].reshape(org_w.shape)
    masked_w2 = masked_w_list[1].reshape(org_w.shape)

    

    mask1 = mask_list[0].reshape(org_w.shape)
    mask2 = mask_list[1].reshape(org_w.shape)

    assert (f_new_w == masked_w1 + masked_w2).all(), "f_new_w != masked_w1 + masked_w2"

    # masked_w2 = gptq_unscaled_w * mask2
    if os.environ.get("AWQ_GPTQ_QUANT_DEBUG", "0") == "1":
        print("mask1.sum():", mask1.sum())
        print("mask2.sum():", mask2.sum())

    # # 将三个相似度值堆叠，用于后续统计
    #sim_stack = torch.stack([sim1, sim2, sim3], dim=0)  # shape: (3, num_rows)
    
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
    # max_values = sim_stack.max(dim=0)[0]  # shape: (num_rows,)，每行的最大值
    # is_max_0 = (sim_stack[0] == max_values).float()  # shape: (num_rows,)，sim1是否等于最大值
    # is_max_1 = (sim_stack[1] == max_values).float()  # shape: (num_rows,)，sim2是否等于最大值
    # is_max_2 = (sim_stack[2] == max_values).float()  # shape: (num_rows,)，sim3是否等于最大值
    
    # 计算每行的重要性权重（使用 w_original 每行的L2范数）
      # shape: (num_rows,)
   
    #row_weights = torch.norm(o_w, p=2, dim=0)  # shape: (num_rows,)
    # row_weights = o_w.abs().view(-1)
    # #row_weights = torch.ones_like(row_weights)
    # # 使用行重要性权重进行加权统计
    # col_weights = torch.norm(w_original, p=2, dim=0)
    # col_weights = col_weights.unsqueeze(0).expand(w_original.shape[0], -1) 
    # row_weights = torch.norm(w_original, p=2, dim=-1)
    # row_weights = row_weights.unsqueeze(1).expand(-1, w_original.shape[1]) # shape: (num_rows, num_cols)
    # row_weights = row_weights.reshape(-1)
    # row_weights = torch.ones_like(row_weights)
    # count_0 = (is_max_0 * row_weights).sum().float()  # 加权统计sim1等于最大值的行
    # count_1 = (is_max_1 * row_weights).sum().float()  # 加权统计sim2等于最大值的行
    # count_2 = (is_max_2 * row_weights).sum().float()  # 加权统计sim3等于最大值的行
    # dot_values = torch.stack([count_0, count_1, count_2])  # shape: (3,)
    # print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    # max_dot_idx = dot_values.argmax().item()  # 标量值：0, 1, 或 2
    
    # # 更新全局统计
    # max_dot_idx_stats['total'] += 1
    # if max_dot_idx == 0:
    #     max_dot_idx_stats['count_0'] += 1
    #     new_w = w1
    #     #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    # elif max_dot_idx == 1:
    #     max_dot_idx_stats['count_1'] += 1
    #     #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=1")
    #     new_w = w2
    #     #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    # else:
    #     max_dot_idx_stats['count_2'] += 1
    #     #print(f"dot1={dot1.item():.6f}, dot2={dot2.item():.6f}, dot3={dot3.item():.6f}, max_idx=0")
    #     #print(f"dot_values (counts in max_idx): [{count_0.item():.0f}, {count_1.item():.0f}, {count_2.item():.0f}], max_idx={dot_values.argmax().item()}")
    #     #print(f"dot_values (counts in max_idx): [{count_0.item():.2f}, {count_1.item():.2f}, {count_2.item():.2f}], max_idx={dot_values.argmax().item()}")
  
    #     new_w = w3
    
    #new_w = gptq_w.clone()
    #new_w = w1
    new_w = f_new_w
    w = new_w.reshape(org_w_shape)
    awq_w = masked_w1
    gptq_w = masked_w2
    # gptq_w = torch.zeros_like(gptq_w)

    #awq_w = torch.zeros_like(awq_w)
    #gptq_w = gptq_unscaled_w 

    
    
    awq_scales = scales.view(w.shape[0], -1)
    awq_zeros = zeros.view(w.shape[0], -1)

    if os.environ.get("AWQ_GPTQ_QUANT_DEBUG", "0") == "1":
        print(f"awq_scales.shape: {awq_scales.shape}")
        print(f"awq_zeros.shape: {awq_zeros.shape}")
        print(f"gptq_scales.shape: {gptq_scales.shape}")
        print(f"gptq_zeros.shape: {gptq_zeros.shape}")

    if tuple(gptq_scales.shape) != tuple(awq_scales.shape):
        raise ValueError(
            f"Incomplete GPTQ scales: got {tuple(gptq_scales.shape)}, "
            f"expected {tuple(awq_scales.shape)}. Re-export with --static-groups."
        )
    if tuple(gptq_zeros.shape) != tuple(awq_zeros.shape):
        raise ValueError(
            f"Incomplete GPTQ zeros: got {tuple(gptq_zeros.shape)}, "
            f"expected {tuple(awq_zeros.shape)}. Re-export with --static-groups."
        )
    gptq_scales = gptq_scales.clone()
    gptq_zeros = gptq_zeros.clone()
    if get_scale_zp:
        # full_w contains the element selected from the AWQ/GPTQ candidates;
        # mask1==1 selects AWQ and mask1==0 selects GPTQ.  The real backend
        # packs this into one int4 matrix and keeps the 1-bit selector.
        return (
            awq_w,
            awq_scales,
            awq_zeros,
            gptq_w,
            gptq_scales,
            gptq_zeros,
            w,
            mask1.bool(),
        )
        #return awq_w, gptq_scales, gptq_zeros, gptq_w, awq_scales, awq_zeros
        
    else:
        return w



VALID_MIX_SOURCES = ("awq", "gptq", "omni", "w_spqr", "rtn")
MIX_SOURCE_ALIASES = {
    "spqr": "w_spqr",
    "spqr_weight": "w_spqr",
    "spqr_weights": "w_spqr",
    "omniquant": "omni",
}


def _normalize_mix_sources(mix_sources, quant_type=None):
    if mix_sources is None or str(mix_sources).strip() == "":
        if quant_type in VALID_MIX_SOURCES:
            return [quant_type]
        return None

    sources = []
    for item in str(mix_sources).split(","):
        source = item.strip().lower()
        if not source:
            continue
        source = MIX_SOURCE_ALIASES.get(source, source)
        if source not in VALID_MIX_SOURCES:
            raise ValueError(f"Unknown mix source {source}; choose from {VALID_MIX_SOURCES}")
        if source not in sources:
            sources.append(source)
    if not sources:
        raise ValueError("--mix_sources did not contain any valid source")
    return sources


def _default_gptq_param_path(model_path, w_bit):
    if "Llama-7b" in model_path:
        if w_bit == 3:
            return "/home/zch/project/临时存放/llama-7b-w3-g128-gptq_params.pt"
        return "/home/zch/project/临时存放/llama-7b-w4-g128-gptq_params.pt"
    if "llama-2-7b" in model_path:
        if w_bit == 3:
            return "/home/zch/project/awq_code/models/gptq3bit/gptq_llama2_pilezch.pt"
        return "/home/zch/project/awq_code/models/gptq4bit/gptq_llama2_pilezch.pt"
    if "Llama-13b" in model_path:
        if w_bit == 3:
            return "/home/zch/project/awq_code/models/gptq3bit/gptq_llama_13b_pilezch.pt"
        return "/home/zch/project/awq_code/models/gptq4bit/gptq_llama_13_pilezch.pt"
    if "Llama2-13b" in model_path:
        if w_bit == 3:
            return "/home/zch/project/awq_code/models/gptq3bit/gptq_llama2_13b_pilezch.pt"
        return "/home/zch/project/awq_code/models/gptq4bit/gptq_llama2_13b_pilezch.pt"
    if "Meta-Llama-3-8B" in model_path:
        return "/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_all_3.pt"
    lower_path = str(model_path).lower()
    if "opt-6.7b" in lower_path:
        if w_bit == 3:
            return "/mnt/dfq/gptq_output/opt-6.7b-w3-g128-gptq_params.pt"
        return "/mnt/dfq/gptq_output/opt-6.7b-w4-g128-gptq_params.pt"
    if "opt-13b" in lower_path:
        if w_bit == 3:
            return "/mnt/dfq/gptq_output/opt-13b-w3-g128-gptq_params.pt"
        return "/mnt/dfq/gptq_output/opt-13b-w4-g128-gptq_params.pt"
    return None


def _layer_name_prefix(model):
    if model.__class__.__name__ == "OPTForCausalLM":
        return "model.decoder.layers"
    return "model.layers"


def _format_layer_name(model, layer_idx, linear_name):
    return f"{_layer_name_prefix(model)}.{layer_idx}.{linear_name}"


def _layer_name_aliases(full_name):
    aliases = [full_name]
    if full_name.startswith("model.layers."):
        aliases.append("model.decoder.layers." + full_name[len("model.layers."):])
    elif full_name.startswith("model.decoder.layers."):
        aliases.append("model.layers." + full_name[len("model.decoder.layers."):])
    if full_name.startswith("model."):
        aliases.append(full_name[len("model."):])
    return list(dict.fromkeys(aliases))


def _get_record_by_alias(mapping, full_name):
    if mapping is None:
        return None
    for name in _layer_name_aliases(full_name):
        if name in mapping:
            return mapping[name]
    return None


def _stack_spqr_group_params(values):
    if torch.is_tensor(values):
        return values.detach().cpu()
    return torch.stack([v.detach().cpu() for v in values], dim=0)


def _dequantize_spqr_legacy_tensor(params):
    qweight = params["quant_weights"].float().cpu()
    out_dim, in_dim = tuple(params["weight_shape"])
    groupsize = int(params.get("groupsize", in_dim))
    dtype = params.get("save_float_dtype", torch.float16)

    scales = _stack_spqr_group_params(params["quant_layer_scale"]).float()
    zeros = _stack_spqr_group_params(params["quant_layer_zeros"]).float()
    if scales.dim() == 2:
        scales = scales.unsqueeze(-1)
    if zeros.dim() == 2:
        zeros = zeros.unsqueeze(-1)

    weight = torch.empty((out_dim, in_dim), dtype=torch.float32)
    for group_idx, start in enumerate(range(0, in_dim, groupsize)):
        end = min(start + groupsize, in_dim)
        scale = scales[group_idx]
        zero = zeros[group_idx]
        weight[:, start:end] = scale * (qweight[:, start:end] - zero)

    outliers = params.get("outliers_matrix")
    if outliers is not None:
        dense_outliers = outliers.to_dense().float() if outliers.is_sparse else outliers.float()
        weight = weight * (dense_outliers == 0) + dense_outliers

    perm = params.get("perm")
    if perm is not None:
        invperm = torch.argsort(perm.long().cpu())
        weight = weight[:, invperm]

    return weight.to(dtype)


def _load_spqr_legacy_dir(path):
    result = {}
    tensor_paths = []
    for layer_name in sorted(os.listdir(path), key=lambda x: int(x) if x.isdigit() else 10**9):
        layer_dir = os.path.join(path, layer_name)
        if not layer_name.isdigit() or not os.path.isdir(layer_dir):
            continue
        for tensor_name in sorted(os.listdir(layer_dir)):
            tensor_path = os.path.join(layer_dir, tensor_name)
            if os.path.isfile(tensor_path):
                tensor_paths.append((layer_name, tensor_name, tensor_path))

    for layer_name, tensor_name, tensor_path in tqdm(
        tensor_paths,
        desc=f"loading {os.path.basename(path)} SpQR tensors",
        leave=False,
    ):
        params = torch.load(tensor_path, map_location="cpu")
        full_name = f"model.layers.{int(layer_name)}.{tensor_name}"
        scales = _stack_spqr_group_params(params["quant_layer_scale"]).squeeze(-1).transpose(0, 1).contiguous()
        zeros = _stack_spqr_group_params(params["quant_layer_zeros"]).squeeze(-1).transpose(0, 1).contiguous()
        result[full_name] = {
            "weight": _dequantize_spqr_legacy_tensor(params),
            "scale": scales,
            "zero": zeros,
            "source": "w_spqr",
        }
    print(f"Loaded {len(result)} SpQR layers from legacy directory {path}")
    return result


def _load_quant_param_map(path, source):
    if path is None:
        return None
    if source == "w_spqr" and os.path.isdir(path):
        return _load_spqr_legacy_dir(path)
    start_time = time.time()
    file_size = os.path.getsize(path) / (1024 ** 3) if os.path.isfile(path) else None
    size_msg = f" ({file_size:.2f} GB)" if file_size is not None else ""
    tqdm.write(f"Loading {source} params from {path}{size_msg} ...")
    params = torch.load(path, map_location="cpu")
    layer_count = len(params) if hasattr(params, "__len__") else "unknown"
    tqdm.write(f"Loaded {source} params from {path}; layers={layer_count}; elapsed={time.time() - start_time:.1f}s")
    return params


def _get_layer_record(param_map, full_name, source):
    if param_map is None:
        raise ValueError(f"No parameter map was loaded for source {source}")
    record = _get_record_by_alias(param_map, full_name)
    if record is None and isinstance(param_map.get("weight"), dict):
        alt_name = full_name[len("model."):] if full_name.startswith("model.") else full_name
        weight_map = param_map["weight"]
        weight = weight_map.get(full_name, weight_map.get(alt_name))
        if weight is not None:
            record = {"weight": weight}
            for field in ("scale", "zero"):
                field_map = param_map.get(field)
                if isinstance(field_map, dict):
                    record[field] = field_map.get(full_name, field_map.get(alt_name))
    if record is None:
        raise KeyError(f"Layer {full_name} not found in {source} params")
    if torch.is_tensor(record):
        record = {"weight": record}
    if "weight" not in record:
        raise KeyError(f"Layer {full_name} in {source} params has no weight field")
    return record


def _fake_awq_candidate(weight, current_awq_x_s, n_bit, zero_point=True, q_group_size=-1):
    org_shape = weight.shape
    scale_x = current_awq_x_s.view(1, -1).to(device=weight.device, dtype=weight.dtype)
    scaled_weight = weight * scale_x

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        q_weight = scaled_weight.reshape(-1, q_group_size)
    else:
        q_weight = scaled_weight.reshape(scaled_weight.shape[0], -1)

    if zero_point:
        max_val = q_weight.amax(dim=1, keepdim=True)
        min_val = q_weight.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        q_dequant = (torch.clamp(torch.round(q_weight / scales) + zeros, min_int, max_int) - zeros) * scales
    else:
        max_val = q_weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        q_dequant = torch.clamp(torch.round(q_weight / scales), min_int, max_int) * scales

    return q_dequant.reshape(org_shape) / scale_x


def _real_awq_gptq_candidates(
    original_weight,
    current_awq_x_s,
    gptq_weight,
    gptq_scales,
    gptq_zeros,
    n_bit,
    zero_point=True,
    q_group_size=-1,
    gptq_g_idx=None,
):
    """Build one canonical AWQ/GPTQ representation for fake and real paths.

    AWQ is stored in the ``W * x_s`` domain and GPTQ stays in the original
    weight domain.  All qparams are first reduced to the FP16 representation
    that WQLinear0124 actually persists.  The returned ``effective_weight`` is
    therefore the dense fake-quant view of the exact same candidates and mask
    that the real INT4 checkpoint packs.
    """
    if not zero_point:
        raise NotImplementedError(
            "The fused AWQ/GPTQ real backend requires zero_point"
        )
    if n_bit != 4:
        raise NotImplementedError(
            "The fused AWQ/GPTQ real backend supports 4-bit weights only"
        )
    if q_group_size <= 0:
        raise ValueError(
            "The fused AWQ/GPTQ real backend requires a positive q_group_size"
        )
    if (
        original_weight.dim() != 2
        or original_weight.shape[-1] % q_group_size != 0
    ):
        raise ValueError(
            "AWQ/GPTQ real quantization requires a 2D weight whose input "
            f"dimension is divisible by q_group_size; got "
            f"{tuple(original_weight.shape)} and q_group_size={q_group_size}"
        )

    org_weight = original_weight.detach().clone()
    out_features, in_features = org_weight.shape
    num_groups = in_features // q_group_size
    expected_param_shape = (out_features, num_groups)
    if tuple(gptq_weight.shape) != tuple(org_weight.shape):
        raise ValueError(
            f"GPTQ weight shape {tuple(gptq_weight.shape)} does not match "
            f"model weight shape {tuple(org_weight.shape)}"
        )
    if tuple(gptq_scales.shape) != expected_param_shape:
        raise ValueError(
            f"Incomplete GPTQ scales: got {tuple(gptq_scales.shape)}, "
            f"expected {expected_param_shape}. Re-export GPTQ with all "
            "dynamic/static group parameters."
        )
    if tuple(gptq_zeros.shape) != expected_param_shape:
        raise ValueError(
            f"Incomplete GPTQ zeros: got {tuple(gptq_zeros.shape)}, "
            f"expected {expected_param_shape}. Re-export GPTQ with all "
            "dynamic/static group parameters."
        )

    if gptq_g_idx is None:
        normalized_gptq_g_idx = torch.arange(
            in_features, device=org_weight.device, dtype=torch.long
        ) // q_group_size
    else:
        normalized_gptq_g_idx = torch.as_tensor(
            gptq_g_idx, device=org_weight.device, dtype=torch.long
        ).reshape(-1)
        if normalized_gptq_g_idx.numel() != in_features:
            raise ValueError(
                f"GPTQ g_idx has {normalized_gptq_g_idx.numel()} elements, "
                f"expected {in_features}"
            )
        if (
            (normalized_gptq_g_idx < 0).any()
            or (normalized_gptq_g_idx >= num_groups).any()
        ):
            min_group = normalized_gptq_g_idx.min().item()
            max_group = normalized_gptq_g_idx.max().item()
            raise ValueError(
                "GPTQ g_idx contains an invalid group id: "
                f"range=[{min_group}, {max_group}], valid=[0, {num_groups - 1}]"
            )

    scale_x = current_awq_x_s.reshape(1, -1).to(
        device=org_weight.device, dtype=org_weight.dtype
    )
    if scale_x.shape[-1] != in_features:
        raise ValueError(
            f"AWQ activation scale has {scale_x.shape[-1]} elements, "
            f"expected {in_features}"
        )
    if not torch.isfinite(scale_x).all() or (scale_x <= 0).any():
        raise ValueError("AWQ activation scales must be finite and positive")

    qmin = 0
    qmax = 2**n_bit - 1

    # AWQ candidate: quantize W*x_s groupwise, then emulate the FP16 scale and
    # scaled-zero affine dequantization used by the CUDA kernel.
    awq_grouped = (org_weight * scale_x).reshape(
        out_features, num_groups, q_group_size
    )
    awq_max = awq_grouped.amax(dim=-1)
    awq_min = awq_grouped.amin(dim=-1)
    awq_scales = ((awq_max - awq_min).clamp(min=1e-5) / qmax).to(
        org_weight.dtype
    )
    awq_zeros = (-torch.round(awq_min / awq_scales)).clamp_(qmin, qmax).to(
        org_weight.dtype
    )
    awq_int = torch.round(
        awq_grouped / awq_scales.unsqueeze(-1) + awq_zeros.unsqueeze(-1)
    ).clamp_(qmin, qmax)
    awq_bias = -(
        awq_scales.float() * awq_zeros.float()
    ).to(org_weight.dtype)
    awq_dequant_scaled = (
        awq_int.float() * awq_scales.float().unsqueeze(-1)
        + awq_bias.float().unsqueeze(-1)
    ).to(org_weight.dtype).reshape_as(org_weight)
    awq_effective = (awq_dequant_scaled / scale_x).to(org_weight.dtype)

    # GPTQ candidate: recover the original integer codes with the exported
    # FP32 qparams, then dequantize with the FP16 qparams persisted by the real
    # checkpoint.  This makes fake and real compare the same representable
    # GPTQ tensor rather than two subtly different round-trip variants.
    gptq_reference = gptq_weight.to(
        device=org_weight.device, dtype=org_weight.dtype
    )
    source_scales = gptq_scales.to(device=org_weight.device).float()
    source_zeros = gptq_zeros.to(device=org_weight.device).float()
    if not torch.isfinite(source_scales).all() or (source_scales <= 0).any():
        raise ValueError("GPTQ scales must be finite and positive")
    rounded_source_zeros = torch.round(source_zeros)
    if not torch.equal(source_zeros, rounded_source_zeros):
        max_zero_error = (source_zeros - rounded_source_zeros).abs().max().item()
        raise ValueError(
            "GPTQ zero points must be integer-valued; "
            f"maximum fractional error is {max_zero_error:.6g}"
        )
    if ((rounded_source_zeros < qmin) | (rounded_source_zeros > qmax)).any():
        raise ValueError("GPTQ zero points fall outside the 4-bit range")

    # Dynamic-groups + act-order GPTQ does not use the physical column's
    # ``k // group_size``.  ``g_idx[k]`` maps each original-domain input
    # column back to the qparam row selected while GPTQ quantized it.
    selected_source_scales = source_scales[:, normalized_gptq_g_idx]
    selected_source_zeros = rounded_source_zeros[:, normalized_gptq_g_idx]
    gptq_int = torch.round(
        gptq_reference.float() / selected_source_scales
        + selected_source_zeros
    )
    out_of_range = ((gptq_int < qmin) | (gptq_int > qmax)).sum().item()
    if out_of_range:
        raise ValueError(
            "GPTQ weight/qparams produce values outside the 4-bit range: "
            f"out_of_range={out_of_range}. Verify that weight, scale, zero "
            "and g_idx came from the same fixed GPTQ export."
        )
    gptq_int.clamp_(qmin, qmax)
    persistent_gptq_scales = source_scales.to(org_weight.dtype)
    persistent_gptq_zeros = rounded_source_zeros.to(org_weight.dtype)
    gptq_bias = -(
        persistent_gptq_scales.float() * persistent_gptq_zeros.float()
    ).to(org_weight.dtype)
    selected_persistent_gptq_scales = persistent_gptq_scales.float()[
        :, normalized_gptq_g_idx
    ]
    selected_gptq_bias = gptq_bias.float()[:, normalized_gptq_g_idx]
    gptq_dequant = (
        gptq_int * selected_persistent_gptq_scales + selected_gptq_bias
    ).to(org_weight.dtype)

    max_error = (gptq_dequant - gptq_reference).abs().max().item()
    max_scale = persistent_gptq_scales.abs().max().item()
    tolerance = max(2e-3, max_scale * 0.02)
    if max_error > tolerance:
        raise ValueError(
            "GPTQ weight/qparams do not round-trip for FP16 INT4 packing: "
            f"max_error={max_error:.6g}, tolerance={tolerance:.6g}. "
            "Verify that weight, qparams and g_idx belong to the same export."
        )

    awq_sim = mse_sim(awq_effective, org_weight, torch.ones_like(org_weight))
    gptq_sim = mse_sim(gptq_dequant, org_weight, torch.ones_like(org_weight))
    awq_mask = awq_sim >= gptq_sim
    gptq_mask = ~awq_mask

    # WQLinear packs the two branches in their respective storage domains.
    full_weight = torch.where(awq_mask, awq_dequant_scaled, gptq_dequant)
    masked_awq = torch.where(awq_mask, awq_dequant_scaled, 0)
    masked_gptq = torch.where(gptq_mask, gptq_dequant, 0)
    # Dense fake inference consumes original-domain activations and therefore
    # uses the effective AWQ tensor after undoing x_s.
    effective_weight = torch.where(awq_mask, awq_effective, gptq_dequant)

    awq_count = awq_mask.sum().item()
    print(
        "[awq/gptq mix] selected elements: "
        f"awq={awq_count}, gptq={awq_mask.numel() - awq_count}"
    )
    return (
        masked_awq,
        awq_scales,
        awq_zeros,
        masked_gptq,
        persistent_gptq_scales,
        persistent_gptq_zeros,
        full_weight,
        awq_mask,
        effective_weight,
        normalized_gptq_g_idx.to(torch.int32),
    )


def _real_awq_rtn_candidates(
    weight,
    current_awq_x_s,
    n_bit,
    zero_point=True,
    q_group_size=-1,
):
    """Build a fused AWQ/RTN representation without external quant params.

    The AWQ candidate is kept in the scaled-weight domain because the fused
    GEMM consumes ``input / current_awq_x_s``.  RTN stays in the original
    weight domain; the masked CUDA path multiplies RTN-selected lanes by the
    corresponding input scale before MMA.

    Returns the same eight tensors consumed by ``WQLinear0124.from_linear``:
    masked AWQ weight/params, masked RTN weight/params, the selected mixed
    weight (whose two branches intentionally live in different domains), and
    a True-for-AWQ selector mask.
    """
    if not zero_point:
        raise NotImplementedError("The fused AWQ/RTN real backend requires zero_point")
    if q_group_size <= 0:
        raise ValueError("The fused AWQ/RTN real backend requires a positive q_group_size")
    if weight.dim() != 2 or weight.shape[-1] % q_group_size != 0:
        raise ValueError(
            "AWQ/RTN real quantization requires a 2D weight whose input "
            f"dimension is divisible by q_group_size; got {tuple(weight.shape)} "
            f"and q_group_size={q_group_size}"
        )

    org_weight = weight.detach().clone()
    scale_x = current_awq_x_s.reshape(1, -1).to(
        device=org_weight.device, dtype=org_weight.dtype
    )
    if scale_x.shape[-1] != org_weight.shape[-1]:
        raise ValueError(
            f"AWQ activation scale has {scale_x.shape[-1]} elements, "
            f"expected {org_weight.shape[-1]}"
        )

    # Match _fake_awq_candidate exactly so fake and real choose the same AWQ
    # candidate before the latter is packed into int4.
    awq_scaled = org_weight * scale_x
    awq_grouped = awq_scaled.reshape(-1, q_group_size)
    qmin = 0
    qmax = 2**n_bit - 1
    awq_max = awq_grouped.amax(dim=1, keepdim=True)
    awq_min = awq_grouped.amin(dim=1, keepdim=True)
    awq_scales_flat = (awq_max - awq_min).clamp(min=1e-5) / qmax
    awq_zeros_flat = (-torch.round(awq_min / awq_scales_flat)).clamp_(qmin, qmax)
    awq_int = torch.round(awq_grouped / awq_scales_flat) + awq_zeros_flat
    awq_dequant_scaled = (
        awq_int.clamp_(qmin, qmax) - awq_zeros_flat
    ) * awq_scales_flat
    awq_dequant_scaled = awq_dequant_scaled.reshape_as(org_weight)
    awq_scales = awq_scales_flat.view(org_weight.shape[0], -1)
    awq_zeros = awq_zeros_flat.view(org_weight.shape[0], -1)
    awq_effective = awq_dequant_scaled / scale_x

    # RTN is generated locally from the original-domain weight.  Its qparams
    # therefore contain every contiguous K/group_size group and need no
    # external metadata, expansion, act-order permutation, or g_idx table.
    rtn_dequant, rtn_scales, rtn_zeros = rtn_quant_dequant_group_asym(
        org_weight.clone(), bits=n_bit, group_size=q_group_size
    )
    rtn_dequant = rtn_dequant[:, : org_weight.shape[-1]].to(org_weight.dtype)
    rtn_scales = rtn_scales.to(device=org_weight.device, dtype=org_weight.dtype)
    rtn_zeros = rtn_zeros.to(device=org_weight.device, dtype=org_weight.dtype)

    awq_sim = mse_sim(
        awq_effective, org_weight, torch.ones_like(org_weight)
    )
    rtn_sim = mse_sim(
        rtn_dequant, org_weight, torch.ones_like(org_weight)
    )
    awq_mask = awq_sim >= rtn_sim
    rtn_mask = ~awq_mask

    full_weight = torch.where(awq_mask, awq_dequant_scaled, rtn_dequant)
    masked_awq = torch.where(awq_mask, awq_dequant_scaled, 0)
    masked_rtn = torch.where(rtn_mask, rtn_dequant, 0)

    awq_count = awq_mask.sum().item()
    print(
        "[real mix] selected elements: "
        f"awq={awq_count}, rtn={awq_mask.numel() - awq_count}"
    )
    return (
        masked_awq,
        awq_scales,
        awq_zeros,
        masked_rtn,
        rtn_scales,
        rtn_zeros,
        full_weight,
        awq_mask,
    )


def _real_gptq_rtn_candidates(
    original_weight,
    gptq_weight,
    gptq_scales,
    gptq_zeros,
    n_bit,
    zero_point=True,
    q_group_size=-1,
):
    """Build same-domain GPTQ/RTN candidates for the fused real backend."""
    if not zero_point:
        raise NotImplementedError("The fused GPTQ/RTN real backend requires zero_point")
    if q_group_size <= 0:
        raise ValueError("The fused GPTQ/RTN real backend requires a positive q_group_size")
    if original_weight.dim() != 2 or original_weight.shape[-1] % q_group_size != 0:
        raise ValueError(
            "GPTQ/RTN real quantization requires a 2D weight whose input "
            f"dimension is divisible by q_group_size; got {tuple(original_weight.shape)} "
            f"and q_group_size={q_group_size}"
        )

    org_weight = original_weight.detach().clone()
    out_features, in_features = org_weight.shape
    num_groups = in_features // q_group_size
    expected_param_shape = (out_features, num_groups)
    if tuple(gptq_weight.shape) != tuple(org_weight.shape):
        raise ValueError(
            f"GPTQ weight shape {tuple(gptq_weight.shape)} does not match "
            f"model weight shape {tuple(org_weight.shape)}"
        )
    if tuple(gptq_scales.shape) != expected_param_shape:
        raise ValueError(
            f"Incomplete GPTQ scales: got {tuple(gptq_scales.shape)}, "
            f"expected {expected_param_shape}. Re-export GPTQ with --static-groups."
        )
    if tuple(gptq_zeros.shape) != expected_param_shape:
        raise ValueError(
            f"Incomplete GPTQ zeros: got {tuple(gptq_zeros.shape)}, "
            f"expected {expected_param_shape}. Re-export GPTQ with --static-groups."
        )

    # Reconstruct exactly what the real FP16-qparam/int4 checkpoint can store.
    gptq_weight = gptq_weight.to(device=org_weight.device, dtype=org_weight.dtype)
    # Keep the exported fake-quant tensor as the selection reference.  Its
    # qparams were applied in FP32 by GPTQ before the result was stored as
    # FP16.  Reconstructing it first with persistent FP16 qparams changes a
    # small number of values; because GPTQ and RTN are frequently near-tied,
    # that otherwise flips roughly 2--3% of selector bits on Llama-7B.  The
    # real checkpoint must use exactly the same winner mask as fake quant.
    gptq_selection_reference = gptq_weight
    gptq_scales = gptq_scales.to(device=org_weight.device, dtype=org_weight.dtype)
    gptq_zeros = gptq_zeros.to(device=org_weight.device, dtype=org_weight.dtype)
    grouped_gptq = gptq_weight.reshape(out_features, num_groups, q_group_size)
    grouped_scales = gptq_scales.unsqueeze(-1)
    grouped_zeros = gptq_zeros.unsqueeze(-1)
    qmax = 2**n_bit - 1
    raw_gptq_int = torch.round(grouped_gptq / grouped_scales + grouped_zeros)
    out_of_range = ((raw_gptq_int < 0) | (raw_gptq_int > qmax)).sum().item()
    gptq_dequant = (
        raw_gptq_int.clamp(0, qmax) - grouped_zeros
    ) * grouped_scales
    gptq_dequant = gptq_dequant.reshape_as(org_weight)

    # Reject old [out, 1]-style or otherwise inconsistent exports before they
    # can silently create another high-PPL real checkpoint.
    max_error = (gptq_dequant - gptq_weight).abs().max().item()
    max_scale = gptq_scales.abs().max().item()
    tolerance = max(2e-3, max_scale * 0.02)
    if out_of_range or max_error > tolerance:
        raise ValueError(
            "GPTQ weight/qparams do not round-trip for real int4 packing: "
            f"out_of_range={out_of_range}, max_error={max_error:.6g}, "
            f"tolerance={tolerance:.6g}. Re-export with the fixed GPTQ code "
            "and --static-groups."
        )

    rtn_dequant, rtn_scales, rtn_zeros = rtn_quant_dequant_group_asym(
        org_weight.clone(), bits=n_bit, group_size=q_group_size
    )
    rtn_dequant = rtn_dequant[:, :in_features].to(org_weight.dtype)
    rtn_scales = rtn_scales.to(device=org_weight.device, dtype=org_weight.dtype)
    rtn_zeros = rtn_zeros.to(device=org_weight.device, dtype=org_weight.dtype)

    gptq_sim = mse_sim(
        gptq_selection_reference, org_weight, torch.ones_like(org_weight)
    )
    rtn_sim = mse_sim(rtn_dequant, org_weight, torch.ones_like(org_weight))
    gptq_mask = gptq_sim >= rtn_sim
    rtn_mask = ~gptq_mask
    full_weight = torch.where(gptq_mask, gptq_dequant, rtn_dequant)
    masked_gptq = torch.where(gptq_mask, gptq_dequant, 0)
    masked_rtn = torch.where(rtn_mask, rtn_dequant, 0)

    gptq_count = gptq_mask.sum().item()
    print(
        "[real mix] selected elements: "
        f"gptq={gptq_count}, rtn={gptq_mask.numel() - gptq_count}"
    )
    return (
        masked_gptq,
        gptq_scales,
        gptq_zeros,
        masked_rtn,
        rtn_scales,
        rtn_zeros,
        full_weight,
        gptq_mask,
    )


def pseudo_quantize_tensor_mix(
    weight,
    sources,
    external_candidates=None,
    current_awq_x_s=None,
    n_bit=8,
    zero_point=True,
    q_group_size=-1,
):
    org_weight = weight.detach().clone()
    candidates = []
    candidate_names = []
    external_candidates = external_candidates or {}

    for source in sources:
        if source == "awq":
            if current_awq_x_s is None:
                raise ValueError("Source awq requires --load_awq so awq_x_s is available")
            cand = _fake_awq_candidate(org_weight, current_awq_x_s, n_bit, zero_point, q_group_size)
        elif source == "rtn":
            cand, _, _ = rtn_quant_dequant_group_asym(org_weight.clone(), bits=n_bit, group_size=q_group_size)
        else:
            if source not in external_candidates:
                raise ValueError(f"Missing external candidate for source {source}")
            cand = external_candidates[source]

        cand = cand.to(device=org_weight.device, dtype=org_weight.dtype)
        assert cand.shape == org_weight.shape, f"{source} shape {cand.shape} != {org_weight.shape}"
        candidates.append(cand)
        candidate_names.append(source)

    if len(candidates) == 1:
        print(f"[mix] only source {candidate_names[0]} selected")
        return candidates[0]

    sims = [mse_sim(cand, org_weight, torch.ones_like(org_weight)).view(-1) for cand in candidates]
    sim_stack = torch.stack(sims, dim=0)
    winner = sim_stack.argmax(dim=0)
    counts = [(winner == idx).sum().item() for idx in range(len(candidate_names))]
    summary = ", ".join(f"{name}={count}" for name, count in zip(candidate_names, counts))
    print(f"[mix] selected elements: {summary}")

    mixed = select_rows_by_max_similarity(sims, [cand.view(-1) for cand in candidates])
    return mixed.reshape_as(org_weight)




@torch.no_grad()
def pseudo_quantize_model_weight722(
    model,
    w_bit,
    model_path,
    quant_type,
    q_config,
    th_t_dict={},
    awq_results=None,
    mix_sources=None,
    gptq_params=None,
    omni_params=None,
    spqr_params=None,
    keep_on_cpu=False,
):
    from .pre_quant import get_blocks, get_named_linears

    normalized_sources = _normalize_mix_sources(mix_sources, quant_type)
    if mix_sources is not None:
        print(f"Using mix sources: {normalized_sources}")
        param_maps = {}
        sources_to_load = [source for source in normalized_sources if source in ("gptq", "omni", "w_spqr")]
        with tqdm(sources_to_load, desc="loading mix source params") as load_pbar:
            for source in load_pbar:
                load_pbar.set_postfix_str(source)
                if source == "gptq":
                    gptq_path = gptq_params or _default_gptq_param_path(model_path, w_bit)
                    if gptq_path is None:
                        raise ValueError("Source gptq requires --gptq_params for this model")
                    param_maps["gptq"] = _load_quant_param_map(gptq_path, "gptq")
                elif source == "omni":
                    if omni_params is None:
                        raise ValueError("Source omni requires --omni_params")
                    param_maps["omni"] = _load_quant_param_map(omni_params, "omni")
                elif source == "w_spqr":
                    if spqr_params is None:
                        raise ValueError("Source w_spqr requires --spqr_params")
                    param_maps["w_spqr"] = _load_quant_param_map(spqr_params, "w_spqr")

        if "awq" in normalized_sources and (awq_results is None or "awq_x_s" not in awq_results):
            raise ValueError("Source awq requires --load_awq so awq_x_s is available")

        layers = get_blocks(model)
        tmp_i = 0
        legacy_gptq_weight_only_warned = False
        layer_linears = [(i, get_named_linears(layer)) for i, layer in enumerate(layers)]
        total_linears = sum(len(named_linears) for _, named_linears in layer_linears)
        with tqdm(total=total_linears, desc="mixed pseudo weight quantization") as mix_pbar:
            for i, named_linears in layer_linears:
                for n, m in named_linears.items():
                    full_name = _format_layer_name(model, i, n)
                    mix_pbar.set_postfix_str(full_name)
                    if not keep_on_cpu:
                        m.cuda()

                    external_candidates = {}
                    external_records = {}
                    for source in normalized_sources:
                        if source in ("gptq", "omni", "w_spqr"):
                            record = _get_layer_record(param_maps[source], full_name, source)
                            external_records[source] = record
                            external_candidates[source] = record["weight"].to(device=m.weight.device, dtype=m.weight.dtype)

                    current_awq_x_s = None
                    if "awq" in normalized_sources:
                        current_awq_x_s = _get_record_by_alias(awq_results["awq_x_s"], full_name)
                        if current_awq_x_s is None:
                            # Some LLaMA-like GQA layers do not have an AWQ scale entry.
                            # Use identity scale so the AWQ candidate remains well-defined.
                            current_awq_x_s = torch.ones(m.weight.shape[-1], device=m.weight.device, dtype=m.weight.dtype)
                        else:
                            current_awq_x_s = current_awq_x_s.to(device=m.weight.device)

                    use_awq_gptq_parity_path = False
                    if set(normalized_sources) == {"awq", "gptq"}:
                        gptq_record = external_records["gptq"]
                        gptq_scale = gptq_record.get("scale")
                        gptq_zero = gptq_record.get("zero")
                        group_size = q_config["q_group_size"]
                        num_groups = (
                            m.weight.shape[-1] // group_size
                            if group_size > 0
                            else 1
                        )
                        expected_qparam_shape = (m.weight.shape[0], num_groups)
                        use_awq_gptq_parity_path = (
                            torch.is_tensor(gptq_scale)
                            and torch.is_tensor(gptq_zero)
                            and tuple(gptq_scale.shape) == expected_qparam_shape
                            and tuple(gptq_zero.shape) == expected_qparam_shape
                        )
                        if (
                            not use_awq_gptq_parity_path
                            and not legacy_gptq_weight_only_warned
                        ):
                            scale_shape = (
                                tuple(gptq_scale.shape)
                                if torch.is_tensor(gptq_scale)
                                else None
                            )
                            zero_shape = (
                                tuple(gptq_zero.shape)
                                if torch.is_tensor(gptq_zero)
                                else None
                            )
                            print(
                                "[warn] Incomplete legacy GPTQ qparams detected "
                                f"(scale={scale_shape}, zero={zero_shape}, "
                                f"expected={expected_qparam_shape}). Fake quantization "
                                "will use the exported GPTQ weight directly; this file "
                                "is still not valid for real quantization."
                            )
                            legacy_gptq_weight_only_warned = True

                    if use_awq_gptq_parity_path:
                        candidates = _real_awq_gptq_candidates(
                            m.weight.data,
                            current_awq_x_s,
                            gptq_record["weight"],
                            gptq_record["scale"],
                            gptq_record["zero"],
                            n_bit=w_bit,
                            gptq_g_idx=gptq_record.get("g_idx"),
                            **q_config,
                        )
                        m.weight.data = candidates[8]
                    else:
                        m.weight.data = pseudo_quantize_tensor_mix(
                            m.weight.data,
                            normalized_sources,
                            external_candidates=external_candidates,
                            current_awq_x_s=current_awq_x_s,
                            n_bit=w_bit,
                            **q_config,
                        )
                    tmp_i += 1
                    if not keep_on_cpu:
                        m.cpu()
                        torch.cuda.empty_cache()
                    mix_pbar.update(1)
        print(f"tmp_i: {tmp_i}")
        return

    # 导入gptq模型权重
    # 如果 awq_results 中已经有 scaled_gptq_weights，直接使用（已经在 entry.py 中通过重新 apply_awq722 得到）
    if awq_results is not None and "scaled_gptq_weights" in awq_results:
        gptq_weights = awq_results["scaled_gptq_weights"]
        print(f"Using pre-scaled gptq_weights from awq_results")
    else:
        # 否则加载原始的 gptq_weights
        
        if 'Llama-7b' in model_path:
            gptq_weights = _load_quant_param_map("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_1.pt", "gptq")
        elif 'llama-2-7b' in model_path:
            gptq_weights = _load_quant_param_map("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_2.pt", "gptq")
        elif 'Meta-Llama-3-8B' in model_path:
            gptq_weights = _load_quant_param_map("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_3.pt", "gptq")
        else:
            gptq_weights = {}
        
        # 如果提供了 awq_results，对 gptq_weights 也应用 scales（旧方法，保留作为后备）
        # if awq_results is not None and "scale" in awq_results:
        #     scales_list = awq_results["scale"]
        #     apply_scale_to_weights_dict(gptq_weights, scales_list)
        #     print(f"Applied scales to gptq_weights (fallback method)")
    original_model_weights = awq_results["original_weights"]

    gptq_weights = awq_results['gptq_weights']
    scaled_gptq_weights = awq_results['scaled_gptq_weights']
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

            gptq_W_int = gptq_weights[full_name]["weight"].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )
            scaled_gptq_W_int = scaled_gptq_weights[full_name].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )
            gptq_W_scales = gptq_weights[full_name]["scale"].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )
            gptq_W_zeros = gptq_weights[full_name]["zero"].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )
            W_o = original_model_weights[full_name].to(
                device=m.weight.device,
                dtype=m.weight.dtype
            )

            # print('pseudo_quantize_model_weight722')
            # print(n)
            current_awq_x_s = awq_results["awq_x_s"][full_name].to(device=m.weight.device)
            m.weight.data = pseudo_quantize_tensor722(
                m.weight.data,current_awq_x_s.view(1, -1), gptq_W_int, gptq_W_scales, gptq_W_zeros, n_bit=w_bit, quant_type = quant_type, th_t=1, **q_config, search_flag=search_flag
            )
            # m.weight.data.div_(current_awq_x_s.view(1, -1))
            tmp_i += 1
            m.cpu()
    print(f"tmp_i: {tmp_i}")
    print_max_dot_idx_stats()



@torch.no_grad()
def real_quantize_model_weight_old(model, w_bit, q_config, init_only=False):
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
                module.weight.data, scales, zeros = pseudo_quantize_tensor722(
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



@torch.no_grad()
def prepare_wqlinear0124_cuda_runtime(
    model, desc="Preparing fused AWQ/GPTQ CUDA runtime"
):
    """Convert checkpoint layouts once and bind reusable input scaling."""
    from .qmodule import WQLinear0124, bind_wqlinear0124_shared_input_scale
    from .pre_quant import get_blocks

    layers = get_blocks(model)
    for layer in tqdm(layers, desc=desc):
        cuda_device = None
        for module in layer.modules():
            if isinstance(module, WQLinear0124):
                module.prepare_input_domain_mode()
                module.prepare_gptq_group_index()
                module.prepare_cuda_native_mask()
                module.prepare_cuda_paired_quant_params()
                if module.full_mask_packed.is_cuda:
                    cuda_device = module.full_mask_packed.device
                    # Bound temporary overlap to one linear while converting
                    # the backward-compatible checkpoint representation.
                    torch.cuda.synchronize(cuda_device)

    if os.environ.get("WQ0124_SHARE_INPUT_SCALE", "1") != "0":
        for layer in layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                bind_wqlinear0124_shared_input_scale(
                    [
                        getattr(attention, "q_proj", None),
                        getattr(attention, "k_proj", None),
                        getattr(attention, "v_proj", None),
                    ]
                )
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                bind_wqlinear0124_shared_input_scale(
                    [
                        getattr(mlp, "gate_proj", None),
                        getattr(mlp, "up_proj", None),
                    ]
                )


@torch.no_grad()
def real_quantize_model_weight_0124(
    model,
    w_bit,
    q_config,
    init_only=False,
    awq_results=None,
    model_path=None,
    gptq_params=None,
    mix_sources=None,
):
    from .qmodule import WQLinear0124
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."
    if w_bit != 4:
        raise NotImplementedError("The fused two-source real backend supports w_bit=4 only")

    real_sources = {
        item.strip().lower()
        for item in str(mix_sources or "awq,gptq").split(",")
        if item.strip()
    }
    if real_sources == {"awq", "gptq"}:
        first_source, second_source = "awq", "gptq"
    elif real_sources == {"awq", "rtn"}:
        first_source, second_source = "awq", "rtn"
    elif real_sources == {"gptq", "rtn"}:
        first_source, second_source = "gptq", "rtn"
    else:
        raise ValueError(
            "The fused real backend supports exactly awq,gptq, awq,rtn, "
            "or gptq,rtn; "
            f"got {sorted(real_sources)}"
        )

    layers = get_blocks(model)
    if not init_only:
        if "awq" in real_sources and awq_results is None:
            raise ValueError("Real AWQ mixed quantization requires --load_awq")
        if "awq" in real_sources and "awq_x_s" not in awq_results:
            raise ValueError("AWQ results do not contain per-layer activation scales")
        if "gptq" in real_sources:
            if awq_results is not None and "gptq_weights" in awq_results:
                gptq = awq_results["gptq_weights"]
            else:
                gptq_path = gptq_params or _default_gptq_param_path(model_path, w_bit)
                if gptq_path is None:
                    raise ValueError(
                        "Real quantization with GPTQ requires --gptq_params for this model"
                    )
                if not os.path.exists(gptq_path):
                    raise FileNotFoundError(
                        f"GPTQ parameter file not found: {gptq_path}; pass --gptq_params"
                    )
                gptq = _load_quant_param_map(gptq_path, "gptq")

        print(f"[real mix] sources: {first_source},{second_source}")
    
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            full_name = f"model.layers.{i}.{name}"
            
            if init_only:
                q_linear = WQLinear0124.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                selected_gptq_g_idx = None
                if first_source == "gptq":
                    gptq_record = _get_layer_record(gptq, full_name, "gptq")
                    if gptq_record.get("scale") is None or gptq_record.get("zero") is None:
                        raise KeyError(
                            f"Layer {full_name} in GPTQ params must provide weight/scale/zero"
                        )
                    # A resident all-ones scale marks that both candidates use
                    # the original activation domain. Runtime preparation uses
                    # this to select the no-alpha masked CUDA kernel once.
                    current_awq_x_s = torch.ones(
                        module.weight.shape[-1],
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    (
                        awq_w,
                        awq_scales,
                        awq_zeros,
                        second_w,
                        second_scales,
                        second_zeros,
                        full_w,
                        full_mask,
                    ) = _real_gptq_rtn_candidates(
                        module.weight.data,
                        gptq_record["weight"],
                        gptq_record["scale"],
                        gptq_record["zero"],
                        n_bit=w_bit,
                        **q_config,
                    )
                else:
                    current_awq_x_s = _get_record_by_alias(
                        awq_results["awq_x_s"], full_name
                    )
                    if current_awq_x_s is None:
                        current_awq_x_s = torch.ones(
                            module.weight.shape[-1],
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                    else:
                        current_awq_x_s = current_awq_x_s.to(
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                    if second_source == "rtn":
                        (
                            awq_w,
                            awq_scales,
                            awq_zeros,
                            second_w,
                            second_scales,
                            second_zeros,
                            full_w,
                            full_mask,
                        ) = _real_awq_rtn_candidates(
                            module.weight.data,
                            current_awq_x_s,
                            n_bit=w_bit,
                            **q_config,
                        )
                    else:
                        gptq_record = _get_layer_record(gptq, full_name, "gptq")
                        if gptq_record.get("scale") is None or gptq_record.get("zero") is None:
                            raise KeyError(
                                f"Layer {full_name} in GPTQ params must provide weight/scale/zero"
                            )
                        (
                            awq_w,
                            awq_scales,
                            awq_zeros,
                            second_w,
                            second_scales,
                            second_zeros,
                            full_w,
                            full_mask,
                            _effective_weight,
                            selected_gptq_g_idx,
                        ) = _real_awq_gptq_candidates(
                            module.weight.data,
                            current_awq_x_s,
                            gptq_record["weight"],
                            gptq_record["scale"],
                            gptq_record["zero"],
                            n_bit=w_bit,
                            gptq_g_idx=gptq_record.get("g_idx"),
                            **q_config,
                        )
                # awq_w, awq_scales, awq_zeros, gptq_w, gptq_scales, gptq_zeros = pseudo_quantize_tensor0124(
                #     module.weight.data,gptq_w, gptq_w, gptq_scale, gptq_zeros, n_bit=w_bit, get_scale_zp=True, **q_config
                # )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear0124.from_linear(
                    module,
                    w_bit,
                    q_config["q_group_size"],
                    False,
                    awq_w,
                    awq_scales,
                    awq_zeros,
                    second_w,
                    second_scales,
                    second_zeros,
                    current_awq_x_s.view(1, -1),
                    full_w,
                    full_mask,
                    selected_gptq_g_idx,
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
