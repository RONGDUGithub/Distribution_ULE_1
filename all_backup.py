import numpy as np
from scipy.special import rel_entr
import torch
import scipy.special

def get_positions(a, b, A, n_segments=1024):
    """
    将区间 [a, b] 拆分成 n_segments 份，根据数组 A 中的值，返回位置参数（0 到 n_segments-1）。

    参数:
    a (float): 区间的起始点。
    b (float): 区间的结束点。
    A (array-like): 包含属于 [a, b] 区间的数值的数组。
    n_segments (int): 拆分的份数，默认值为 1024。

    返回:
    array: 包含位置参数的数组。
    """
    # 生成 n_segments + 1 个等间隔点
    edges = np.linspace(a, b, n_segments + 1)

    # 使用 np.digitize 确定 A 中每个值的位置参数
    positions = np.digitize(A, edges) - 1

    # 确保位置参数在合法范围内
    positions = np.clip(positions, 0, n_segments - 1)

    return positions


def elementwise_multiplication_three(vector1, vector2, vector3):
    # 检查向量长度是否相等
    if len(vector1) != len(vector2) or len(vector1) != len(vector3):
        raise ValueError("三个向量的长度必须相等")

    # 对应元素相乘
    result = np.multiply(np.multiply(vector1, vector2), vector3)

    return result

def elementwise_multiplication_two(vector1, vector2):
    # 检查向量长度是否相等
    # if len(vector1) != len(vector2) or len(vector1) != len(vector3):
    #     raise ValueError("三个向量的长度必须相等")
    # 对应元素相乘
    result = np.multiply(vector1, vector2)

    return result


def elementwise_multiplication_six(vector1, vector2, vector3, vector4, vector5, vector6):
    # 检查向量长度是否相等
    # if len(vector1) != len(vector2) or len(vector1) != len(vector3):
        # raise ValueError("三个向量的长度必须相等")

    # 对应元素相乘
    # result = np.multiply(np.multiply(vector1, vector2), vector3)
    result = np.multiply(
        np.multiply(np.multiply(np.multiply(np.multiply(vector1, vector2), vector3), vector4), vector5), vector6)

    return result

def elementwise_multiplication_four(vector1, vector2, vector3, vector4):
    # 检查向量长度是否相等
    # if len(vector1) != len(vector2) or len(vector1) != len(vector3):
        # raise ValueError("三个向量的长度必须相等")

    # 对应元素相乘
    # result = np.multiply(np.multiply(vector1, vector2), vector3)
    result = np.multiply(np.multiply(np.multiply(vector1, vector2), vector3), vector4)

    return result

def kl_divergence(p, q):
    """
    Calculate the KL divergence between two distributions.

    Parameters:
    p (array-like): The first distribution (true distribution).
    q (array-like): The second distribution (approximate distribution).

    Returns:
    float: The KL divergence between distribution p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure the distributions sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # Compute the KL divergence
    kl_div = np.sum(rel_entr(p, q))

    return kl_div


def js_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two distributions.

    Parameters:
    p (array-like): The first distribution.
    q (array-like): The second distribution.

    Returns:
    float: The JS divergence between distribution p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure the distributions sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # Calculate the middle distribution
    m = 0.5 * (p + q)

    # Compute the JS divergence
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return js_div


def js_divergence_gpu(p, q):
    # Assuming p and q are torch tensors
    if p.is_cuda:
        p = p.cpu()
    if q.is_cuda:
        q = q.cpu()

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    m = 0.5 * (p + q)
    return 0.5 * (scipy.special.rel_entr(p, m).sum() + scipy.special.rel_entr(q, m).sum())

def normalize_data(data):
    """
    将数据归一化到 [0, 1] 区间。

    参数:
    data (list 或 numpy array): 要归一化的数据

    返回:
    numpy array: 归一化后的数据
    """
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)

    # 检查是否所有值都相同
    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def js_divergence_gpu2(p, q):
    # Ensure p and q are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)

    # Move tensors to CPU if they are on GPU
    if p.is_cuda:
        p = p.cpu()
    if q.is_cuda:
        q = q.cpu()

    # Convert tensors to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Calculate the Jensen-Shannon divergence
    m = 0.5 * (p + q)
    return 0.5 * (scipy.special.rel_entr(p, m).sum() + scipy.special.rel_entr(q, m).sum())
