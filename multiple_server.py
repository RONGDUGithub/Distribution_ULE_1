import numpy as np
import matplotlib.pyplot as plt
from Duchinvp import duchi
from Hybridnvp import hybrid
from Laplacenvp import laplace1
from Piecewisenvp import piecewise
from SWunbiasednvp import sw_unbiased
from sw import sw
from discrete import *
from bayesain_updating import *
from calculate_matrix import *
from variance_final import *
from duchi_distribution import *
from pm_distribution import *
from fusion_all import EMS, fusion, EMS_large
from all_backup import *
import pandas as pd
import os
from functools import reduce
import time


def multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons, runs=10, bins=1024):
    """
    简化版本，只返回核心结果
    """
    results = {
        'distributions': [],  # 所有分布结果
        'noisy_data': [],  # 所有加噪数据
        'locations': [],  # 所有位置数据
        'transforms': [],  # 所有变换数据
        'epsilons': epsilons  # 使用的epsilon值
    }

    if isinstance(epsilons, list):
        if len(epsilons) != len(mechanisms):
            raise ValueError("Number of epsilons must match number of mechanisms")
        epsilon_dict = {i: eps for i, eps in enumerate(epsilons)}
    else:
        epsilon_dict = epsilons

    for mech_idx, mechanism in enumerate(mechanisms):
        current_epsilon = epsilon_dict[mech_idx]

        if mechanism.lower() == 'duchi':
            result_duchi, noise_duchi = duchi(data, min_data, max_data, current_epsilon)
            location, transform, distribution = duchi_distribution(result_duchi, current_epsilon, domain_bins=bins)
            distribution = distribution / len(data)

            results['distributions'].append(distribution)
            results['noisy_data'].append(noise_duchi)
            results['locations'].append(location)
            results['transforms'].append(transform)

        elif mechanism.lower() == 'piecewise':
            result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, current_epsilon)
            location, transform, ns_hist, distribution = pm_distribution(
                noise_piecewise,
                current_epsilon,
                randomized_bins=bins,
                domain_bins=bins
            )
            distribution_all = distribution / len(data)

            results['distributions'].append(distribution)
            results['noisy_data'].append(noise_piecewise)
            results['locations'].append(location)
            results['transforms'].append(transform)

        elif mechanism.lower() == 'sw':
            location, transform, distribution = sw(data, min_data, max_data, current_epsilon)
            distribution = distribution / len(data)

            results['distributions'].append(distribution)
            results['noisy_data'].append(None)  # sw没有noisy_data
            results['locations'].append(location)
            results['transforms'].append(transform)

        else:
            raise ValueError("Mechanism must be one of: 'duchi', 'piecewise', 'sw'")

    distribution = results['distributions']
    locations = results['locations']  # 所有location的列表
    transforms = results['transforms']  # 所有transform的列表
    min_vals = [np.min(np.abs(matrix[matrix != 0])) for matrix in transforms]
    overall_min = min(min_vals)
    scale = 10 ** len(str(overall_min).split('.')[1])
    print(f"放大倍数: {scale}")

    scaled_transforms = [matrix * scale for matrix in transforms]
    h = len(data)

    start_time = time.time()
    distribution_fusion3 = fusion_any(len(data), locations, scaled_transforms, h)
    end_time = time.time()
    execution_time = end_time - start_time
    time_sta = execution_time

    distribution_fusion4 = distribution_fusion3 / len(data)
    distribution_fusion5 = distribution_fusion4.cpu().numpy()

    # 将转换后的数组添加到 distribution 列表中
    distribution.append(distribution_fusion5)

    num_bins = 1024
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(data.min(), data.max()), density=True)
    hist = hist / 1024

    D = []

    # 循环处理distribution中的每个数组
    for dist in distribution:
        d_value = js_divergence(hist, dist)
        D.append(d_value)

    # 如果你希望D是numpy数组而不是列表，可以转换
    D = np.array(D)
    return D, time_sta


def fusion_any(data_len, locations, transforms, h):
    n = 1024
    max_iteration = 10000
    loglikelihood_threshold = 1e-3

    unique_combinations = []
    combination_counter = {}
    probability_matrix = []
    probability_matrix2 = []
    # 遍历数组
    for i in range(h):
        combination = tuple(trans[i] for trans in locations)

        if combination in combination_counter:
            # 组合已经存在，计数器+1
            combination_counter[combination] += 1
        else:
            # 组合是新的，添加到列表并初始化计数器
            unique_combinations.append(combination)
            combination_counter[combination] = 1
            # 更新概率矩阵

            # 修改后的代码，获取所有3个元素
            arrays = [transforms[j][locations[j][i], :] for j in range(len(transforms))]
            result = reduce(np.multiply, arrays)
            # result = elementwise_multiplication_six(arrays)
            probability_matrix.append(result)

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist = list(combination_counter.values())
    transform = np.array(probability_matrix)

    # return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * h

    if data_len > 1000000:
        return EMS_large(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * h
    else:
        return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * h


def sample_data(input_file, n_cols):
    # 读取数据
    df = pd.read_csv('output4.csv', header=None)
    total_rows = len(df)
    samples_per_col = total_rows // n_cols  # 每列需要采样的数量

    print(f"总行数: {total_rows}")
    print(f"每列采样数量: {samples_per_col}")

    # 只选择指定列数
    df = df.iloc[:, :n_cols]

    # 创建结果DataFrame
    result_df = pd.DataFrame().reindex_like(df)

    # 保持所有可用的行索引
    available_indices = set(range(total_rows))

    # 对每一列进行采样
    for col in df.columns:
        # 检查是否还有足够的可用行
        if len(available_indices) < samples_per_col:
            print(f"警告：列 {col} 没有足够的可用行进行采样")
            current_indices = np.random.choice(list(available_indices),
                                               size=len(available_indices),
                                               replace=False)
        else:
            current_indices = np.random.choice(list(available_indices),
                                               size=samples_per_col,
                                               replace=False)

        # 将采样到的数据放入结果DataFrame
        result_df.iloc[list(current_indices), col] = df.iloc[list(current_indices), col]

        # 从可用索引中移除已使用的索引
        available_indices -= set(current_indices)

        # 打印剩余可用行数
        print(f"列 {col} 采样后，剩余可用行数: {len(available_indices)}")

    return result_df

# data_array = pd.read_excel('beta25.xlsx')
# data0 = data_array.to_numpy()
# user_number = len(data0)
# data = data0.flatten()
# mechanisms = ['duchi', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw']
# epsilons = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# min_data = np.min(data)
# max_data = np.max(data)
#
# js_divergences = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
# print(js_divergences)
