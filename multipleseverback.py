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
from fusion_all import EMS, fusion
from all_backup import *
import pandas as pd
import os
from functools import reduce


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

    # 展开列表作为参数
    args = []
    for loc, trans in zip(locations, transforms):
        args.extend([loc, trans])

    distribution_fusion = fusion(*args)
    # distribution_fusion1 = fusion(*args)
    h = len(data)

    distribution_fusion3 = fusion_any(locations, transforms, h)
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
    return D


def fusion_any(locations, transforms, h):
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
    # transform = np.array(probability_matrix)
    transform = np.array(probability_matrix)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * h


data_array = pd.read_excel('beta25.xlsx')
data0 = data_array.to_numpy()
user_number = len(data0)
data = data0.flatten()
mechanisms = ['duchi', 'piecewise', 'sw', 'sw', 'sw']
epsilons = [1, 1, 1, 1, 1]
min_data = np.min(data)
max_data = np.max(data)

js_divergences = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
print(js_divergences)
