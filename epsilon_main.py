import numpy as np
import matplotlib.pyplot as plt
from Duchinvp import duchi
from Piecewisenvp import piecewise
from sw import sw
from duchi_distribution import duchi_distribution
from pm_distribution import pm_distribution
from fusion_all import fusion
from all_backup import js_divergence, js_divergence_gpu2


def calculate_js_divergences(data, epsilon):
    epsilon_default = epsilon/3
    min_data = 0
    max_data = 1
    # NVP:输出的为扰动后的数组
    # duchi:
    result_duchi, noise_duchi = duchi(data, min_data, max_data, epsilon_default)
    result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, epsilon_default)

    location_sw, transform_sw, distribution_sw = sw(data, min_data, max_data, epsilon_default)
    distribution_sw = distribution_sw / len(data)

    location_pm, transform_pm, ns_hist, distribution_pm = pm_distribution(noise_piecewise, epsilon_default,
                                                                          randomized_bins=1024, domain_bins=1024)
    distribution_pm = distribution_pm / len(data)

    location_duchi, transform_duchi, distribution_duchi = duchi_distribution(result_duchi, epsilon_default,
                                                                             domain_bins=1024)
    distribution_duchi = distribution_duchi / len(data)

    distribution_fusion = fusion(location_sw, transform_sw, location_pm, transform_pm, location_duchi, transform_duchi)
    distribution_fusion = distribution_fusion / len(data)

    # 画图
    x = np.linspace(min_data, max_data, 1024)

    # plt.figure()

    num_bins = 1024

    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(data.min(), data.max()), density=True)
    hist = hist / 1024
    # bin_edges 是一个长度为 num_bins + 1 的数组，表示每个 bin 的边界
    # 我们计算每个 bin 的中心点以绘制概率密度函数
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    D_sw = js_divergence(hist, distribution_sw)
    D_pm = js_divergence(hist, distribution_pm)
    D_duchi = js_divergence(hist, distribution_duchi)
    D_fusion = js_divergence_gpu2(hist, distribution_fusion)

    # Store results in a list
    js_divergences = [D_sw, D_pm, D_duchi, D_fusion]

    # Output the list
    # print(f"JS Divergences: {js_divergences}")

    return js_divergences


# 调用函数
# calculate_js_divergences(data, epsilon)
