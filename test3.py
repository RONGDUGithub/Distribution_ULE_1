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
from fusion_all import fusion
from all_backup import *
import pandas as pd
import os
import winsound
from tqdm import tqdm  # 导入 tqdm 库

# 读取数据
data_array = pd.read_excel('Retirement.xlsx')
data0 = data_array.to_numpy()
user_number = len(data0)
data = data0.flatten()

# 定义 epsilon 数组
# epsilons = [0.5, 1, 1.5, 2, 2.5, 3]  # example array, you can modify as needed
epsilon = 1

# 一定要归一化成[0,1]
min_data = np.min(data)
max_data = np.max(data)

# 创建存储结果的文件夹
output_dir = "group2"
os.makedirs(output_dir, exist_ok=True)

# A(2,3,6) B(3,2,6) C(6,2,3) D(3,6,2)
# 打开一个文件以写入所有结果
output_file = os.path.join(output_dir, "group2_Retirement_comb362.txt")
with open(output_file, "w") as f:
    print('epsilon:', epsilon)
    # 定义不同算法的epsilon
    epsilon_duchi = epsilon / 3
    epsilon_pm = epsilon / 6
    epsilon_sw = epsilon / 2

    js_divergences_list = []

    for i in tqdm(range(2), desc="Processing"):
        # NVP:输出的为扰动后的数组
        result_duchi, noise_duchi = duchi(data, min_data, max_data, epsilon_duchi)
        result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, epsilon_pm)

        location_sw, transform_sw, distribution_sw = sw(data, min_data, max_data, epsilon_sw)
        distribution_sw = distribution_sw / len(data)

        location_pm, transform_pm, ns_hist, distribution_pm = pm_distribution(noise_piecewise, epsilon_pm,
                                                                              randomized_bins=1024, domain_bins=1024)
        distribution_pm = distribution_pm / len(data)

        location_duchi, transform_duchi, distribution_duchi = duchi_distribution(result_duchi, epsilon_duchi,
                                                                                 domain_bins=1024)
        distribution_duchi = distribution_duchi / len(data)

        distribution_fusion = fusion(location_sw, transform_sw, location_pm, transform_pm, location_duchi,
                                     transform_duchi)
        distribution_fusion = distribution_fusion / len(data)

        num_bins = 1024

        # 计算直方图
        hist, bin_edges = np.histogram(data, bins=num_bins, range=(data.min(), data.max()), density=True)
        hist = hist / 1024
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        D_sw = js_divergence(hist, distribution_sw)
        D_pm = js_divergence(hist, distribution_pm)
        D_duchi = js_divergence(hist, distribution_duchi)
        D_fusion = js_divergence_gpu2(hist, distribution_fusion)

        # Store results in a list
        js_divergences = [D_sw, D_pm, D_duchi, D_fusion]
        js_divergences_list.append(js_divergences)

    # 计算每个epsilon的平均JS散度
    js_divergences_avg = np.mean(js_divergences_list, axis=0)

    # Output the average list to the file
    f.write(f"Epsilon: {epsilon}, Average JS Divergences: {js_divergences_avg.tolist()}\n")


winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
