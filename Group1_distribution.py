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
import time
from datetime import datetime



# 文件名
filename = 'Retirement.xlsx'

# 读取数据
data_array = pd.read_excel(filename)
data0 = data_array.to_numpy()
user_number = len(data0)
data = data0.flatten()

# 定义 epsilon 数组
epsilons = [0.2, 0.4, 0.6, 0.8, 1, 1.2]  # example array, you can modify as needed

# 一定要归一化成 [0,1]
min_data = np.min(data)
max_data = np.max(data)
normalized_data = (data - min_data) / (max_data - min_data)

# 创建存储结果的文件夹
output_dir = "group1"
os.makedirs(output_dir, exist_ok=True)

# 打开一个文件以写入所有结果
output_file = os.path.join(output_dir, f"group1_{os.path.splitext(os.path.basename(filename))[0]}_distribution.txt")

with open(output_file, "w") as f:
    for epsilon in epsilons:
        print('epsilon:', epsilon)
        epsilon_default = epsilon

        js_divergences_list = []
        time_lsit = []
        for i in tqdm(range(2), desc="Processing"):
        # for _ in range(10):  # 运行10次
            # NVP:输出的为扰动后的数组
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
            start_time = time.time()
            start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            D_fusion = js_divergence_gpu2(hist, distribution_fusion)
            end_time = time.time()
            end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            execution_time = end_time - start_time
            js_divergences = [D_sw, D_pm, D_duchi, D_fusion, execution_time]

            js_divergences_list.append(js_divergences)

        # 计算每个epsilon的平均JS散度
        js_divergences_avg = np.mean(js_divergences_list, axis=0)

        # f.write(f"Epsilon: {epsilon}, Average JS Divergences: {js_divergences_avg.tolist()}\n")
        f.write(f"{js_divergences_avg.tolist()}\n")

winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
