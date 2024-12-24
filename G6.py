from all_backup import *
import pandas as pd
import os
from functools import reduce
from multiple_server import *
import time
import winsound

# file_name = 'Betawave_500000.txt'
#
# try:
#     # 尝试读取txt文件
#     data_array = pd.read_csv(file_name, header=None, sep='\n')
#
#     # 转换为numpy数组并进行数据处理
#     data0 = data_array.to_numpy()
#     user_number = len(data0)
#
#     # 确保数据被正确展平
#     if data0.ndim > 1:
#         data = data0.flatten()
#     else:
#         data = data0
#
#     # 确保数据类型是float
#     data = data.astype(float)
#
#     # 计算最大最小值
#     min_data = np.min(data)
#     max_data = np.max(data)
def run_experiment(file_name, n_cols):
    # 读取数据
    result = sample_data('output4.csv', n_cols)

    mechanisms = ['duchi', 'piecewise', 'sw']
    epsilons = [0.1, 0.1, 0.1]

    num_runs = 10  # 重复运行10次
    all_JS = []
    all_times = []

    # 对每一列进行处理
    for col in result.columns:
        JS_matrix = []
        time_matrix = []

        # 获取当前列的数据
        data = result[col].dropna().values
        min_data = np.min(data)
        max_data = np.max(data)

        # 对每列重复运行10次
        for _ in range(num_runs):
            js_divergences, time_final = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
            JS_matrix.append(js_divergences)
            time_matrix.append(time_final)

        # 计算这一列10次运行的平均值
        js_avg_col = np.mean(JS_matrix, axis=0)
        time_avg_col = np.mean(time_matrix, axis=0)

        all_JS.append(js_avg_col)
        all_times.append(time_avg_col)

    # 转换为numpy数组并计算所有列的平均值
    all_JS = np.array(all_JS)
    all_times = np.array(all_times)

    # 计算所有列的总平均值
    JS_final_avg = np.mean(all_JS, axis=0)
    time_final_avg = np.mean(all_times, axis=0)

    # 保存结果
    base_name = os.path.splitext(file_name)[0]
    output_filename = './multiple_data/Group6_{}_{}_B.txt'.format(base_name, n_cols)

    with open(output_filename, 'w') as f:
        f.write('Mechanisms:\n')
        f.write(str(mechanisms) + '\n')
        f.write('Epsilons:\n')
        f.write(str(epsilons) + '\n')
        f.write('time_matrix:\n')
        f.write(str(time_final_avg) + '\n')
        f.write('JS:\n')
        f.write(str(JS_final_avg) + '\n')

    return JS_final_avg, time_final_avg


# 使用示例：
file_name = 'output4.xlsx'
# n_cols = 60
# JS_avg, time_avg = run_experiment(file_name, n_cols)

# 或者处理不同的列数
col_numbers = [10, 20, 30, 40, 50, 60]
for n_cols in col_numbers:
    JS_avg, time_avg = run_experiment(file_name, n_cols)
# file_name = 'output4.xlsx'
# n_cols = 60  # 现在可以设置更大的列数
# result = sample_data('output4.csv', n_cols)
#
# mechanisms = ['duchi', 'piecewise', 'sw']
# # n_cols = 10  # 现在可以设置更大的列数
# epsilons = [
#     0.1, 0.1, 0.1,
# ]
# JS_matrix = []
# time_matrix = []
#
# for col in result.columns:
#     # 获取当前列的数据
#     data = result[col].dropna().values  # dropna()去掉空值
#     min_data = np.min(data)
#     max_data = np.max(data)
#     js_divergences, time_final = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
#
#     JS_matrix.append(js_divergences)
#     time_matrix.append(time_final)
#
# JS_matrix = np.array(JS_matrix)
# base_name = os.path.splitext(file_name)[0]
# time_matrix = np.array(time_matrix)
#
# JS_avg = np.mean(JS_matrix, axis=0)  # shape: (num_mechanisms, num_epsilons)
# time_avg = np.mean(time_matrix, axis=0)
#
#
# output_filename = './multiple_data/Group6_{}_{}_B.txt'.format(base_name, n_cols)
# with open(output_filename, 'w') as f:
#     f.write('Mechanisms:\n')
#     f.write(str(mechanisms) + '\n')
#     f.write('Epsilons:\n')
#     f.write(str(epsilons) + '\n')  # 直接转换list为字符串
#     f.write('time_matrix:\n')
#     f.write(str(time_matrix) + '\n')  # 直接转换list为字符串
#     f.write('JS:\n')
#     f.write(str(JS_avg) + '\n')  # 假设MSE_final是numpy数组

winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒




