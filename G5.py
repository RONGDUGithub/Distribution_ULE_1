from all_backup import *
import pandas as pd
import os
from functools import reduce
from multiple_server import multi_mechanism_switch
import time
import winsound


def run_experiment(file_name):
    try:
        # 尝试读取txt文件
        data_array = pd.read_csv(file_name, header=None, sep='\n')

        # 转换为numpy数组并进行数据处理
        data0 = data_array.to_numpy()
        user_number = len(data0)

        # 确保数据被正确展平
        if data0.ndim > 1:
            data = data0.flatten()
        else:
            data = data0

        # 确保数据类型是float
        data = data.astype(float)

        # 计算最大最小值
        min_data = np.min(data)
        max_data = np.max(data)

    except Exception as e:
        print(f"第一种读取方法失败: {e}")
        try:
            # 备选读取方法
            with open(file_name, 'r') as file:
                data0 = np.array([float(line.strip()) for line in file])
                user_number = len(data0)
                data = data0.flatten()
                min_data = np.min(data)
                max_data = np.max(data)

        except Exception as e:
            print(f"备选读取方法也失败: {e}")
            raise Exception("无法读取文件")

    mechanisms = ['duchi', 'piecewise', 'sw']
    epsilons_list = [
        [0.1, 0.1, 0.1],
    ]

    num_runs = 10  # 重复运行10次
    JS_matrix = []
    time_matrix = []

    for epsilons in epsilons_list:
        js_sum = 0
        time_sum = 0

        for run in range(num_runs):
            js_divergences, time_final = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
            js_sum += js_divergences
            time_sum += time_final

        # 计算平均值
        js_avg = js_sum / num_runs
        time_avg = time_sum / num_runs

        JS_matrix.append(js_avg)
        time_matrix.append(time_avg)

    JS_matrix = np.array(JS_matrix)
    base_name = os.path.splitext(file_name)[0]
    output_filename = './group5/Group5_{}_B.txt'.format(base_name)
    with open(output_filename, 'w') as f:
        f.write('Mechanisms:\n')
        f.write(str(mechanisms) + '\n')
        f.write('Epsilons:\n')
        f.write(str(epsilons_list) + '\n')
        f.write('time_matrix:\n')
        f.write(str(time_matrix) + '\n')
        f.write('JS:\n')
        f.write(str(JS_matrix) + '\n')

    return JS_matrix, time_matrix


# 使用示例：
# file_name = 'Betawave_5000000.txt'
# JS_matrix, time_matrix = run_experiment(file_name)

# 或者处理多个文件
file_names = ['Betawave_1000000.txt', 'Betawave_5000000.txt', 'Betawave_10000000.txt']
for file_name in file_names:
    JS_matrix, time_matrix = run_experiment(file_name)

# file_name = 'Betawave_5000000.txt'
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
#
# except Exception as e:
#     print(f"第一种读取方法失败: {e}")
#     try:
#         # 备选读取方法
#         with open(file_name, 'r') as file:
#             data0 = np.array([float(line.strip()) for line in file])
#             user_number = len(data0)
#             data = data0.flatten()
#             min_data = np.min(data)
#             max_data = np.max(data)
#
#     except Exception as e:
#         print(f"备选读取方法也失败: {e}")
#         raise Exception("无法读取文件")
#
# mechanisms = ['duchi', 'piecewise', 'sw']
# epsilons_list = [
#     [0.1, 0.1, 0.1],
# ]
# JS_matrix = []
# time_matrix = []
# for epsilons in epsilons_list:
#     js_divergences, time_final = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
#     # MSE_final, time_final = run_experiment(file_name, mechanisms, epsilons)
#     JS_matrix.append(js_divergences)
#     time_matrix.append(time_final)
#
# JS_matrix = np.array(JS_matrix)
# base_name = os.path.splitext(file_name)[0]
# output_filename = './group5/Group5_{}_B.txt'.format(base_name)
# with open(output_filename, 'w') as f:
#     f.write('Mechanisms:\n')
#     f.write(str(mechanisms) + '\n')
#     f.write('Epsilons:\n')
#     f.write(str(epsilons_list) + '\n')  # 直接转换list为字符串
#     f.write('time_matrix:\n')
#     f.write(str(time_matrix) + '\n')  # 直接转换list为字符串
#     f.write('JS:\n')
#     f.write(str(JS_matrix) + '\n')  # 假设MSE_final是numpy数组

winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒




