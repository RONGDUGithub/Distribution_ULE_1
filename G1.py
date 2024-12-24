from all_backup import *
import pandas as pd
import os
from functools import reduce
from multiple_server import multi_mechanism_switch
import time
import winsound


def run_experiment(file_name):
    data_array = pd.read_excel(file_name)

    data0 = data_array.to_numpy()
    user_number = len(data0)
    data = data0.flatten()
    min_data = np.min(data)
    max_data = np.max(data)
    mechanisms = ['duchi', 'piecewise',  'sw']
    epsilons_list = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
        [0.5, 0.5, 0.5],
        [0.6, 0.6, 0.6]
    ]

    num_runs = 20  # 重复运行10次
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
    output_filename = './group1/Group1_{}_B.txt'.format(base_name)
    with open(output_filename, 'w') as f:
        f.write('Mechanisms:\n')
        f.write(str(mechanisms) + '\n')
        f.write('Epsilons:\n')
        f.write(str(epsilons_list) + '\n')
        f.write('time_matrix:\n')
        f.write(str(time_matrix) + '\n')
        f.write('JS:\n')
        f.write(str(JS_matrix) + '\n')



# 使用示例：
run_experiment('taxi.xlsx')

run_experiment('beta25.xlsx')

run_experiment('Retirement.xlsx')

run_experiment('Betawave.xlsx')

winsound.Beep(3000, 500)
winsound.Beep(2000, 500)
winsound.Beep(4000, 500)
winsound.Beep(3000, 500)
winsound.Beep(1000, 500)