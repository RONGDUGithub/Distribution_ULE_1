from all_backup import *
import pandas as pd
import os
from functools import reduce
from multiple_server import multi_mechanism_switch
import time
import winsound

file_name = 'Betawave.xlsx'
data_array = pd.read_excel(file_name)
data0 = data_array.to_numpy()
user_number = len(data0)
data = data0.flatten()
min_data = np.min(data)
max_data = np.max(data)
mechanisms = ['duchi', 'duchi', 'duchi', 'duchi']
epsilons_list = [
    [0.1, 0.2, 0.3, 0.4],
]
JS_matrix = []
time_matrix = []
for epsilons in epsilons_list:
    js_divergences, time_final = multi_mechanism_switch(mechanisms, data, min_data, max_data, epsilons)
    # MSE_final, time_final = run_experiment(file_name, mechanisms, epsilons)
    JS_matrix.append(js_divergences)
    time_matrix.append(time_final)

JS_matrix = np.array(JS_matrix)
base_name = os.path.splitext(file_name)[0]
output_filename = './group2/Group2_{}_duchi.txt'.format(base_name)
with open(output_filename, 'w') as f:
    f.write('Mechanisms:\n')
    f.write(str(mechanisms) + '\n')
    f.write('Epsilons:\n')
    f.write(str(epsilons_list) + '\n')  # 直接转换list为字符串
    f.write('time_matrix:\n')
    f.write(str(time_matrix) + '\n')  # 直接转换list为字符串
    f.write('JS:\n')
    f.write(str(JS_matrix) + '\n')  # 假设MSE_final是numpy数组

winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(2000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒

winsound.Beep(4000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(3000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒
winsound.Beep(1000, 500)  # 频率为 1000 Hz，持续时间为 500 毫秒




