import numpy as np
from epsilon_main import calculate_js_divergences
import pandas as pd


data_array = pd.read_excel('Beta25.xlsx')
user_number = len(data_array)
data0 = data_array.to_numpy()
data = data0.flatten()

a = np.min(data)
b = np.max(data)
data = (data - a) / (b - a)

# 定义 epsilon 列表
epsilon_values = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
# 重复次数
num_repeats = 50


# 打开一个文件用于写结果
with open("Group1_Beta25_distribution.txt", "w") as file:

    for epsilon in epsilon_values:
        js_divergences_list = []

        for _ in range(num_repeats):
            js_divergences = calculate_js_divergences(data, epsilon=epsilon)
            js_divergences_list.append(js_divergences)

        # 计算每个 epsilon 下的平均 JS 散度
        js_divergences_avg = np.mean(js_divergences_list, axis=0)

        # 将结果写入文件
        # file.write(f"Epsilon: {epsilon}\n")
        file.write(f"{js_divergences_avg.tolist()}\n\n")

print("Results have been written to results.txt")