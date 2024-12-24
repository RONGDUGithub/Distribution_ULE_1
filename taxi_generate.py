import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_excel('taxi.xlsx')

# 设置想要保留的数据量
n = 1000  # 你可以修改这个数字

# 随机抽样
sampled_df = df.sample(n=n, random_state=42)  # random_state确保结果可重复

# 保存到新文件
sampled_df.to_excel(f'taxi_{n}.xlsx', index=False)