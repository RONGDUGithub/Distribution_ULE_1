import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 生成 data1, 服从 Beta(1, 6) 分布
user_number = 10000
data1 = np.random.beta(1, 6, size=user_number)

# 对 data1 进行归一化
data1_normalized = 2 * (data1 - data1.min()) / (data1.max() - data1.min()) - 1

# 生成 data2, 个数为 0.8 * user_number
data2_size = int(0.8 * user_number)
data2 = np.random.normal(loc=0.5, scale=0.2, size=data2_size)

# 对 data2 进行归一化
data2_normalized = 2 * (data2 - data2.min()) / (data2.max() - data2.min()) - 1

# 生成 data4, 它是 data1_normalized 和 data2_normalized 合并后的数据
data4 = np.concatenate((data1_normalized, data2_normalized))


# 从 data1_normalized 中选择 20% 的数据加入到 data2_normalized 中
same_indices = np.random.choice(user_number, int(user_number * 0.2), replace=False)
data2_indices = np.random.choice(data2_size, len(same_indices), replace=False)
data2_normalized[data2_indices] = data1_normalized[same_indices]

# 提取加到 data2_normalized 中的数据
data2_added = data1_normalized[same_indices]

# 绘制 data1_normalized, data2_normalized 和 data2_added 的分布
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data1_normalized, bins=50, alpha=0.5, label='data1_normalized')
ax.hist(data2_normalized, bins=50, alpha=0.5, label='data2_normalized')
ax.hist(data2_added, bins=50, alpha=0.5, label='data2_added')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of data1_normalized, data2_normalized, and data2_added')
ax.legend()

# 计算两个分布的重叠面积
p = np.histogram(data1_normalized, bins=50)[0] / len(data1_normalized)
q = np.histogram(data2_normalized, bins=50)[0] / len(data2_normalized)
overlap = 1 - 0.5 * entropy(p, q)

print(f"The overlap area between data1_normalized and data2_normalized is: {overlap:.4f}")

plt.show()