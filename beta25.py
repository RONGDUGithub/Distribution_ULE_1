import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 创建x值的范围
x = np.linspace(0, 1, 1000)

# 计算Beta(2,5)分布的概率密度
y = beta.pdf(x, 2, 5)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', lw=2, label='Beta(2,5)')
plt.fill_between(x, y, alpha=0.2)
plt.grid(True)
plt.title('Beta(2,5) Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()