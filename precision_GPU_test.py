import numpy as np

# 检查平台是否支持 float128（或者 np.longdouble）
if hasattr(np, 'float128'):
    dtype = np.float128
else:
    dtype = np.longdouble

# 创建一个示例矩阵 A
A = np.array([[1.12345678901234567890, 2.12345678901234567890],
              [3.12345678901234567890, 4.12345678901234567890]], dtype=np.float64)

# 将矩阵 A 的数据类型转换为 float128
A_float128 = A.astype(dtype)

# 打印转换后的矩阵
print("Original matrix A:")
print(A)

print("\nMatrix A with dtype float128:")
print(A_float128)

# 打印转换后矩阵的每个元素的小数点后位数
for row in A_float128:
    for value in row:
        value_str = f"{value:.40e}"  # 格式化为科学计数法，保留 40 位小数
        decimal_part = value_str.split('e')[0].split('.')[1]
        decimal_places = len(decimal_part.rstrip('0'))  # 去掉末尾的零，计算小数位数
        print(f"Value: {value_str}, Decimal Places: {decimal_places}")