import numpy as np
from all_backup import *
import scipy
from numpy import linalg as LA
import numpy as np
import torch
import scipy.special
from sw import EM, EMS
# from fusion_all import EM, EMS


def fusion(location_sw, transform_sw, location_pm, transform_pm, location_duchi, transform_duchi):
    n = 1024
    max_iteration = 10000
    loglikelihood_threshold = 1e-3

    transform_sw1 = transform_sw
    transform_pm1 = transform_pm
    transform_duchi1 = transform_duchi

    h = len(location_sw)
    unique_combinations = []
    combination_counter = {}
    probability_matrix = []
    # 遍历数组
    for i in range(len(location_sw)):
        combination = (location_sw[i], location_duchi[i])

        if combination in combination_counter:
            # 组合已经存在，计数器+1
            combination_counter[combination] += 1
        else:
            # 组合是新的，添加到列表并初始化计数器
            unique_combinations.append(combination)
            combination_counter[combination] = 1
            # 更新概率矩阵

            array_sw = transform_sw[location_sw[i], :]
            array_duchi = 1
            array_pm = transform_pm[location_pm[i], :]

            result = elementwise_multiplication_two(array_sw, array_duchi)
            probability_matrix.append(result)

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist = list(combination_counter.values())
    transform = np.array(probability_matrix)
    # transform = np.array(probability_matrix2)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(location_sw)
