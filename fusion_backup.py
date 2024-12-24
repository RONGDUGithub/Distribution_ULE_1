import numpy as np
from tqdm import tqdm

from all_backup import *
import scipy
from numpy import linalg as LA
import numpy as np
import torch
import scipy.special


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
    probability_matrix2 = []
    # 遍历数组
    for i in range(len(location_sw)):
        combination = (location_sw[i], location_pm[i], location_duchi[i])

        if combination in combination_counter:
            # 组合已经存在，计数器+1
            combination_counter[combination] += 1
        else:
            # 组合是新的，添加到列表并初始化计数器
            unique_combinations.append(combination)
            combination_counter[combination] = 1
            # 更新概率矩阵

            array_sw = transform_sw[location_sw[i], :]
            array_duchi = transform_duchi[location_duchi[i], :]
            array_pm = transform_pm[location_pm[i], :]
            array_sw1 = transform_sw1[location_sw[i], :]
            array_duchi1 = transform_duchi1[location_duchi[i], :]
            array_pm1 = transform_pm1[location_pm[i], :]
            result = elementwise_multiplication_three(array_sw, array_duchi, array_pm)
            result2 = elementwise_multiplication_three(array_sw1, array_duchi1, array_pm1)
            probability_matrix.append(result)
            probability_matrix2.append(result2)
    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist = list(combination_counter.values())
    # transform = np.array(probability_matrix)
    transform = np.array(probability_matrix2)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(location_sw)


def fusion_six(location_sw, location_sw2, transform_sw,
               location_pm, location_pm2, transform_pm,
               location_duchi, location_duchi2, transform_duchi):
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
    probability_matrix2 = []
    # 遍历数组
    for i in range(len(location_sw)):
        combination = (location_sw[i], location_pm[i], location_duchi[i],
                       location_sw2[i], location_pm2[i], location_duchi2[i])

        if combination in combination_counter:
            # 组合已经存在，计数器+1
            combination_counter[combination] += 1
        else:
            # 组合是新的，添加到列表并初始化计数器
            unique_combinations.append(combination)
            combination_counter[combination] = 1
            # 更新概率矩阵

            array_sw = transform_sw[location_sw[i], :]
            array_duchi = transform_duchi[location_duchi[i], :]
            array_pm = transform_pm[location_pm[i], :]
            array_sw2 = transform_sw[location_sw2[i], :]
            array_duchi2 = transform_duchi[location_duchi2[i], :]
            array_pm2 = transform_pm[location_pm2[i], :]
            result = elementwise_multiplication_six(array_sw, array_duchi, array_pm, array_sw2, array_duchi2, array_pm2)
            probability_matrix.append(result)

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist = list(combination_counter.values())
    # transform = np.array(probability_matrix)
    transform = np.array(probability_matrix)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(location_sw)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert inputs to torch tensors and move to GPU if available
    if isinstance(ns_hist, (np.ndarray, list)):
        ns_hist = torch.tensor(ns_hist, dtype=torch.float64, device=device)
    if isinstance(transform, (np.ndarray, list)):
        transform = torch.tensor(transform, dtype=torch.float64, device=device)

    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = torch.tensor([scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)],
                                dtype=torch.float64, device=device)
    smoothing_matrix = torch.zeros((n, n), dtype=torch.float64, device=device)
    central_idx = int(len(binomial_tmp) / 2)

    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]

    row_sum = torch.sum(smoothing_matrix, dim=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = torch.ones(n, dtype=torch.float64, device=device) / float(n)
    theta_old = torch.zeros(n, dtype=torch.float64, device=device)
    r = 0
    sample_size = torch.sum(ns_hist)
    old_loglikelihood = 0

    while torch.norm(theta_old - theta, p=1).item() > 1 / sample_size and r < max_iteration:
        theta_old = theta.clone()
        X_condition = torch.clamp(torch.matmul(transform, theta_old), min=1e-10)

        TMP = transform.T / X_condition

        P = torch.matmul(TMP, ns_hist)
        P = P * theta_old

        theta = P / torch.sum(P)

        # Smoothing step
        theta = torch.matmul(smoothing_matrix, theta)
        theta = theta / torch.sum(theta)

        loglikelihood = torch.dot(ns_hist, torch.log(torch.matmul(transform, theta)))
        improvement = loglikelihood - old_loglikelihood

        if r > 1 and torch.abs(improvement) < loglikelihood_threshold:
            break

        old_loglikelihood = loglikelihood
        r += 1

    return theta


def EMS_large(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert inputs to torch tensors and move to GPU if available
    if isinstance(ns_hist, (np.ndarray, list)):
        ns_hist = torch.tensor(ns_hist, dtype=torch.float64, device=device)
    if isinstance(transform, (np.ndarray, list)):
        transform = torch.tensor(transform, dtype=torch.float64)#, device=device)

    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = torch.tensor([scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)],
                                dtype=torch.float64, device=device)
    smoothing_matrix = torch.zeros((n, n), dtype=torch.float64, device=device)
    central_idx = int(len(binomial_tmp) / 2)

    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]

    row_sum = torch.sum(smoothing_matrix, dim=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = torch.ones(n, dtype=torch.float64, device=device) / float(n)
    theta_old = torch.zeros(n, dtype=torch.float64, device=device)
    r = 0
    sample_size = torch.sum(ns_hist)
    old_loglikelihood = 0

    while torch.norm(theta_old - theta, p=1).item() > 1 / sample_size and r < max_iteration:
        theta_old = theta.clone()
        X_condition = []
        batch_num = 2
        batch_size = int(len(transform)/batch_num)
        for i in tqdm(range(batch_num+1)):
            trans_batch = transform[i*batch_size: min(len(transform), (i+1)*batch_size)]
            trans_batch = trans_batch.to(device)
            X_condition.append(torch.clamp(torch.matmul(trans_batch, theta_old), min=1e-10).cpu())
        #X_condition = torch.clamp(torch.matmul(transform, theta_old), min=1e-10)

        X_condition = torch.cat(X_condition)
        TMP = transform.T / X_condition

        P = torch.matmul(TMP.to(device), ns_hist)
        P = P * theta_old

        theta = P / torch.sum(P)

        # Smoothing step
        theta = torch.matmul(smoothing_matrix, theta)
        theta = theta / torch.sum(theta)

        trans_mul_theta = []
        for i in range(batch_num+1):
            trans_batch = transform[i*batch_size: min(len(transform), (i+1)*batch_size)]
            trans_batch = trans_batch.to(device)
            trans_mul_theta.append(torch.matmul(trans_batch, theta).cpu())

        trans_mul_theta = torch.cat(trans_mul_theta).to(device)

        loglikelihood = torch.dot(ns_hist, torch.log(trans_mul_theta))#torch.matmul(transform, theta)))
        improvement = loglikelihood - old_loglikelihood

        if r > 1 and torch.abs(improvement) < loglikelihood_threshold:
            break

        old_loglikelihood = loglikelihood
        r += 1

    return theta


def fusion_four(location_sw, location_sw2, transform_sw,
               location_pm, location_pm2, transform_pm):
    n = 1024
    max_iteration = 10000
    loglikelihood_threshold = 1e-3

    transform_sw1 = transform_sw
    transform_pm1 = transform_pm


    h = len(location_sw)
    unique_combinations = []
    combination_counter = {}
    probability_matrix = []
    probability_matrix2 = []
    # 遍历数组
    for i in range(len(location_sw)):
        combination = (location_sw[i], location_pm[i],
                       location_sw2[i], location_pm2[i])

        if combination in combination_counter:
            # 组合已经存在，计数器+1
            combination_counter[combination] += 1
        else:
            # 组合是新的，添加到列表并初始化计数器
            unique_combinations.append(combination)
            combination_counter[combination] = 1
            # 更新概率矩阵

            array_sw = transform_sw[location_sw[i], :]

            array_pm = transform_pm[location_pm[i], :]
            array_sw2 = transform_sw[location_sw2[i], :]

            array_pm2 = transform_pm[location_pm2[i], :]
            result = elementwise_multiplication_four(array_sw, array_pm, array_sw2, array_pm2)
            probability_matrix.append(result)

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist = list(combination_counter.values())
    # transform = np.array(probability_matrix)
    transform = np.array(probability_matrix)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(location_sw)

