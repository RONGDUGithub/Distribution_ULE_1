import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy import integrate
import math
from discrete import *
from calculate_matrix import *


# def duchi_bayes(user_value_noise_sr, mid_values__minus1_to_1_sr, epsilon, final):
#     d_sr = len(mid_values__minus1_to_1_sr)
#     term2 = np.zeros(d_sr)
#
#     for j in range(d_sr):
#         value = mid_values__minus1_to_1_sr[j]
#         if user_value_noise_sr > 0:
#             term1 = (((np.exp(epsilon) - 1) * value) / (2 * np.exp(epsilon) + 2)) + 1 / 2
#         else:
#             term1 = 1 - (((np.exp(epsilon) - 1) * value) / (2 * np.exp(epsilon) + 2)) + 1 / 2
#         term2[j] = term1 * final[j]
#
#     term3 = np.sum(term2)
#     final = term2 / term3
#     # print("term3_duchi:", term3)
#     # x = np.linspace(0, 1, d_sr)  # 生成从 0 到 1 的等间隔数值序列
#     # plt.plot(x, final)
#     # plt.show()
#     return final

def duchi_bayes(user_value_noise_sr, mid_values__minus1_to_1_sr, epsilon, final):
    d_sr = len(mid_values__minus1_to_1_sr)

    # Vectorize the calculation
    value = mid_values__minus1_to_1_sr
    term1 = np.where(user_value_noise_sr > 0,
                     (((np.exp(epsilon) - 1) * value) / (2 * np.exp(epsilon) + 2)) + 1 / 2,
                     1 - (((np.exp(epsilon) - 1) * value) / (2 * np.exp(epsilon) + 2)) + 1 / 2)
    term2 = term1 * final
    term3 = np.sum(term2)
    final = term2 / term3

    return final


# def pm_bayes(d_pm, matrix_pm, nearest_index, final):
#     term2 = np.zeros(d_pm)
#
#     for j in range(d_pm):
#         term1 = matrix_pm[nearest_index, j]
#         term2[j] = term1 * final[j]
#
#     term3 = np.sum(term2)
#     final = term2 / term3
#     # print("term3_pm:", term3)
#     return final

def pm_bayes(d_pm, matrix_pm, nearest_index, final):
    term1 = matrix_pm[nearest_index, :]
    term2 = term1 * final
    term3 = np.sum(term2)
    final = term2 / term3
    return final


# def sw_bayes(d_sw, matrix_sw, nearest_index, final):
#     term2 = np.zeros(d_sw)
#
#     for j in range(d_sw):
#         term1 = matrix_sw[nearest_index, j]
#         term2[j] = term1 * final[j]
#
#     term3 = np.sum(term2)
#     final = term2 / term3
#     # print("term3_sw:", term3)
#     return final

def sw_bayes(d_sw, matrix_sw, nearest_index, final):
    term1 = matrix_sw[nearest_index, :]
    term2 = term1 * final
    term3 = np.sum(term2)
    final = term2 / term3
    return final


def laplace_bayes(d_lap, epsilon_lap, v_perturbed, final):
    term2 = np.zeros(d_lap)
    # Generate midpoints
    midpoints = np.linspace(-1, 1 - 2 / d_lap, d_lap) + 1 / d_lap

    # Parameters for the Laplace distribution
    c = 2 / epsilon_lap  # scale parameter

    # Integrate Laplace PDF around perturbed value
    results = []
    for i in range(1, d_lap + 1):
        mu = midpoints[i - 1]
        laplace_pdf = lambda x: laplace.pdf(x, loc=mu, scale=c)
        # Calculate the absolute value of v_perturbed
        if v_perturbed > 200:
            v_perturbed = 200
        if v_perturbed < -200:
            v_perturbed = -200
        abs_v_perturbed = abs(v_perturbed)
        # Calculate the value of ol
        ol = math.pow(abs_v_perturbed, 0.5)
        a = v_perturbed - ol  # lower bound of integration
        b = v_perturbed + ol  # upper bound of integration
        result, _ = integrate.quad(laplace_pdf, a, b)
        term2[i - 1] = result * final[i - 1]
        # print("v_perturbed:", v_perturbed)
        # print("result:", result)
    term3 = np.sum(term2)
    final = term2 / term3
    # print("term3_laplace:", term3)
    # # Normalize results
    # total = sum(results)
    # normalized_results = [r / total for r in results]
    # x = np.linspace(-1, 1, d_lap)  # 生成从 0 到 1 的等间隔数值序列
    # plt.plot(x, final)
    # plt.show()
    return final


def compute_and_apply_pm(epsilon_pm, d_default, result_piecewise):
    """
    Compute and apply the PM (Piecewise Monotonic) method to the given inputs.

    Args:
        epsilon_pm (float): The value of epsilon_pm.
        d_default (float): The default value of d.
        result_piecewise (float): The result of the piecewise method.

    Returns:
        float: The final result after applying the PM method.
    """
    # Calculate C
    C = (np.exp(epsilon_pm / 2) + 1) / (np.exp(epsilon_pm / 2) - 1)

    # Discretize the interval
    midpoints_minus1_to_1_pm = np.linspace(-1, 1, 101)
    db_pm = (2 / 100) * np.ones(100)
    BM_pm = np.exp(-np.abs(midpoints_minus1_to_1_pm))

    # Compute the probability matrix
    matrix_pm = np.exp(
        -np.abs(np.expand_dims(midpoints_minus1_to_1_pm, axis=1) - np.expand_dims(midpoints_minus1_to_1_pm, axis=0)))

    # Find the nearest index
    nearest_index_pm = np.argmin(np.abs(midpoints_minus1_to_1_pm - result_piecewise))

    # Apply the PM Bayes method
    final = np.dot(matrix_pm[:, nearest_index_pm], db_pm) / np.sum(db_pm)

    return final


def pm_pre_bayes(t, epsilon_pm, d_default, final):
    C = (np.exp(epsilon_pm / 2) + 1) / (np.exp(epsilon_pm / 2) - 1)
    midpoints_pm, BM_pm, midpoints_minus1_to_1_pm, db_pm = discretize_interval_pm(C, d_default)
    matrix_pm = compute_matrix_pm(epsilon_pm, midpoints_minus1_to_1_pm, db_pm, BM_pm)
    nearest_index_pm = np.argmin(np.abs(midpoints_minus1_to_1_pm - t))
    final = pm_bayes(d_default, matrix_pm, nearest_index_pm, final)
    return final


def sw_pre_bayes(t, epsilon_sw, d_default, final):
    b_sw = ((epsilon_sw * np.exp(epsilon_sw) - np.exp(epsilon_sw) + 1) / (
            2 * np.exp(epsilon_sw) * (np.exp(epsilon_sw) - 1 - epsilon_sw)))
    midpoints_sw, BM_sw, midpoints_minus0_to_1_sw, db_sw = discretize_interval_sw(b_sw, d_default)
    nearest_index_sw = np.argmin(np.abs(midpoints_minus0_to_1_sw - t))
    matrix_sw = compute_matrix_sw(epsilon_sw, midpoints_minus0_to_1_sw, db_sw, BM_sw)
    final = sw_bayes(d_default, matrix_sw, nearest_index_sw, final)
    return final
