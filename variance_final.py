import numpy as np


def Variance_sr(t, epsilon):
    eexp = np.exp(epsilon)
    Var_sr = np.power((eexp + 1) / (eexp - 1), 2) - np.power(t, 2)
    return Var_sr


def Variance_pm(t, epsilon):
    eexp2 = np.exp(epsilon / 2)
    Var_pm = np.power(t, 2) / (eexp2 - 1) + (eexp2 + 3) / (3 * np.power((eexp2 - 1), 2))
    return Var_pm


def Variance_laplace(t, epislon):
    Var_laplace = 8 / np.power(epislon, 2)
    return Var_laplace


def Variance_squarewave(t, epsilon):
    t2 = (t + 1) / 2
    eexp = np.exp(epsilon)
    budget = eexp
    b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))
    # high_area  S_h
    p = budget / (2 * b * budget + 1)
    # low_area  S_l
    q = 1 / (2 * b * budget + 1)
    Var_sw = 4 * (q * ((1 + 3 * b + 3 * np.power(b, 2) - 6 * b * np.power(t2, 2)) / 3) +
                  p * ((6 * b * np.power(t2, 2) + 2 * np.power(b, 3)) / 3) -
                  np.power(t2 * 2 * b * (p - q) + q * (b + 1 / 2), 2))
    Var_sw_unbiase = Var_sw / (np.power(2 * b * (p - q), 2))
    return Var_sw_unbiase


def compute_pm_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    var_final = 0
    for i in range(d_default):
        t = midpoints_minus1_to_1[i]
        var_final += final[i]* Variance_pm(t, epsilon)
    return var_final


def compute_sw_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    var_final = 0
    for i in range(d_default):
        t = midpoints_minus1_to_1[i]
        var_final += final[i]*Variance_squarewave(t, epsilon)
    return var_final


def compute_laplace_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    var_final = 0
    for i in range(d_default):
        t = midpoints_minus1_to_1[i]
        var_final += final[i]*Variance_laplace(t, epsilon)
    return var_final


def compute_duchi_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    var_final = 0
    for i in range(d_default):
        t = midpoints_minus1_to_1[i]
        var_final += final[i]*Variance_sr(t, epsilon)
    return var_final

# def compute_pm_variance(final, epsilon, d_default, midpoints_minus1_to_1):
#     t = midpoints_minus1_to_1[:d_default]
#     variances = Variance_pm(t, epsilon)
#     var_final = np.dot(final[:d_default], variances)
#     return var_final
#
#
# def compute_sw_variance(final, epsilon, d_default, midpoints_minus1_to_1):
#     t = midpoints_minus1_to_1[:d_default]
#     variances = Variance_squarewave(t, epsilon)
#     var_final = np.dot(final[:d_default], variances)
#     return var_final
#
#
# def compute_laplace_variance(final, epsilon, d_default, midpoints_minus1_to_1):
#     t = midpoints_minus1_to_1[:d_default]
#     variances = Variance_laplace(t, epsilon)
#     var_final = np.dot(final[:d_default], variances)
#     return var_final
#
#
# def compute_duchi_variance(final, epsilon, d_default, midpoints_minus1_to_1):
#     t = midpoints_minus1_to_1[:d_default]
#     variances = Variance_sr(t, epsilon)
#     var_final = np.dot(final[:d_default], variances)
#     return var_final
