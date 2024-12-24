import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from Duchinvp import duchi
from Piecewisenvp import piecewise
from sw import sw
from pm_distribution import pm_distribution
from duchi_distribution import duchi_distribution
from fusion_all import fusion
from all_backup import *


def process_data(data, min_data, max_data, epsilon_default):
    # Duchi
    result_duchi, noise_duchi = duchi(data, min_data, max_data, epsilon_default)
    location_duchi, transform_duchi, distribution_duchi = duchi_distribution(result_duchi, epsilon_default,
                                                                             domain_bins=1024)

    # Piecewise
    result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, epsilon_default)
    location_pm, transform_pm, ns_hist, distribution_pm = pm_distribution(noise_piecewise, epsilon_default,
                                                                          randomized_bins=1024, domain_bins=1024)

    # Sw
    location_sw, transform_sw, distribution_sw = sw(data, min_data, max_data, epsilon_default)

    return distribution_duchi, distribution_pm, distribution_sw, location_sw, transform_sw, location_pm, transform_pm, location_duchi, transform_duchi


def main():
    # data_array = pd.read_excel('taxi.xlsx')
    # # user_number = len(data_array)
    # data0 = data_array.to_numpy()
    #
    # user_number = len(data0)
    # data = data0.flatten()

    user_number = 100000
    data = np.random.beta(2, 5, user_number)
    epsilon = 0.5
    epsilon_default = epsilon / 3

    min_data = 0
    max_data = 1

    # 创建一个进程池并并行处理数据
    with Pool(processes=3) as pool:
        results = pool.starmap(process_data, [(data, min_data, max_data, epsilon_default)])

    # Unpack results
    distribution_duchi, distribution_pm, distribution_sw, location_sw, transform_sw, location_pm, transform_pm, location_duchi, transform_duchi = \
    results[0]

    # Fusion and normalization
    distribution_fusion = fusion(location_sw, transform_sw, location_pm, transform_pm, location_duchi, transform_duchi)
    distribution_fusion = distribution_fusion / len(data)
    distribution_duchi = distribution_duchi / len(data)
    distribution_pm = distribution_pm / len(data)
    distribution_sw = distribution_sw / len(data)

    # Plotting and calculating JS divergences
    num_bins = 1024
    hist, bin_edges = np.histogram(data, bins=num_bins, range=(data.min(), data.max()), density=True)
    hist = hist / 1024
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate JS Divergences
    D_sw = js_divergence(hist, distribution_sw)
    D_pm = js_divergence(hist, distribution_pm)
    D_duchi = js_divergence(hist, distribution_duchi)
    D_fusion = js_divergence(hist, distribution_fusion)

    # Store and print results
    js_divergences = [D_sw, D_pm, D_duchi, D_fusion]
    print(f"JS Divergences: {js_divergences}")


if __name__ == "__main__":
    main()