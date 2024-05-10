import sys
from itertools import combinations

from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr, t

import numpy as np
import pandas as pd

from components.sample_generation import min_max_scaling
from inner_functions.path import sorted_files, build_path, interval_dir
from inner_types.data import Dataset
from inner_types.names import ExportedFiles
from inner_types.path import Path
from utils.plots import plot_correlations, heatmap_matrix_correlation, plot_avg_correlations


def get_sample(file_path: str, interval: int):
    """
    Gets samples of a user inside a window
    :param file_path: Path of the displacement matrix
    :param interval: The interval to pick up a sample from
    :return: A sample as 1D array
    """
    with open(file_path) as file:
        intervals = file.readlines()[1:]
        if interval <= len(intervals) - 1:
            sample = intervals[interval]
            sample = np.array(sample.split(','), dtype="float64")
            if sample.max() > sample.min():
                sample = min_max_scaling(sample)
                return sample
            else:
                return None
        else:
            return None


class SampleCorrelation:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.f9_analysis = Path.f9_results_analysis(dataset.name)
        self.data = None
        self.set_height_width()

    def set_height_width(self):
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = 0, 0
        for file_name in sorted_files(self.f2_data):
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path)
            if df.x.min() < min_x:
                min_x = df.x.min()
            if df.y.min() < min_y:
                min_y = df.y.min()
            if df.x.max() > max_x:
                max_x = df.x.max()
            if df.y.max() > max_y:
                max_y = df.y.max()
            del df
        float_height = (max_y - min_y) / self.dataset.resolution[0]
        float_width = (max_x - min_x) / self.dataset.resolution[1]
        self.dataset.set_height_width(float_height, float_width)

    def load_interval(self, interval: int):
        data = []
        for index, file_name in enumerate(sorted_files(self.f3_dm)):
            file_path = build_path(self.f3_dm, file_name)
            user_sample = get_sample(file_path, interval)
            if user_sample is not None:
                data.append(user_sample)
        self.data = np.array(data)

    def correlation_at_interval(self, correlation):
        dim_matrix = len(self.data)
        matrix = np.ones((dim_matrix, dim_matrix))
        for (index_i, image_i), (index_j, image_j) in combinations(enumerate(self.data), 2):
            corr = correlation(image_i, image_j)
            if type(corr) is tuple:
                corr = corr[0]
            #corr_abs = np.absolute(corr)
            corr_mean = np.mean(corr)
            matrix[index_i, index_j] = corr_mean
            matrix[index_j, index_i] = corr_mean
        return matrix


def info_correlation(matrix: np.array):
    np.fill_diagonal(matrix, np.nan)
    above_diagonal_matrix = matrix[np.triu_indices(matrix.shape[0], k=1)]

    mean_correlations = np.mean(above_diagonal_matrix)
    intervals = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    frequency, _ = np.histogram(above_diagonal_matrix, bins=intervals)

    total = len(above_diagonal_matrix)

    num_zeros = np.count_nonzero(above_diagonal_matrix == 0)
    num_positive_one = np.count_nonzero(above_diagonal_matrix == 1)
    num_negative_one = np.count_nonzero(above_diagonal_matrix == -1)

    percent = lambda freq: f'({freq*100/total:.2f}%)'
    """
    print(f'Num of pairs equal to -1 is {num_negative_one} {percent(num_negative_one)}')
    print(f'Num of pairs between -1 and -0.75 is {frequency[0]} {percent(frequency[0])}')
    print(f'Num of pairs between -0.75 and -0.50 is {frequency[1]} {percent(frequency[1])}')
    print(f'Num of pairs between -0.50 and -0.25 is {frequency[2]} {percent(frequency[2])}')
    print(f'Num of pairs between -0.25 and 0 is {frequency[3]} {percent(frequency[3])}')
    print(f'Num of pairs equal to 0 is {num_zeros} {percent(num_zeros)}')
    print(f'Num of pairs between 0 and 0.25 is {frequency[4]} {percent(frequency[4])}')
    print(f'Num of pairs between 0.25 and 0.50 is {frequency[5]} {percent(frequency[5])}')
    print(f'Num of pairs between 0.50 and 0.75 is {frequency[6]} {percent(frequency[6])}')
    print(f'Num of pairs between 0.75 and 1 is {frequency[7]} {percent(frequency[7])}')
    print(f'Num of pairs equal to 1 is {num_positive_one} {percent(num_positive_one)}')

    print(f'The mean correlation is {mean_correlations}\n------------')
    """
    confidence = 0.95
    std_error = np.std(above_diagonal_matrix, ddof=1) / np.sqrt(total)
    confidence_interval = t.interval(confidence, df=total-1, loc=mean_correlations, scale=std_error)

    return above_diagonal_matrix, mean_correlations, confidence_interval


def get_sample_correlations(dataset, last_interval):
    analysis = SampleCorrelation(dataset)

    pearson_means = []
    pearson_ci = []
    spearman_means = []
    spearman_ci = []
    kendal_means = []
    kendal_ci = []

    for interval in range(last_interval):
        analysis.load_interval(interval)

        if len(analysis.data) == 0:
            break

        pearson_matrix = analysis.correlation_at_interval(np.corrcoef)
        p_matrix, p_mean, p_ci = info_correlation(pearson_matrix)
        pearson_means.append(p_mean)
        pearson_ci.append(p_ci)

        spearman_matrix = analysis.correlation_at_interval(spearmanr)
        s_matrix, s_mean, s_ci = info_correlation(spearman_matrix)
        spearman_means.append(s_mean)
        spearman_ci.append(s_ci)

        kendal_matrix = analysis.correlation_at_interval(kendalltau)
        k_matrix, k_mean, k_ci = info_correlation(kendal_matrix)
        kendal_means.append(k_mean)
        kendal_ci.append(k_ci)

        path = build_path(analysis.f9_analysis, f'{interval_dir(interval)} - {ExportedFiles.CORRELATION.value}')
        plot_correlations(p_matrix, p_mean, s_matrix, s_mean, k_matrix, k_mean, path)

        path = build_path(analysis.f9_analysis, f'{interval_dir(interval)} - {ExportedFiles.MCORRELATION.value}')
        heatmap_matrix_correlation(pearson_matrix, spearman_matrix, kendal_matrix, path)

    path = build_path(analysis.f9_analysis, ExportedFiles.CORRELATION.value)

    plot_avg_correlations(pearson_means, pearson_ci, spearman_means, spearman_ci, kendal_means, kendal_ci, path)
