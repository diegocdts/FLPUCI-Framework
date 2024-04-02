import numpy as np
from scipy.stats import kendalltau, spearmanr, t

from components.sample_generation import min_max_scaling
from inner_functions.path import sorted_files, build_path, interval_dir
from inner_types.data import Dataset
from inner_types.names import ExportedFiles
from inner_types.path import Path
from utils.plots import plot_avg_correlations, plot_correlations, heatmap_matrix_correlation


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

    confidence = 0.95
    std_error = np.std(above_diagonal_matrix, ddof=1) / np.sqrt(total)
    confidence_interval = t.interval(confidence, df=total-1, loc=mean_correlations, scale=std_error)

    return above_diagonal_matrix, mean_correlations, confidence_interval


class SampleCorrelation:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.f3_dm = Path.f3_dm(dataset.name)
        self.f9_analysis = Path.f9_results_analysis(dataset.name)
        self.data = None

    def load_interval(self, interval: int):
        data = np.array([])
        for index, file_name in enumerate(sorted_files(self.f3_dm)):
            file_path = build_path(self.f3_dm, file_name)
            user_sample = get_sample(file_path, interval)
            if user_sample is not None:
                if len(data) == 0:
                    data = user_sample
                else:
                    data = np.vstack((data, user_sample))
        self.data = data

    def pearson_correlation_at_interval(self):
        pearson_correlation_matrix = np.corrcoef(self.data)
        print('Pearson Correlation')
        return pearson_correlation_matrix

    def spearman_correlation_at_interval(self):
        variables = len(self.data)
        spearman_correlation_matrix = np.zeros((variables, variables))

        for i in range(variables):
            for j in range(variables):
                spearman_correlation_matrix[i, j], _ = spearmanr(self.data[i], self.data[j])

        print('Spearman Correlation')
        return spearman_correlation_matrix

    def kendaltau_correlation_at_interval(self):
        variables = len(self.data)
        kendaltau_correlation_matrix = np.zeros((variables, variables))

        for i in range(variables):
            for j in range(variables):
                coeficiente, _ = kendalltau(self.data[i], self.data[j])
                kendaltau_correlation_matrix[i, j] = coeficiente
        print('Kendaltau Correlation')
        return kendaltau_correlation_matrix


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

        pearson_matrix = analysis.pearson_correlation_at_interval()
        p_matrix, p_mean, p_ci = info_correlation(pearson_matrix)
        pearson_means.append(p_mean)
        pearson_ci.append(p_ci)

        spearman_matrix = analysis.spearman_correlation_at_interval()
        s_matrix, s_mean, s_ci = info_correlation(spearman_matrix)
        spearman_means.append(s_mean)
        spearman_ci.append(s_ci)

        kendal_matrix = analysis.kendaltau_correlation_at_interval()
        k_matrix, k_mean, k_ci = info_correlation(kendal_matrix)
        kendal_means.append(k_mean)
        kendal_ci.append(k_ci)

        path = build_path(analysis.f9_analysis, f'{interval_dir(interval + 1)} - {ExportedFiles.CORRELATION.value}')
        plot_correlations(p_matrix, p_mean, s_matrix, s_mean, k_matrix, k_mean, path)

        path = build_path(analysis.f9_analysis, f'{interval_dir(interval + 1)} - {ExportedFiles.MCORRELATION.value}')
        heatmap_matrix_correlation(pearson_matrix, spearman_matrix, kendal_matrix, path)

    path = build_path(analysis.f9_analysis, ExportedFiles.CORRELATION.value)

    plot_avg_correlations(pearson_means, pearson_ci, spearman_means, spearman_ci, kendal_means, kendal_ci, path)
