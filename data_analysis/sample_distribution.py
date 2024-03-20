import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from components.sample_generation import min_max_scaling
from inner_functions.path import sorted_files, build_path
from inner_types.data import Dataset
from inner_types.path import Path


def get_sample(file_path: str, interval: int):
    """
    Gets samples of a user inside a window
    :param file_path: Path of the displacement matrix
    :param interval: The interval to pick up a sample from
    :return: A sample as 1D array
    """
    with open(file_path) as file:
        intervals = file.readlines()[1:]
        sample = intervals[interval]
        sample = np.array(sample.split(','), dtype="float64")
        if sample.max() > sample.min():
            sample = min_max_scaling(sample)
            return sample
        else:
            return None


def heatmap_matrix(matrix_pearson: np.array, matrix_kendaltau: np.array):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plotar heatmap para a matriz de correlação de Pearson
    im1 = axs[0].imshow(matrix_pearson, cmap='viridis', interpolation='nearest', vmin=matrix_pearson.min(), vmax=1)
    axs[0].set_title('Matriz de Correlação - Pearson')
    fig.colorbar(im1, ax=axs[0], label='Correlação')

    # Plotar heatmap para a matriz de correlação de Kendall-Tau
    im2 = axs[1].imshow(matrix_kendaltau, cmap='viridis', interpolation='nearest', vmin=matrix_kendaltau.min(), vmax=1)
    axs[1].set_title('Matriz de Correlação - Kendall-Tau')
    fig.colorbar(im2, ax=axs[1], label='Correlação')

    plt.tight_layout()
    plt.show()


def most_correlated(matrix: np.array):
    total = 0
    e0 = 0
    p25 = 0
    p50 = 0
    p75 = 0
    p1 = 0
    n25 = 0
    n50 = 0
    n75 = 0
    n1 = 0
    correlations = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            total += 1
            correlations.append(matrix[i, j])
            if matrix[i, j] == 0.0:
                e0 += 1
            elif matrix[i, j] > 0:
                if matrix[i, j] < 0.25:
                    p25 += 1
                elif matrix[i, j] < 0.5:
                    p50 += 1
                elif matrix[i, j] < 0.75:
                    p75 += 1
                else:
                    p1 += 1
            else:
                if matrix[i, j] > -0.25:
                    n25 += 1
                elif matrix[i, j] > -0.5:
                    n50 += 1
                elif matrix[i, j] > -0.75:
                    n75 += 1
                else:
                    n1 += 1

    def percent(c):
        return f'{c*100/total:.2f}%'
    print(f'Total de pares = {total}\nIguais a 0: {e0}\n--')
    print(f'Entre 0 e 0.25: {p25} ({percent(p25)})\nEntre 0.25 e 0.50: {p50} ({percent(p50)})\n'
          f'Entre 0.50 e 0.75: {p75} ({percent(p75)})\nEntre 0.75 e 1: {p1} ({percent(p1)})\n--')
    print(f'Entre 0 e -0.25: {n25} ({percent(n25)})\nEntre -0.25 e -0.50: {n50} ({percent(n50)})\n'
          f'Entre -0.50 e -0.75: {n75} ({percent(n75)})\nEntre -0.75 e -1: {n1} ({percent(n1)})\n--')

    mean_correlation = np.mean(correlations)
    print(f'A correlação média é de {mean_correlation}\n')


class SampleAnalysis:

    def __init__(self, dataset: Dataset, interval: int):
        self.dataset = dataset
        self.f3_dm = Path.f3_dm(dataset.name)
        self.data = self.load_interval(interval)

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
        return data

    def pearson_correlation_at_interval(self):
        pearson_correlation_matrix = np.corrcoef(self.data)
        print('Pearson Correlation')
        most_correlated(pearson_correlation_matrix)
        return pearson_correlation_matrix

    def kendaltau_correlation_at_interval(self):
        variables = len(self.data)
        kendaltau_correlation_matrix = np.zeros((variables, variables))

        for i in range(variables):
            for j in range(variables):
                coeficiente, _ = kendalltau(self.data[i], self.data[j])
                kendaltau_correlation_matrix[i, j] = coeficiente
        print('Kendaltau Correlation')
        most_correlated(kendaltau_correlation_matrix)
        return kendaltau_correlation_matrix
