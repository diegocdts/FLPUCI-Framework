import numpy as np
import pandas as pd
from itertools import product
from sklearn.mixture import GaussianMixture

from inner_functions.files import read_json
from inner_functions.names import sources, column_k
from inner_functions.path import get_file_path, interval_dir, interval_json, metric_interval_json
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, WindowStrategyType
from inner_types.path import Path
from inner_types.validation import HeatmapMetric


def dataframe_basis():
    return pd.DataFrame(columns=['source'], data=sources())


class Clustering:

    def __init__(self, input_data: np.array):
        self.input_data = input_data

    def gmm(self, k: int):
        gmm = GaussianMixture(n_components=k, warm_start=True, max_iter=1000, random_state=1, n_init=10, verbose=0)
        gmm.fit(self.input_data)
        aic = gmm.aic(self.input_data)
        bic = gmm.bic(self.input_data)
        labels = gmm.predict(self.input_data)
        clusters = np.unique(labels)
        return clusters, labels, aic, bic


def intra_cluster_computation(dictionary: dict, clusters: np.array, labels: np.array,
                              user_indexes: np.array):
    all_intra_values = np.array([])
    for cluster in clusters:
        in_cluster_indices = np.where(labels == cluster)[0]
        in_cluster_users = user_indexes[in_cluster_indices]
        intra_cluster_values = []

        for node_i, node_j, in product(in_cluster_users, repeat=2):
            value = dictionary.get(f'{node_i}_{node_j}')
            if value is not None:
                intra_cluster_values.append(value)
        all_intra_values = np.append(all_intra_values, intra_cluster_values)
    return all_intra_values

def inter_cluster_computation(dictionary: dict, clusters: np.array, labels: np.array,
                              user_indexes: np.array):
    all_inter_values = np.array([])
    for cluster in clusters:
        in_cluster_indices = np.where(labels == cluster)[0]
        in_cluster_users = user_indexes[in_cluster_indices]
        out_cluster_indices = np.where(labels != cluster)[0]
        out_cluster_users = user_indexes[out_cluster_indices]
        inter_cluster_values = []

        for node_i, node_j in product(in_cluster_users, out_cluster_users):
            value = dictionary.get(f'{node_i}_{node_j}')
            if value is not None:
                inter_cluster_values.append(value)
        all_inter_values = np.append(all_inter_values, inter_cluster_values)
    return all_inter_values


def best_candidate(contact_time_dataframe: pd.DataFrame, k: int, best_contact_time_avg: float, best_k: int):
    intra_contact_time_avg = contact_time_dataframe[contact_time_dataframe.columns[-1]][1]
    if intra_contact_time_avg > best_contact_time_avg:
        return intra_contact_time_avg, k
    else:
        return best_contact_time_avg, best_k


class Validation:

    def __init__(self, dataset: Dataset, type_learning: LearningApproach, strategy_type: WindowStrategyType):
        self.type_learning = type_learning
        self.strategy_type = strategy_type

        self.f6_contact_time = Path.f6_contact_time(dataset.name)
        self.f7_metrics = Path.f7_metrics(dataset.name)
        self.f9_results = Path.f9_results(dataset.name, type_learning, strategy_type)

    def generate_communities(self, interval: int, input_data: np.array, user_indexes: np.array):
        path = get_file_path(self.f9_results, interval_dir(interval))
        contact_time_dataframe = dataframe_basis()
        mse_dataframe = dataframe_basis()
        ssim_dataframe = dataframe_basis()
        ari_dataframe = dataframe_basis()

        clustering = Clustering(input_data)
        helper = ValidationHelper(interval, path)

        best_contact_time_avg, best_k = 0, None

        for k in range(2, 20):
            print(f'Validation in interval {interval} with k = {k}')
            clusters, labels, aic, bic = clustering.gmm(k)
            helper.append_score(k, aic, bic)

            contact_time_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes)

            mse_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                HeatmapMetric.MSE)
            ssim_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                 HeatmapMetric.SSIM)
            ari_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                HeatmapMetric.ARI)

            best_contact_time_avg, best_k = best_candidate(contact_time_dataframe, k, best_contact_time_avg, best_k)

        helper.sort_scores(best_k)
        #continuar a partir daqui

    def metric_validation(self, interval: int, clusters: np.array, labels: np.array, user_indexes: np.array,
                          metric: HeatmapMetric = None):
        if metric is None:
            path = get_file_path(self.f6_contact_time, interval_json(interval))
        else:
            path = get_file_path(self.f7_metrics, metric_interval_json(metric, interval))
        dictionary = read_json(path)

        all_pairs = np.array(list(dictionary.values()))
        intra_community = intra_cluster_computation(dictionary, clusters, labels, user_indexes)
        inter_community = inter_cluster_computation(dictionary, clusters, labels, user_indexes)
        return [all_pairs, intra_community, inter_community]


class ValidationHelper:

    def __init__(self, interval: int, path: str):
        self.interval = interval
        self.aic_list = []
        self.bic_list = []
        self.columns_4_plot = None

    def append_score(self, k: int, aic: float, bic: float):
        self.aic_list.append([aic, k])
        self.bic_list.append([bic, k])

    def sort_scores(self, best_k: int):
        self.aic_list.sort()
        self.bic_list.sort()

        chosen_aic = column_k(self.aic_list[0][1])
        chosen_bic = column_k(self.bic_list[0][1])
        chosen_best = column_k(best_k)

        self.columns_4_plot = ['source', chosen_aic, chosen_bic, chosen_best]

