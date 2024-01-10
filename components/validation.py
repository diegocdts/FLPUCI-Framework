import numpy as np
import pandas as pd
from itertools import product

from scipy import stats
from sklearn.mixture import GaussianMixture

from inner_functions.files import read_json
from inner_functions.names import sources, column_k, curves
from inner_functions.path import get_file_path, interval_dir, interval_json, metric_interval_json
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, WindowStrategyType
from inner_types.names import ExportedFiles
from inner_types.path import Path
from inner_types.plots import AxisLabel
from inner_types.validation import HeatmapMetric
from utils.plots import plot_metric


def dataframe_basis():
    return pd.DataFrame(columns=['source'], data=sources())


def curve_dataframe_basis():
    return pd.DataFrame(columns=['curve'], data=curves())


def confidence_interval(data):
    mean, sigma = np.mean(data), np.std(data)
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(data)))
    return conf_int


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


def export_avg_ci(dataframe: pd.DataFrame,
                  path: str,
                  csv_name: str,
                  png_name: str,
                  k_candidates: np.arange,
                  ks_chosen: list,
                  axis_label: AxisLabel):
    curve_dataframe = curve_dataframe_basis()
    for index, row in dataframe.iterrows():
        for column in dataframe.columns[1:]:  # columns[0] is source name (all_pairs, intra and inter_community)
            values = row[column]
            curves_values = [confidence_interval(values)[0], values.mean(), confidence_interval(values)[1]]
            if index == 0:  # index == 0 (all_pairs)
                curve_dataframe[sources()[0]] = curves_values
            else:
                curve_dataframe[f'{column}|{sources()[index]}'] = curves_values
    csv_path = get_file_path(path, csv_name)
    curve_dataframe.to_csv(csv_path, sep=',', index=False)
    png_path = get_file_path(path, png_name)
    plot_metric(curve_dataframe, k_candidates, ks_chosen, axis_label, png_path)


class Validation:

    def __init__(self, dataset: Dataset, approach: LearningApproach, strategy_type: WindowStrategyType):
        self.dataset = dataset
        self.type_learning = approach
        self.strategy_type = strategy_type

        self.f6_contact_time = Path.f6_contact_time(dataset.name)
        self.f7_metrics = Path.f7_metrics(dataset.name)
        self.f9_results = Path.f9_results(dataset.name, approach, strategy_type)

    def generate_communities(self, interval: int, input_data: np.array, user_indexes: np.array):
        path = get_file_path(self.f9_results, interval_dir(interval))
        contact_time_dataframe = dataframe_basis()
        mse_dataframe = dataframe_basis()
        ssim_dataframe = dataframe_basis()
        ari_dataframe = dataframe_basis()

        clustering = Clustering(input_data)
        helper = ValidationHelper(interval)

        best_contact_time_avg, best_k = 0, None

        for k in self.dataset.k_candidates:
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

        export_avg_ci(contact_time_dataframe, path,
                      ExportedFiles.CONTACT_TIME_CSV.value,
                      ExportedFiles.CONTACT_TIME_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.CONTACT_TIME.value)
        export_avg_ci(mse_dataframe, path,
                      ExportedFiles.MSE_CSV.value, ExportedFiles.MSE_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.MSE.value)
        export_avg_ci(ssim_dataframe, path,
                      ExportedFiles.SSIM_CSV.value, ExportedFiles.SSIM_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.SSIM.value)
        export_avg_ci(ari_dataframe, path,
                      ExportedFiles.ARI_CSV.value, ExportedFiles.ARI_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.ARI.value)

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

    def __init__(self, interval: int):
        self.interval = interval
        self.aic_list = []
        self.bic_list = []
        self.ks_chosen = None

    def append_score(self, k: int, aic: float, bic: float):
        self.aic_list.append([aic, k])
        self.bic_list.append([bic, k])

    def sort_scores(self, best_k: int):
        self.aic_list.sort()
        self.bic_list.sort()

        chosen_aic = self.aic_list[0][1]
        chosen_bic = self.bic_list[0][1]
        chosen_best = best_k

        self.ks_chosen = [chosen_aic, chosen_bic, chosen_best]
