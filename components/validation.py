import numpy as np
import pandas as pd
from itertools import product

from scipy import stats
from sklearn.mixture import GaussianMixture

from inner_functions.files import read_json
from inner_functions.names import sources, column_k, curves
from inner_functions.path import build_path, interval_dir, interval_json, metric_interval_json, mkdir, get_subdir_list
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, WindowStrategyType
from inner_types.names import ExportedFiles
from inner_types.path import Path
from inner_types.plots import AxisLabel
from inner_types.validation import HeatmapMetric
from utils.plots import plot_metric, plot_strategy_comparison, plot_time_evolution


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


def best_candidate(best_metric_dataframe: pd.DataFrame, k: int, best_metric_avg: float, best_k: int):
    intra_metric_avg = best_metric_dataframe[best_metric_dataframe.columns[-1]][1].mean()
    if intra_metric_avg > best_metric_avg:
        return intra_metric_avg, k
    else:
        return best_metric_avg, best_k


def export_avg_ci(dataframe: pd.DataFrame,
                  path: str,
                  csv_name: str,
                  png_name: str,
                  k_candidates: np.arange,
                  ks_chosen: np.array,
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
    csv_path = build_path(path, csv_name)
    curve_dataframe.to_csv(csv_path, sep=',', index=False)
    png_path = build_path(path, png_name)
    plot_metric(curve_dataframe, k_candidates, ks_chosen, axis_label, png_path)


class Validation:

    def __init__(self, dataset: Dataset, approach: LearningApproach, strategy_type: WindowStrategyType,
                 best_metric: str):
        self.dataset = dataset
        self.approach = approach
        self.strategy_type = strategy_type
        self.best_metric = best_metric

        self.f6_contact_time = Path.f6_contact_time(dataset.name)
        self.f7_metrics = Path.f7_metrics(dataset.name)
        self.f9_results = Path.f9_results(dataset.name, approach, strategy_type)
        self.f9_results_compare_strategies = Path.f9_results_compare_strategies(dataset.name, approach)
        self.f9_results_time_evolution = Path.f9_results_time_evolution(dataset.name)

    def generate_communities(self, interval: int, input_data: np.array, user_indexes: np.array):
        path = mkdir(build_path(self.f9_results, interval_dir(interval)))
        contact_time_dataframe = dataframe_basis()
        mse_dataframe = dataframe_basis()
        ssim_dataframe = dataframe_basis()
        ari_dataframe = dataframe_basis()

        clustering = Clustering(input_data)
        helper = ValidationHelper(interval, path)

        best_metric_avg, best_k = 0, None  # the best metric must be contact_time or ssim or ari

        for k in self.dataset.k_candidates:
            print(f'Validation at interval {interval} with k = {k}')
            clusters, labels, aic, bic = clustering.gmm(k)
            helper.append_score(k, aic, bic)

            contact_time_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes)

            mse_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                HeatmapMetric.MSE)
            ssim_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                 HeatmapMetric.SSIM)
            ari_dataframe[column_k(k)] = self.metric_validation(interval, clusters, labels, user_indexes,
                                                                HeatmapMetric.ARI)

            if self.best_metric == 'ssim':
                best_metric_avg, best_k = best_candidate(ssim_dataframe, k, best_metric_avg, best_k)
            else:
                best_metric_avg, best_k = best_candidate(contact_time_dataframe, k, best_metric_avg, best_k)

        helper.sort_scores(best_k)

        export_avg_ci(contact_time_dataframe, path,
                      ExportedFiles.CONTACT_TIME_CSV.value,
                      ExportedFiles.CONTACT_TIME_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.CONTACT_TIME)
        export_avg_ci(mse_dataframe, path,
                      ExportedFiles.MSE_CSV.value, ExportedFiles.MSE_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.MSE)
        export_avg_ci(ssim_dataframe, path,
                      ExportedFiles.SSIM_CSV.value, ExportedFiles.SSIM_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.SSIM)
        export_avg_ci(ari_dataframe, path,
                      ExportedFiles.ARI_CSV.value, ExportedFiles.ARI_PNG.value,
                      self.dataset.k_candidates, helper.ks_chosen,
                      AxisLabel.ARI)

    def metric_validation(self, interval: int, clusters: np.array, labels: np.array, user_indexes: np.array,
                          metric: HeatmapMetric = None):
        if metric is None:
            path = build_path(self.f6_contact_time, interval_json(interval))
        else:
            path = build_path(self.f7_metrics, metric_interval_json(metric, interval))
        dictionary = read_json(path)

        all_pairs = np.array(list(dictionary.values()))
        intra_community = intra_cluster_computation(dictionary, clusters, labels, user_indexes)
        inter_community = inter_cluster_computation(dictionary, clusters, labels, user_indexes)
        return [all_pairs, intra_community, inter_community]

    def compare_strategies(self):
        acc_results_path = Path.f9_results(self.dataset.name, self.approach, WindowStrategyType.ACC)
        sli_results_path = Path.f9_results(self.dataset.name, self.approach, WindowStrategyType.SLI)

        acc_subdir_list = get_subdir_list(acc_results_path)
        sli_subdir_list = get_subdir_list(sli_results_path)

        if not acc_subdir_list or not sli_subdir_list:
            print('\n[INFO] To execute the compare_strategies function, it is mandatory to first perform the training '
                  'and validation of the FL-based/ACC and FL-based/SLI or Centralized/ACC and Centralized/SLI '
                  'combinations.\n')
            return

        common_intervals = acc_subdir_list.intersection(sli_subdir_list)

        csvs = [ExportedFiles.CONTACT_TIME_CSV, ExportedFiles.MSE_CSV, ExportedFiles.SSIM_CSV, ExportedFiles.ARI_CSV]
        axis = [AxisLabel.CONTACT_TIME, AxisLabel.MSE, AxisLabel.SSIM, AxisLabel.ARI]
        pngs = [ExportedFiles.CONTACT_TIME_PNG, ExportedFiles.MSE_PNG, ExportedFiles.SSIM_PNG, ExportedFiles.ARI_PNG]

        for subdir in common_intervals:
            path = mkdir(build_path(self.f9_results_compare_strategies, subdir))
            acc_interval_path = build_path(acc_results_path, subdir)
            sli_interval_path = build_path(sli_results_path, subdir)
            acc_ks = np.loadtxt(build_path(acc_interval_path, ExportedFiles.KS_CHOSEN.value), delimiter=',', dtype=int)
            sli_ks = np.loadtxt(build_path(sli_interval_path, ExportedFiles.KS_CHOSEN.value), delimiter=',', dtype=int)

            # The acc_columns and sli_columns will be in the following order:
            # aic_intra|aic_inter|bic_intra|bic_inter|best_intra|best_inter
            acc_columns = []
            sli_columns = []

            for k in acc_ks:
                acc_columns.append(f'{column_k(k)}|{sources()[1]}')
                acc_columns.append(f'{column_k(k)}|{sources()[2]}')
            for k in sli_ks:
                sli_columns.append(f'{column_k(k)}|{sources()[1]}')
                sli_columns.append(f'{column_k(k)}|{sources()[2]}')

            for index, file in enumerate(csvs):
                acc_dataframe = pd.read_csv(build_path(acc_interval_path, file.value), sep=',')
                sli_dataframe = pd.read_csv(build_path(sli_interval_path, file.value), sep=',')

                all_pairs_mean = acc_dataframe[sources()[0]].iloc[1]

                acc_means = acc_dataframe[acc_columns].iloc[1].to_numpy()
                sli_means = sli_dataframe[sli_columns].iloc[1].to_numpy()

                plot_strategy_comparison(all_pairs_mean,
                                         acc_means, sli_means,
                                         acc_ks, sli_ks,
                                         axis_label=axis[index],
                                         path=build_path(path, pngs[index].value))

    def time_evolution(self, choice_index: int = 2):
        fed_acc_results_path = Path.f9_results(self.dataset.name, LearningApproach.FED, WindowStrategyType.ACC)
        fed_sli_results_path = Path.f9_results(self.dataset.name, LearningApproach.FED, WindowStrategyType.SLI)
        cen_acc_results_path = Path.f9_results(self.dataset.name, LearningApproach.CEN, WindowStrategyType.ACC)
        cen_sli_results_path = Path.f9_results(self.dataset.name, LearningApproach.CEN, WindowStrategyType.SLI)

        fed_acc_subdir_list = get_subdir_list(fed_acc_results_path)
        fed_sli_subdir_list = get_subdir_list(fed_sli_results_path)
        cen_acc_subdir_list = get_subdir_list(cen_acc_results_path)
        cen_sli_subdir_list = get_subdir_list(cen_sli_results_path)

        if not fed_acc_subdir_list or not fed_sli_subdir_list or not cen_acc_subdir_list or not cen_sli_subdir_list:
            print('\n[INFO] To execute the time_evolution function, it is mandatory to first perform the training and '
                  'validation of the FL-based/ACC, FL-based/SLI, Centralized/ACC and Centralized/SLI combinations.\n')
            return

        acc_common_intervals = fed_acc_subdir_list.intersection(cen_acc_subdir_list)
        sli_common_intervals = fed_sli_subdir_list.intersection(cen_sli_subdir_list)
        common_intervals = acc_common_intervals.intersection(sli_common_intervals)

        csvs = [ExportedFiles.CONTACT_TIME_CSV, ExportedFiles.MSE_CSV, ExportedFiles.SSIM_CSV, ExportedFiles.ARI_CSV]
        axis = [AxisLabel.CONTACT_TIME, AxisLabel.MSE, AxisLabel.SSIM, AxisLabel.ARI]
        pngs = [ExportedFiles.CONTACT_TIME_PNG, ExportedFiles.MSE_PNG, ExportedFiles.SSIM_PNG, ExportedFiles.ARI_PNG]

        for index, file in enumerate(csvs):

            all_pairs_lowers = []
            all_pairs_means = []
            all_pairs_uppers = []

            fed_acc_intra_lowers = []
            fed_acc_intra_means = []
            fed_acc_intra_uppers = []

            fed_acc_inter_lowers = []
            fed_acc_inter_means = []
            fed_acc_inter_uppers = []

            fed_sli_intra_lowers = []
            fed_sli_intra_means = []
            fed_sli_intra_uppers = []

            fed_sli_inter_lowers = []
            fed_sli_inter_means = []
            fed_sli_inter_uppers = []

            cen_acc_intra_lowers = []
            cen_acc_intra_means = []
            cen_acc_intra_uppers = []

            cen_acc_inter_lowers = []
            cen_acc_inter_means = []
            cen_acc_inter_uppers = []

            cen_sli_intra_lowers = []
            cen_sli_intra_means = []
            cen_sli_intra_uppers = []

            cen_sli_inter_lowers = []
            cen_sli_inter_means = []
            cen_sli_inter_uppers = []

            for subdir in sorted(common_intervals):
                fed_acc_interval_path = build_path(fed_acc_results_path, subdir)
                fed_sli_interval_path = build_path(fed_sli_results_path, subdir)
                cen_acc_interval_path = build_path(cen_acc_results_path, subdir)
                cen_sli_interval_path = build_path(cen_sli_results_path, subdir)
                fed_acc_ks = np.loadtxt(build_path(fed_acc_interval_path, ExportedFiles.KS_CHOSEN.value),
                                        delimiter=',', dtype=int)
                fed_sli_ks = np.loadtxt(build_path(fed_sli_interval_path, ExportedFiles.KS_CHOSEN.value),
                                        delimiter=',', dtype=int)
                cen_acc_ks = np.loadtxt(build_path(cen_acc_interval_path, ExportedFiles.KS_CHOSEN.value),
                                        delimiter=',', dtype=int)
                cen_sli_ks = np.loadtxt(build_path(cen_sli_interval_path, ExportedFiles.KS_CHOSEN.value),
                                        delimiter=',', dtype=int)
                fed_acc_k = fed_acc_ks[choice_index]
                fed_sli_k = fed_sli_ks[choice_index]
                cen_acc_k = cen_acc_ks[choice_index]
                cen_sli_k = cen_sli_ks[choice_index]

                # The fed_columns and cen_columns will be in the following order:
                # choice_intra|choice_inter
                fed_acc_columns = []
                fed_sli_columns = []
                cen_acc_columns = []
                cen_sli_columns = []

                fed_acc_columns.append(f'{column_k(int(fed_acc_k))}|{sources()[1]}')
                fed_acc_columns.append(f'{column_k(int(fed_acc_k))}|{sources()[2]}')
                fed_sli_columns.append(f'{column_k(int(fed_sli_k))}|{sources()[1]}')
                fed_sli_columns.append(f'{column_k(int(fed_sli_k))}|{sources()[2]}')
                cen_acc_columns.append(f'{column_k(int(cen_acc_k))}|{sources()[1]}')
                cen_acc_columns.append(f'{column_k(int(cen_acc_k))}|{sources()[2]}')
                cen_sli_columns.append(f'{column_k(int(cen_sli_k))}|{sources()[1]}')
                cen_sli_columns.append(f'{column_k(int(cen_sli_k))}|{sources()[2]}')

                fed_acc_dataframe = pd.read_csv(build_path(fed_acc_interval_path, file.value), sep=',')
                fed_sli_dataframe = pd.read_csv(build_path(fed_sli_interval_path, file.value), sep=',')
                cen_acc_dataframe = pd.read_csv(build_path(cen_acc_interval_path, file.value), sep=',')
                cen_sli_dataframe = pd.read_csv(build_path(cen_sli_interval_path, file.value), sep=',')

                all_pairs_lowers.append(fed_acc_dataframe[sources()[0]].iloc[0])
                all_pairs_means.append(fed_acc_dataframe[sources()[0]].iloc[1])
                all_pairs_uppers.append(fed_acc_dataframe[sources()[0]].iloc[2])

                fed_acc_dataframe = fed_acc_dataframe[fed_acc_columns]
                fed_sli_dataframe = fed_sli_dataframe[fed_sli_columns]
                cen_acc_dataframe = cen_acc_dataframe[cen_acc_columns]
                cen_sli_dataframe = cen_sli_dataframe[cen_sli_columns]

                fed_acc_intra_lowers.append(fed_acc_dataframe[fed_acc_columns[0]].iloc[0])
                fed_acc_intra_means.append(fed_acc_dataframe[fed_acc_columns[0]].iloc[1])
                fed_acc_intra_uppers.append(fed_acc_dataframe[fed_acc_columns[0]].iloc[2])

                fed_acc_inter_lowers.append(fed_acc_dataframe[fed_acc_columns[1]].iloc[0])
                fed_acc_inter_means.append(fed_acc_dataframe[fed_acc_columns[1]].iloc[1])
                fed_acc_inter_uppers.append(fed_acc_dataframe[fed_acc_columns[1]].iloc[2])

                fed_sli_intra_lowers.append(fed_sli_dataframe[fed_sli_columns[0]].iloc[0])
                fed_sli_intra_means.append(fed_sli_dataframe[fed_sli_columns[0]].iloc[1])
                fed_sli_intra_uppers.append(fed_sli_dataframe[fed_sli_columns[0]].iloc[2])

                fed_sli_inter_lowers.append(fed_sli_dataframe[fed_sli_columns[1]].iloc[0])
                fed_sli_inter_means.append(fed_sli_dataframe[fed_sli_columns[1]].iloc[1])
                fed_sli_inter_uppers.append(fed_sli_dataframe[fed_sli_columns[1]].iloc[2])

                cen_acc_intra_lowers.append(cen_acc_dataframe[cen_acc_columns[0]].iloc[0])
                cen_acc_intra_means.append(cen_acc_dataframe[cen_acc_columns[0]].iloc[1])
                cen_acc_intra_uppers.append(cen_acc_dataframe[cen_acc_columns[0]].iloc[2])

                cen_acc_inter_lowers.append(cen_acc_dataframe[cen_acc_columns[1]].iloc[0])
                cen_acc_inter_means.append(cen_acc_dataframe[cen_acc_columns[1]].iloc[1])
                cen_acc_inter_uppers.append(cen_acc_dataframe[cen_acc_columns[1]].iloc[2])

                cen_sli_intra_lowers.append(cen_sli_dataframe[cen_sli_columns[0]].iloc[0])
                cen_sli_intra_means.append(cen_sli_dataframe[cen_sli_columns[0]].iloc[1])
                cen_sli_intra_uppers.append(cen_sli_dataframe[cen_sli_columns[0]].iloc[2])

                cen_sli_inter_lowers.append(cen_sli_dataframe[cen_sli_columns[1]].iloc[0])
                cen_sli_inter_means.append(cen_sli_dataframe[cen_sli_columns[1]].iloc[1])
                cen_sli_inter_uppers.append(cen_sli_dataframe[cen_sli_columns[1]].iloc[2])

            all_pairs = [all_pairs_lowers, all_pairs_means, all_pairs_uppers]
            fed_acc_intra = [fed_acc_intra_lowers, fed_acc_intra_means, fed_acc_intra_uppers]
            fed_acc_inter = [fed_acc_inter_lowers, fed_acc_inter_means, fed_acc_inter_uppers]
            fed_sli_intra = [fed_sli_intra_lowers, fed_sli_intra_means, fed_sli_intra_uppers]
            fed_sli_inter = [fed_sli_inter_lowers, fed_sli_inter_means, fed_sli_inter_uppers]
            cen_acc_intra = [cen_acc_intra_lowers, cen_acc_intra_means, cen_acc_intra_uppers]
            cen_acc_inter = [cen_acc_inter_lowers, cen_acc_inter_means, cen_acc_inter_uppers]
            cen_sli_intra = [cen_sli_intra_lowers, cen_sli_intra_means, cen_sli_intra_uppers]
            cen_sli_inter = [cen_sli_inter_lowers, cen_sli_inter_means, cen_sli_inter_uppers]

            plot_time_evolution(all_pairs, fed_acc_intra, fed_acc_inter, fed_sli_intra, fed_sli_inter,
                                cen_acc_intra, cen_acc_inter, cen_sli_intra, cen_sli_inter,
                                axis_label=axis[index],
                                path=build_path(self.f9_results_time_evolution, pngs[index].value))


class ValidationHelper:

    def __init__(self, interval: int, path: str):
        self.interval = interval
        self.path = path
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

        self.ks_chosen = np.array([chosen_aic, chosen_bic, chosen_best])
        np.savetxt(build_path(self.path, ExportedFiles.KS_CHOSEN.value), self.ks_chosen, delimiter=',', fmt='%d')
