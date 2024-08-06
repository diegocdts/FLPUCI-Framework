import numpy as np
import pandas as pd

from app_validation.plots import stacked_columns
from inner_functions.path import build_path, interval_dir, path_exists
from inner_types.names import ExportedFiles
from inner_types.path import Path, labels_for_k


class EpidemicAnalysis:

    def __init__(self, dataset, approach, strategy_type, choice_index: int = 2):
        self.dataset = dataset
        self.f9_results = Path.f9_results(dataset.name, approach, strategy_type, dataset.proximal_term)
        self.choice_index = choice_index

    def analysis(self, path_time_infection: str):
        interval_size = self.dataset.hours_per_interval * 3600

        time_infection = pd.read_csv(path_time_infection, names=['time', 'node'], delimiter=' ')
        max_time = time_infection.time.max()

        interval_threshold = 0
        interval = 0

        while interval_threshold + interval_size < max_time:
            interval_infection = time_infection[(interval_threshold <= time_infection.time) &
                                                (time_infection.time < interval_threshold + interval_size)]
            next_interval_infection = time_infection[(interval_threshold + interval_size <= time_infection.time) &
                                                     (time_infection.time < interval_threshold + (2 * interval_size))]

            previous_interval_infection = time_infection[time_infection.time < interval_threshold]

            nodes_per_community, label_counts = self.check_infected_nodes_at_interval(interval_infection,
                                                                                      interval)
            previous_label_counts = self.check_previous_intervals(previous_interval_infection, interval)
            self.check_next_interval(interval_infection, next_interval_infection, interval)

            stacked_columns(nodes_per_community, label_counts, previous_label_counts, interval)

            interval_threshold += interval_size
            interval += 1

    def check_infected_nodes_at_interval(self, interval_infection: pd.DataFrame, interval: int,
                                         is_previous: bool = False):
        interval_path = build_path(self.f9_results, interval_dir(interval))
        user_labels_mapping = self.load_user_labels(interval_path)

        nodes_per_community = user_labels_mapping.groupby('label').size()

        merged = pd.merge(interval_infection, user_labels_mapping, on='node', how='inner')

        label_counts = merged['label'].value_counts()

        distinct_labels = label_counts.count()

        total_labels = user_labels_mapping['label'].nunique()

        interval_to_print = interval if is_previous else f'before {interval}'
        print(
            f'\n\n{len(interval_infection)} nodes were infected in interval {interval_to_print}. '
            f'They were distributed in {distinct_labels} of {total_labels} clusters, where:')
        for label, count in label_counts.items():
            percent = (count / len(interval_infection)) * 100
            print(f'{count} users ({round(percent, 2)}%) were in cluster {label}')

        return nodes_per_community, label_counts

    def check_previous_intervals(self, interval_infection: pd.DataFrame, interval: int):
        nodes_per_community, label_counts = (
                self.check_infected_nodes_at_interval(interval_infection, interval, is_previous=True))

        return label_counts

    def check_next_interval(self, interval_infection: pd.DataFrame, next_interval_infection: pd.DataFrame,
                            interval: int):
        next_interval_path = build_path(self.f9_results, interval_dir(interval + 1))
        if path_exists(next_interval_path):
            next_user_labels_mapping = self.load_user_labels(next_interval_path)

            merged_current = pd.merge(interval_infection, next_user_labels_mapping, on='node', how='inner')
            merged_next = pd.merge(next_interval_infection, next_user_labels_mapping, on='node', how='inner')

            labels_next = merged_next['label'].unique()

            filtered_current = merged_current[merged_current['label'].isin(labels_next)]

            unique_filtered_current = filtered_current['label'].nunique()

            total_next_clusters = next_user_labels_mapping['label'].nunique()

            current_nodes_in_next_clusters = filtered_current[['node', 'label']]

            percent = (len(current_nodes_in_next_clusters) / len(merged_current)) * 100

            print(f'{len(current_nodes_in_next_clusters)} of the {len(interval_infection)} nodes infected in '
                  f'interval {interval} ({round(percent, 2)}%) will be in the same communities as nodes that will '
                  f'be infected in interval {interval + 1} (in {unique_filtered_current} of the '
                  f'{total_next_clusters} communities)')

    def load_user_labels(self, interval_path):
        ks = np.loadtxt(build_path(interval_path, ExportedFiles.KS_CHOSEN.value), delimiter=',',
                        dtype=int)
        k = ks[self.choice_index]
        labels_path = build_path(interval_path, labels_for_k())
        user_labels_mapping = pd.read_csv(build_path(labels_path, f'k_{k}.txt'), names=['label', 'node'], delimiter=' ')
        return user_labels_mapping
