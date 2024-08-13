import shutil

import numpy as np
import pandas as pd

from itertools import product
from inner_functions.path import build_path, sorted_files
from inner_types.names import ExportedFiles
from inner_types.path import Path, labels_for_k


class StrategyInfos:

    def __init__(self, dataset, approach, strategy_type, choice_index: int = 2):
        self.dataset = dataset
        self.f9_results = Path.f9_results(dataset.name, approach, strategy_type, dataset.proximal_term)
        self.f9_info = Path.f9_community_info(dataset.name, approach, strategy_type, dataset.proximal_term)
        self.f9_community_id_maps = Path.f9_community_id_maps(self.f9_info)
        self.f9_previous_community_count = Path.f9_previous_community_count(self.f9_info)
        self.choice_index = choice_index

    def get_community_id_maps(self):
        intervals = sorted_files(self.f9_results)

        for interval in intervals:
            interval_path = build_path(self.f9_results, interval)
            ks = np.loadtxt(build_path(interval_path, ExportedFiles.KS_CHOSEN.value), delimiter=',', dtype=int)
            k = ks[self.choice_index]
            labels_for_k_path = build_path(interval_path, labels_for_k())
            source_path = build_path(labels_for_k_path, f'k_{k}.txt')
            print(interval, source_path)
            destination_path = build_path(self.f9_community_id_maps, f'{interval}.txt')
            shutil.copy(source_path, destination_path)

    def get_previous_community_count(self):
        community_id_maps = sorted_files(self.f9_community_id_maps)

        dict_previous_community_count = {}

        for interval in community_id_maps:
            map_path = build_path(self.f9_community_id_maps, interval)
            count_path = build_path(self.f9_previous_community_count, interval)
            users_label_map = pd.read_csv(map_path, names=['label', 'node'], delimiter=' ')
            labels = users_label_map.label.unique()
            for label in labels:
                community = users_label_map[users_label_map.label == label]
                community_nodes = community['node'].to_numpy()
                for node_i, node_j in product(community_nodes, repeat=2):
                    if node_i != node_j:
                        par_ij = f'{node_i}_{node_j}'
                        if dict_previous_community_count.get(par_ij) is None:
                            dict_previous_community_count[par_ij] = 1
                        else:
                            dict_previous_community_count[par_ij] += 1

            with open(count_path, 'w') as file:
                for key, value in dict_previous_community_count.items():
                    file.write(f'{key} {value}\n')
