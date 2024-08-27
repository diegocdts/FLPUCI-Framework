import os.path
from collections import defaultdict

import numpy as np
import pandas as pd

from app_validation.plots import plot_opportunistic_routing_metric
from inner_functions.path import build_path


class RoutingMetricAnalysis:

    def __init__(self, report_root):
        self.report_root = report_root

        self.protocols = ['epidemic', 'pr', 'pc']

        self.reports_path = [build_path(report_root, protocol) for protocol in self.protocols]

        self.routers = ['EpidemicRouter', 'ProphetRouter', 'PCRouter']

        self.reports = ['CreatedMessagesReport', 'DeliveredMessagesReport', 'EventLogReport']

        self.loads = ['load_5', 'load_3', 'load_1']
        self.x_ticks = ['60s', '30s', '5s']

        self.prefixes = ['pfxA', 'pfxB', 'pfxC']

        self.all_paths = self.set_dict_paths()

    def set_dict_paths(self):
        router_paths = defaultdict(dict)
        for index, router in enumerate(self.routers):
            report_dict = defaultdict(dict)
            for report in self.reports:
                for load in self.loads:
                    files_name = []
                    for prefix in self.prefixes:
                        file_name = build_path(self.reports_path[index], f'{router}_{load}_{prefix}_{report}.txt')
                        files_name.append(file_name)
                    report_dict[report][load] = files_name
                router_paths[router] = report_dict
        return router_paths

    def metrics(self):
        created_dict = defaultdict(dict)
        delivered_dict = defaultdict(dict)
        hops_dict = defaultdict(dict)
        latency_dict = defaultdict(dict)
        relayed_dict = defaultdict(dict)

        for index_router, (router_name, router_dict) in enumerate(self.all_paths.items()):
            for index_report, (report_name, report_dict) in enumerate(router_dict.items()):
                protocol_dict = defaultdict(list)
                protocol_dict1 = defaultdict(list)
                protocol_dict2 = defaultdict(list)
                for load_name, paths_list in report_dict.items():
                    for path in paths_list:
                        if os.path.exists(path):
                            with open(path, 'r') as file:
                                lines = file.readlines()[1:]
                                if report_name == self.reports[0]:
                                    num_created = len(lines)
                                    protocol_dict[load_name].append(num_created)
                                    created_dict[router_name] = protocol_dict
                                if report_name == self.reports[1]:
                                    num_delivered = len(lines)
                                    hops = []
                                    latencies = []
                                    for line in lines:
                                        split = line.split(' ')
                                        hops.append(int(split[3]))
                                        latencies.append(round(float(split[4]), 2))
                                    protocol_dict[load_name].append(num_delivered)

                                    protocol_dict1[load_name] += remove_outliers(hops)
                                    protocol_dict2[load_name] += remove_outliers(latencies)

                                    delivered_dict[router_name] = protocol_dict
                                    hops_dict[router_name] = protocol_dict1
                                    latency_dict[router_name] = protocol_dict2
                                if report_name == self.reports[2]:
                                    relays = 0
                                    for line in lines:
                                        if line.endswith(' R\n'):
                                            relays += 1
                                    protocol_dict[load_name].append(relays)
                                    relayed_dict[router_name] = protocol_dict
        delivery_prob = delivered_dict
        overhead = relayed_dict

        for router_name, router_dict in delivered_dict.items():
            for load_name, load_list in router_dict.items():
                delivered = np.array(delivery_prob[router_name][load_name])
                created = np.array(created_dict[router_name][load_name])
                relayed = np.array(relayed_dict[router_name][load_name])
                print(f'{load_name} {delivered} / {created} = {delivered/created}')
                delivery_prob[router_name][load_name] = delivered/created
                overhead[router_name][load_name] = (relayed-(created+delivered))/delivered

        plot_opportunistic_routing_metric(delivery_prob, self.x_ticks, 'Delivery Probability', self.report_root)
        plot_opportunistic_routing_metric(overhead, self.x_ticks, 'Overhead', self.report_root)
        plot_opportunistic_routing_metric(hops_dict, self.x_ticks, 'Avg hops', self.report_root)
        plot_opportunistic_routing_metric(latency_dict, self.x_ticks, 'Latency', self.report_root)


def remove_outliers(data):
    data = {'values': data}
    df = pd.DataFrame(data)

    q1 = df['values'].quantile(0.25)
    q3 = df['values'].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_clean = df[(df['values'] >= lower_bound) & (df['values'] <= upper_bound)]

    return df_clean['values'].to_numpy().tolist()
