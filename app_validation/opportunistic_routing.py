import os.path
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.plots import plot_opportunistic_routing_metric
from inner_functions.path import build_path


class RoutingMetricAnalysis:

    def __init__(self, report_root):
        self.report_root = report_root

        if '/sfc' in self.report_root or '/rt' in self.report_root:
            self.routers = ['Epidemic', 'Bubblerap', 'PC - 4h update']
            self.ttl_values = [180, 360, 540, 720]
        else:
            self.routers = ['Epidemic', 'Bubblerap', 'PC - 40min update']
            self.ttl_values = [30, 40, 50, 60]


        self.reports_path = [build_path(report_root, protocol) for protocol in self.routers]

        self.reports = ['CreatedMessagesReport', 'DeliveredMessagesReport', 'EventLogReport']

        self.ttls = [f'ttl_{value}' for value in self.ttl_values]

        self.all_paths = self.set_dict_paths()

    def set_dict_paths(self):
        router_paths = defaultdict(dict)
        for index, router in enumerate(self.routers):
            report_dict = defaultdict(dict)
            for report in self.reports:
                for ttl in self.ttls:
                    files_name = [build_path(self.reports_path[index], f'{router}_{ttl}_{report}.txt')]
                    report_dict[report][ttl] = files_name
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
                for ttl_name, paths_list in report_dict.items():
                    for path in paths_list:
                        if os.path.exists(path):
                            with open(path, 'r') as file:
                                lines = file.readlines()[1:]
                                if report_name == self.reports[0]:
                                    num_created = len(lines)
                                    protocol_dict[ttl_name].append(num_created)
                                    created_dict[router_name] = protocol_dict
                                if report_name == self.reports[1]:
                                    num_delivered = len(lines)
                                    hops = []
                                    latencies = []
                                    for line in lines:
                                        split = line.split(' ')
                                        hops.append(int(split[3]))
                                        latencies.append(round(float(split[4]), 2))
                                    protocol_dict[ttl_name].append(num_delivered)

                                    protocol_dict1[ttl_name] += remove_outliers(hops)
                                    protocol_dict2[ttl_name] += remove_outliers(latencies)

                                    delivered_dict[router_name] = protocol_dict
                                    hops_dict[router_name] = protocol_dict1
                                    latency_dict[router_name] = protocol_dict2
                                if report_name == self.reports[2]:
                                    relays = 0
                                    for line in lines:
                                        if line.endswith(' R\n'):
                                            relays += 1
                                    protocol_dict[ttl_name].append(relays)
                                    relayed_dict[router_name] = protocol_dict
        delivery_prob = delivered_dict
        overhead = relayed_dict

        for router_name, router_dict in delivered_dict.items():
            for ttl_name, ttl_list in router_dict.items():
                delivered = np.array(delivery_prob[router_name][ttl_name])
                created = np.array(created_dict[router_name][ttl_name])
                relayed = np.array(relayed_dict[router_name][ttl_name])
                latency = np.array(latency_dict[router_name][ttl_name]) / 60
                hops = np.array(hops_dict[router_name][ttl_name])

                delivery_prob[router_name][ttl_name] = (delivered / created)
                latency_dict[router_name][ttl_name] = latency
                hops_dict[router_name][ttl_name] = hops
                overhead[router_name][ttl_name] = (relayed-delivered)/delivered

        plot_opportunistic_routing_metric(delivery_prob, self.ttl_values, 'Delivery Probability', self.report_root)
        plot_opportunistic_routing_metric(overhead, self.ttl_values, 'Overhead', self.report_root)
        plot_opportunistic_routing_metric(hops_dict, self.ttl_values, 'Avg hops', self.report_root)
        plot_opportunistic_routing_metric(latency_dict, self.ttl_values, 'Latency (minutes)', self.report_root)

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
