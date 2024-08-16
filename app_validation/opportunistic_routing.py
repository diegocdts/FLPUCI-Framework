import os

from collections import defaultdict

from inner_functions.path import build_path


class RoutingMetricAnalysis:

    def __init__(self, report_root):
        self.protocols = ['epidemic', 'prophet', 'pc']

        self.reports_path = [build_path(report_root, protocol) for protocol in self.protocols]

        self.routers = ['EpidemicRouter', 'ProphetRouter', 'PCRouter']

        self.reports = ['CreatedMessagesReport', 'DeliveredMessagesReport', 'EventLogReport']

        self.loads = ['load_0.25', 'load_0.5', 'load_0.75', 'load_1']

        self.prefixes = ['pfxM', 'pfxI', 'pfxT', 'pfxW']

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
            for index_report, (report_name, load_dict) in enumerate(router_dict.items()):
                protocol_dict = defaultdict(list)
                protocol_dict1 = defaultdict(list)
                protocol_dict2 = defaultdict(list)
                for load_name, paths_list in load_dict.items():
                    for path in paths_list:
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
                                protocol_dict1[load_name].append(hops)
                                protocol_dict2[load_name].append(latencies)

                                delivered_dict[router_name] = protocol_dict
                                hops_dict[router_name] = protocol_dict1
                                latency_dict[router_name] = protocol_dict2
                            if report_name == self.reports[2]:
                                relays = 0
                                for line in lines:
                                    if line.endswith(' R\n'):
                                        relays += 1
                                    elif line.endswith(' D\n'):
                                        break
                                protocol_dict[load_name].append(relays)
                                relayed_dict[router_name] = protocol_dict
