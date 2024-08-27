import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from inner_functions.path import interval_dir, build_path


def stacked_columns(current_nodes_per_community, current_label_counts,
                    previous_label_counts, interval, path):
    communities = [f'Community {i}' for i in range(len(current_nodes_per_community))]
    nodes = [current_nodes_per_community[i] for i in range(len(communities))]
    current_infected = [0] * len(current_nodes_per_community)
    previous_infected = [0] * len(current_nodes_per_community)

    for label, count in current_label_counts.items():
        current_infected[label] = count

    for label, count in previous_label_counts.items():
        previous_infected[label] = count

    plt.figure(figsize=(15, 6))
    plt.bar(communities, nodes, color='lightgray',
            label=f'Users not infected until the current interval '
                  f'({sum(nodes) - (sum(previous_infected) + sum(current_infected))})')
    plt.bar(communities, previous_infected, color='yellow',
            label=f'Previously infected users still in the scenario ({sum(previous_infected)})')
    plt.bar(communities, current_infected, color='red', bottom=previous_infected,
            label=f'Users infected in this interval ({sum(current_infected)})')

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    plt.text(-2, y_max + 1, f'Total nodes present at the interval {interval}: {sum(nodes)}\n'
                            f'Total number of users infected and present at the interval {interval}: '
                            f'{sum(previous_infected) + sum(current_infected)}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Communities')
    plt.ylabel('Number of users')
    plt.title(f'Distribution of nodes in communities at interval {interval}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/{interval_dir(interval)}.png')
    plt.close()


def plot_opportunistic_routing_metric(metric: dict, x_ticks: list, title: str, path: str):
    plt.figure(figsize=(5, 6))
    for router_name, router_dict in metric.items():
        means = []
        lower_bounds = []
        upper_bounds = []
        for load_name, load_list in router_dict.items():
            mean = np.mean(load_list)
            conf_interval = stats.norm.interval(0.95, loc=mean, scale=stats.sem(load_list))
            means.append(mean)
            lower_bounds.append(conf_interval[0])
            upper_bounds.append(conf_interval[1])
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        x = [i for i in range(len(x_ticks))]
        plt.errorbar(x, means, yerr=[means-lower_bounds, upper_bounds-means], label=router_name,
                      linewidth=2, capsize=6, capthick=2)
        for index, value in enumerate(means):
            plt.text(x[index], value, f'{round(value, 2)}')
        plt.xticks(x, x_ticks)
    plt.xlabel('TTL (minutes)')
    plt.ylabel(title)
    plt.suptitle(title)
    plt.legend()
    plt.savefig(build_path(path, f'{title}.png'))
    plt.close()
