import os
import re

from inner_types.validation import HeatmapMetric


def natural_sort_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split('(\d+)', text)]


def sorted_files(dir_path: str):
    return sorted(os.listdir(dir_path), key=natural_sort_key)

def sorted_list(_list):
    return sorted(_list, key=natural_sort_key)


def build_path(path_to_dir_or_file: str, dir_or_file_name: str):
    return os.path.join(path_to_dir_or_file, dir_or_file_name)


def interval_csv(interval: int):
    return 'interval_{}.csv'.format(interval)


def interval_json(interval: int):
    return 'interval_{}.json'.format(interval)


def metric_interval_csv(metric: HeatmapMetric, interval: int):
    return '{}_interval_{}.csv'.format(metric.value, interval)


def metric_interval_json(metric: HeatmapMetric, interval: int):
    return '{}_interval_{}.json'.format(metric.value, interval)


def start_end_window_dir(start_window, end_window):
    return 'win_{}_{}'.format(start_window, end_window)


def interval_dir(interval):
    return 'interval_{}'.format(interval)


def path_exists(path: str):
    return os.path.exists(path)


def mkdir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_subdir_list(path):
    return set(os.listdir(path))
