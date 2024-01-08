import os

from inner_types.validation import ImageMetric


def sorted_files(dir_path: str):
    return sorted(os.listdir(dir_path))


def get_file_path(dir_path: str, file_name: str):
    return os.path.join(dir_path, file_name)


def interval_csv(interval: int):
    return 'interval_{}.csv'.format(interval)


def interval_json(interval: int):
    return 'interval_{}.json'.format(interval)


def metric_interval_csv(metric: ImageMetric, interval: int):
    return '{}_interval_{}.csv'.format(metric.value, interval)


def metric_interval_json(metric: ImageMetric, interval: int):
    return '{}_interval_{}.json'.format(metric.value, interval)


def start_end_window_dir(start_window, end_window):
    return 'win_{}_{}'.format(start_window, end_window)

def path_exists(path: str):
    return os.path.exists(path)
