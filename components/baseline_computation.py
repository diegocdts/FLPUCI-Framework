import json

import numpy as np
import pandas as pd

from components.sample_generation import SampleHandler
from inner_functions.path import build_path, sorted_files, interval_csv, path_exists, metric_interval_json, \
    interval_json
from inner_types.data import Dataset
from inner_types.path import Path
from inner_types.validation import HeatmapMetric

from itertools import product
from skimage.metrics import mean_squared_error as f_mse, structural_similarity as f_ssim
from sklearn.metrics.cluster import adjusted_rand_score as f_ari

dictionary_key = lambda i, j: f'{i}_{j}'


def discrete_pixels(pixels: np.array):
    """
    Converts the values of pixels to the range [0, 255]
    :param pixels: The numpy array of pixels
    :return: A numpy array with converted values of pixels
    """
    pixels = pixels * 255
    pixels = pixels.astype("int64")
    return pixels


def add_dictionary_entry(dictionary, index_i, index_j, value):
    """
    Adds a new entry in a provided dictionary
    :param dictionary: A dictionary where the key is the combination of user indexes
    :param index_i: The index of the user i
    :param index_j: The index of the user j
    :param value: The value to be added
    """
    key = dictionary_key(index_i, index_j)
    if dictionary.get(key) is not None:
        value = dictionary.get(key) + value
    dictionary[key] = float(value)


def export_dictionary(dictionary, output_file_path):
    """
    Exports a provided dictionary to a json file
    :param dictionary: A dictionary to export
    :param output_file_path: The path of the file
    """
    with open(output_file_path, 'w') as json_file:
        json.dump(dictionary, json_file)


class BaselineComputation:

    def __init__(self, dataset: Dataset):
        """
        Runs baseline computations regarding contact time and similarity and dissimilarity metrics among nodes
        :param dataset: A Dataset object
        """
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.f4_entry_exit = Path.f4_entry_exit(dataset.name)
        self.f5_interval_entry_exit = Path.f5_interval_entry_exit(dataset.name)
        self.f6_contact_time = Path.f6_contact_time(dataset.name)
        self.f7_metrics = Path.f7_metrics(dataset.name)

        self.dataset = dataset
        self.sample_handler = SampleHandler(dataset)
        self.last_interval = self.get_last_intervals()

    def get_last_intervals(self):
        """
        Gets the last interval in the trace as an index
        :return: The last interval index
        """
        root = Path.f3_dm(self.dataset.name)
        path = build_path(root, sorted_files(root)[0])
        df = pd.read_csv(path)
        last_interval = len(df)
        return last_interval

    def cell_entry_exit(self):
        """
        Computes the entry and exit times of each user in cells
        """
        y_padding = 1 if self.dataset.paddingYX[0] else 0
        x_padding = 1 if self.dataset.paddingYX[1] else 0

        for file_name in sorted_files(self.f2_data):

            file_path = build_path(self.f2_data, file_name)
            output_file_path = build_path(self.f4_entry_exit, file_name)

            with open(file_path) as input_file:
                file_lines = input_file.readlines()[1:]
                new_lines = "interval,cell,entry,exit\n"

                if len(file_lines) > 0:

                    first_line = file_lines[0].replace('\n', '').split(',')
                    previous_interval = int(first_line[0])
                    x, y = float(first_line[1]), float(first_line[2])
                    y_index = int(y / self.dataset.resolution[0]) + y_padding
                    x_index = int(x / self.dataset.resolution[1]) + x_padding
                    previous_cell = (x_index * self.dataset.height) + y_index

                    entry_cell, exit_cell = float(first_line[3]), float(first_line[3])

                    for line in file_lines:
                        split = line.replace('\n', '').split(',')
                        current_interval, x, y, time = int(split[0]), float(split[1]), float(split[2]), float(split[3])
                        y_index = int(y / self.dataset.resolution[0]) + y_padding
                        x_index = int(x / self.dataset.resolution[1]) + x_padding
                        current_cell = (x_index * self.dataset.height) + y_index

                        if current_cell != previous_cell or current_interval != previous_interval:
                            new_lines += f'{previous_interval},{previous_cell},{entry_cell},{exit_cell}\n'
                            previous_interval = current_interval
                            previous_cell = current_cell
                            entry_cell = time
                        exit_cell = time
                    new_lines += f'{previous_interval},{previous_cell},{entry_cell},{exit_cell}\n'
                with open(output_file_path, 'w') as output_file:
                    output_file.write(new_lines)

    def interval_entry_exit(self):
        """
        Separates the users' entry and exit into cells by intervals
        """
        file_list = sorted_files(self.f4_entry_exit)
        file_name = build_path(self.f3_dm, file_list[0])
        displacement_matrix = pd.read_csv(file_name)

        last_interval = len(displacement_matrix)

        for interval in range(last_interval):

            output_file_path = build_path(self.f5_interval_entry_exit, interval_csv(interval))

            if not path_exists(output_file_path):

                print(' > interval_entry_exit:', interval)
                df = pd.DataFrame(columns=['interval', 'id', 'cell', 'entry', 'exit'])

                for file_index, file_name in enumerate(file_list):
                    file_path = build_path(self.f4_entry_exit, file_name)
                    cell_entry_exit = pd.read_csv(file_path)
                    cell_entry_exit = cell_entry_exit[cell_entry_exit.interval == interval]
                    cell_entry_exit['id'] = file_index
                    df = pd.concat([df, cell_entry_exit], ignore_index=True)
                df.to_csv(output_file_path, index=False)

    def contact_time(self):
        """
        Computes the contact time between each pair of users at each interval
        """
        for interval in range(self.last_interval):
            output_file_path = build_path(self.f6_contact_time, interval_json(interval))

            if not path_exists(output_file_path):

                print(' > contact_time:', interval)
                file_path = build_path(self.f5_interval_entry_exit, interval_csv(interval))
                file_df = pd.read_csv(file_path)

                contact_times = {}

                for index_i, series_i in file_df.iterrows():
                    id_i = int(series_i.id)
                    aux_df = file_df[file_df.id != series_i.id]
                    aux_df = aux_df[aux_df.cell == series_i.cell]
                    aux_df = aux_df[aux_df.exit > series_i.entry]
                    aux_df = aux_df[aux_df.entry < series_i.exit]

                    for index_j, series_j in aux_df.iterrows():
                        id_j = int(series_j.id)
                        last_entry = max(series_i.entry, series_j.entry)
                        first_exit = min(series_i.exit, series_j.exit)
                        contact_time_value = first_exit - last_entry
                        add_dictionary_entry(contact_times, id_i, id_j, contact_time_value)
                        add_dictionary_entry(contact_times, id_j, id_i, contact_time_value)
                    del aux_df

                for id_i, id_j in product(file_df.id, repeat=2):
                    id_i, id_j = int(id_i), int(id_j)
                    if id_i != id_j:
                        if contact_times.get(dictionary_key(id_i, id_j)) is None:
                            add_dictionary_entry(contact_times, id_i, id_j, 0.0)

                del file_df
                export_dictionary(contact_times, output_file_path)

    def image_metrics(self):
        """
        Computes the similarity and dissimilarity metrics between each pair of users at each interval
        """
        win_size = 3

        for interval in range(self.last_interval):

            mse_output_path = build_path(self.f7_metrics, metric_interval_json(HeatmapMetric.MSE, interval))
            ssim_output_path = build_path(self.f7_metrics, metric_interval_json(HeatmapMetric.SSIM, interval))
            ari_output_path = build_path(self.f7_metrics, metric_interval_json(HeatmapMetric.ARI, interval))

            if not path_exists(mse_output_path):
                print(' > image_metrics:', interval)
                samples = []
                user_indexes = []

                for index, file_name in enumerate(sorted_files(self.f3_dm)):

                    file_path = build_path(self.f3_dm, file_name)

                    samples_from_interval = self.sample_handler.get_samples(file_path, interval, interval + 1)
                    if len(samples_from_interval) > 0:
                        samples.append(samples_from_interval[0])
                        user_indexes.append(index)

                total_samples = len(samples)
                if total_samples > 0:

                    mse = {}
                    ssim = {}
                    ari = {}

                    for i, index_i in enumerate(user_indexes):
                        image_i = samples[i]
                        image_i = np.squeeze(image_i)

                        for j, index_j in enumerate(user_indexes):

                            if index_i == index_j:
                                continue

                            image_j = samples[j]
                            image_j = np.squeeze(image_j)

                            mse_value = f_mse(image_i, image_j)
                            add_dictionary_entry(mse, index_i, index_j, mse_value)

                            data_range = max(image_i.max(), image_j.max()) - min(image_i.min(), image_j.min())
                            ssim_value = f_ssim(image_i, image_j, win_size=win_size, data_range=data_range)
                            add_dictionary_entry(ssim, index_i, index_j, ssim_value)

                            ari_value = f_ari(
                                discrete_pixels(image_i.reshape(self.dataset.width * self.dataset.height)),
                                discrete_pixels(image_j.reshape(self.dataset.width * self.dataset.height)))
                            add_dictionary_entry(ari, index_i, index_j, ari_value)
                    export_dictionary(mse, mse_output_path)
                    export_dictionary(ssim, ssim_output_path)
                    export_dictionary(ari, ari_output_path)


def compute_baseline(dataset: Dataset):
    """
    Runs the BaselineComputation script for a dataset
    :param dataset: The Dataset object
    """
    baseline = BaselineComputation(dataset)
    baseline.cell_entry_exit()
    baseline.interval_entry_exit()
    baseline.contact_time()
    baseline.image_metrics()
