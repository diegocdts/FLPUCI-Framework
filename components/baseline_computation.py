import json

import numpy as np
import pandas as pd

from components.sample_generation import SampleHandler
from inner_functions.path import get_file_path, sorted_files, interval_csv, path_exists, metric_interval_csv, \
    interval_json
from inner_types.data import Dataset
from inner_types.path import Path
from inner_types.validation import ImageMetric

from skimage.metrics import mean_squared_error as f_mse, structural_similarity as f_ssim
from sklearn.metrics.cluster import adjusted_rand_score as f_ari


def discrete_pixels(pixels: np.array):
    """
    Converts the values of pixels to the range [0, 255]
    :param pixels: The numpy array of pixels
    :return: A numpy array with converted values of pixels
    """
    pixels = pixels * 255
    pixels = pixels.astype("int64")
    return pixels


def add_contact_time(dictionary, tuple_of_users, contact_time_value):
    if dictionary.get(tuple_of_users) is not None:
        contact_time_value = dictionary.get(tuple_of_users) + contact_time_value
    dictionary[tuple_of_users] = contact_time_value


def export_contact_time(dictionary, output_file_path):
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
        Gets the last interval in the trace as a index
        :return: The last interval index
        """
        root = Path.f3_dm(self.dataset.name)
        path = get_file_path(root, sorted_files(root)[0])
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

            file_path = get_file_path(self.f2_data, file_name)
            output_file_path = get_file_path(self.f4_entry_exit, file_name)

            with open(file_path) as input_file:
                file_lines = input_file.readlines()[1:]

                if len(file_lines) > 0:

                    first_line = file_lines[0].split(',')
                    previous_interval = int(first_line[0])
                    x, y = float(first_line[1]), float(first_line[2])
                    y_index = int(y / self.dataset.resolution[0]) + y_padding
                    x_index = int(x / self.dataset.resolution[1]) + x_padding
                    previous_cell = (x_index * self.dataset.height) + y_index

                    entry_cell, exit_cell = int(first_line[3]), int(first_line[3])
                    new_lines = "interval,cell,entry,exit\n"

                    for line in file_lines:
                        split = line.split(',')
                        current_interval, x, y, time = int(split[0]), float(split[1]), float(split[2]), int(split[3])
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
        file_name = get_file_path(self.f3_dm, file_list[0])
        displacement_matrix = pd.read_csv(file_name)

        last_interval = len(displacement_matrix)

        for interval in range(last_interval):

            output_file_path = get_file_path(self.f5_interval_entry_exit, interval_csv(interval))

            if not path_exists(output_file_path):

                print(' > interval_entry_exit:', interval)
                df = pd.DataFrame(columns=['interval', 'id', 'cell', 'entry', 'exit'])

                for file_index, file_name in enumerate(file_list):

                    file_path = get_file_path(self.f4_entry_exit, file_name)
                    cell_entry_exit = pd.read_csv(file_path)
                    cell_entry_exit = cell_entry_exit[cell_entry_exit.interval == interval]
                    cell_entry_exit['id'] = file_index
                    df = df.append(cell_entry_exit)
                df.to_csv(output_file_path, index=False)

    def contact_time(self):
        """
        Computes the contact time between each pair of users at each interval
        """
        for interval in range(self.last_interval):
            output_file_path = get_file_path(self.f6_contact_time, interval_json(interval))

            if not path_exists(output_file_path):

                print(' > contact_time:', interval)
                file_path = get_file_path(self.f5_interval_entry_exit, interval_csv(interval))
                file_df = pd.read_csv(file_path)

                contact_times = {}

                for index_i, series_i in file_df.iterrows():
                    id_i = series_i.id
                    aux_df = file_df[file_df.id != series_i.id]
                    aux_df = aux_df[aux_df.cell == series_i.cell]
                    aux_df = aux_df[aux_df.entry >= series_i.entry]
                    aux_df = aux_df[aux_df.entry < series_i.exit]
                    for index_j, series_j in aux_df.iterrows():
                        id_j = series_j.id
                        first_exit = min(series_i.exit, series_j.exit)
                        contact_time_value = first_exit - series_j.entry
                        add_contact_time(contact_times, (id_i, id_j), contact_time_value)
                        add_contact_time(contact_times, (id_j, id_i), contact_time_value)
                    del aux_df
                del file_df
                export_contact_time(contact_times, output_file_path)

    def image_metrics(self):
        """
        Computes the similarity and dissimilarity metrics between each pair of users at each interval
        """
        win_size = 3

        for interval in range(self.last_interval):

            mse_output_path = get_file_path(self.f7_metrics, metric_interval_csv(ImageMetric.MSE, interval))
            ssim_output_path = get_file_path(self.f7_metrics, metric_interval_csv(ImageMetric.SSIM, interval))
            ari_output_path = get_file_path(self.f7_metrics, metric_interval_csv(ImageMetric.ARI, interval))

            if not path_exists(mse_output_path):
                print(' > image_metrics:', interval)
                samples = []

                for file_name in sorted_files(self.f3_dm):

                    file_path = get_file_path(self.f3_dm, file_name)

                    samples_from_interval = self.sample_handler.get_samples(file_path, interval, interval + 1)
                    if len(samples_from_interval) > 0:
                        samples.append(samples_from_interval[0])

                total_samples = len(samples)
                if total_samples > 0:

                    mse = np.zeros(shape=(total_samples, total_samples))
                    ssim = np.zeros(shape=(total_samples, total_samples))
                    ari = np.zeros(shape=(total_samples, total_samples))

                    for index_i, image_i in enumerate(samples):
                        image_i = np.squeeze(image_i)

                        for index_j, image_j in enumerate(samples):
                            image_j = np.squeeze(image_j)

                            if index_i == index_j:
                                continue

                            mse[index_i, index_j] = f_mse(image_i, image_j)
                            data_range = max(image_i.max(), image_j.max()) - min(image_i.min(), image_j.min())
                            ssim[index_i, index_j] = f_ssim(image_i, image_j, win_size=win_size, data_range=data_range)
                            ari[index_i, index_j] = f_ari(
                                discrete_pixels(image_i.reshape(self.dataset.width * self.dataset.height)),
                                discrete_pixels(image_j.reshape(self.dataset.width * self.dataset.height)))
                    mse_df = pd.DataFrame(mse)
                    mse_df.to_csv(mse_output_path, index=False)
                    ssim_df = pd.DataFrame(ssim)
                    ssim_df.to_csv(ssim_output_path, index=False)
                    ari_df = pd.DataFrame(ari)
                    ari_df.to_csv(ari_output_path, index=False)


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
