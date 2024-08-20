import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from inner_functions.path import sorted_files, build_path
from inner_types.data import Dataset
from inner_types.path import Path


def compare_samples_reconstructions(samples, reconstructions):
    """
    Compares heatmap samples to their reconstructions
    :param samples: The list of heatmaps
    :param reconstructions: The list of reconstructions
    """
    if len(samples) > 0 and len(reconstructions) > 0:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        for i in range(len(samples)):
            plt.subplot(1, 2, 1)  # Subplot for sample
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(samples[i])
            plt.title("Sample")

            plt.subplot(1, 2, 2)  # Subplot for reconstruction
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(reconstructions[i])
            plt.title("Reconstruction")

            plt.show()


def sigmoid(pixels: np.array):
    """
    Applies the sigmoid function over the data to undo the logit transformation
    :param pixels: The pixels of a heatmap
    :return: The new pixels of the heatmap
    """
    return 1 / (1 + np.exp(-pixels))


def min_max_scaling(pixels: np.array):
    return (pixels - pixels.min()) / (pixels.max() - pixels.min())


def reshape(samples: np.array):
    """
    Reshapes the numpy array of samples from 3 to 4 dimensions and from 4 to 5 dimensions
    :param samples: The numpy array of samples
    :return: The reshaped numpy array of samples
    """
    if len(samples.shape) == 4:
        samples = np.reshape(samples, (samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3], 1))
    elif len(samples.shape) == 3:
        samples = np.reshape(samples, (samples.shape[0], samples.shape[1], samples.shape[2], 1))
    return samples


def heat_maps_samples_view(samples, rows, columns):
    """
    Shows a matrix of heatmap samples
    :param samples: The list of samples to show
    :param rows: Number of rows
    :param columns: Number of columns
    """
    plt.figure(figsize=(40, 20))
    for i in range(int(rows * columns)):
        i = int(i)
        plt.subplot(rows, columns, i + 1)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i])
    plt.show()


class SampleHandler:

    def __init__(self, dataset: Dataset):
        """
        Prepares users samples to be used
        :param dataset: A Dataset object
        """
        self.dataset = dataset
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.set_height_width()

    def set_height_width(self):
        """
        Sets the height and width attributes of the dataset
        """
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = 0, 0
        for file_name in sorted_files(self.f2_data):
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path)
            if df.x.min() < min_x:
                min_x = df.x.min()
            if df.y.min() < min_y:
                min_y = df.y.min()
            if df.x.max() > max_x:
                max_x = df.x.max()
            if df.y.max() > max_y:
                max_y = df.y.max()
            del df
        float_height = (max_y - min_y) / self.dataset.resolution[0]
        float_width = (max_x - min_x) / self.dataset.resolution[1]
        self.dataset.set_height_width(float_height, float_width)

    def get_datasets(self, start_window: int, end_window: int):
        """
        Gets the user datasets of samples inside a window
        :param start_window: The interval that indicates the start of the window to get samples
        :param end_window: The interval that indicates the end of the window to get samples
        :return: The list of user datasets and the indexes of users
        """
        user_names = []
        user_indexes = []
        datasets = []
        for index, file_name in enumerate(sorted_files(self.f3_dm)):
            file_path = build_path(self.f3_dm, file_name)
            user_samples = self.get_samples(file_path, start_window, end_window)
            if len(user_samples) > 0:
                user_names.append(f'{file_name}'.replace('.csv', '').replace('.txt', ''))
                user_indexes.append(index)
                datasets.append(user_samples)
            del user_samples
        return datasets, np.array(user_indexes), np.array(user_names)

    def get_samples(self, file_path: str, start_window: int, end_window: int, add_empty: bool = False):
        """
        Gets samples of a user inside a window
        :param file_path: Path of the displacement matrix
        :param start_window: The interval that indicates the start of the window to get samples
        :param end_window: The interval that indicates the end of the window to get samples
        :param add_empty: Bool flag to add or not empty samples
        :return: The list of samples
        """
        samples = []
        with open(file_path) as file:
            intervals = file.readlines()[1:]
            intervals = intervals[start_window:end_window]
            for interval in intervals:
                sample = np.array(interval.split(','), dtype="float64")
                if sample.max() > sample.min():
                    sample = min_max_scaling(sample)
                    sample = sample.reshape(self.dataset.width, self.dataset.height)
                    samples.append(sample)
        samples = np.array(samples)
        return reshape(samples)

    def samples_as_list(self, start_window: int, end_window: int):
        """
        Returns the user datasets of samples as a flat list of samples
        :param start_window: The interval that indicates the start of the window to get samples
        :param end_window: The interval that indicates the end of the window to get samples
        :return: Flat list of samples and the indexes of users
        """
        datasets, user_indexes, user_names = self.get_datasets(start_window, end_window)
        samples = []
        for dataset in datasets:
            for sample in dataset:
                samples.append(sample)
                del sample
            del dataset
        del datasets
        return np.array(samples), user_indexes, user_names

    def random_dataset(self):
        """
        Returns a random user dataset of samples
        :return: The dataset of samples of a random user
        """

        def get_random():
            total_users = len(sorted_files(self.f3_dm))
            file_name = sorted_files(self.f3_dm)[random.randrange(total_users)]
            file_path = build_path(self.f3_dm, file_name)
            single_dataset = self.get_samples(file_path, 0, 1, add_empty=True)
            return single_dataset

        dataset = get_random()
        while len(dataset) < 1:
            dataset = get_random()
        return dataset
