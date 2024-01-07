import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from inner_functions.path import sorted_files, get_file_path
from inner_types.data import Dataset
from inner_types.path import Path


def compare_samples_reconstructions(samples, reconstructions):
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


def handle_pixels(pixels: np.array):
    if pixels.max() != pixels.min():
        min_value = np.min(pixels)
        max_value = np.max(pixels)
        mapped_pixels = (pixels - min_value) / (max_value - min_value)
    else:
        mapped_pixels = np.absolute(pixels) * 0
    return mapped_pixels


def reshape(samples: np.array):
    if len(samples.shape) == 4:
        samples = np.reshape(samples, (samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3], 1))
    elif len(samples.shape) == 3:
        samples = np.reshape(samples, (samples.shape[0], samples.shape[1], samples.shape[2], 1))
    return samples


def heat_maps_samples_view(samples_list, rows, columns):
    plt.figure(figsize=(40, 20))
    for i in range(int(rows * columns)):
        i = int(i)
        plt.subplot(rows, columns, i + 1)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples_list[i])
    plt.show()


class SampleHandler:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.set_height_width()

    def set_height_width(self):
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = 0, 0
        for file_name in sorted_files(self.f2_data):
            file_path = get_file_path(self.f2_data, file_name)
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
        indices = []
        datasets = []
        for index, file_name in enumerate(sorted_files(self.f3_dm)):
            file_path = get_file_path(self.f3_dm, file_name)
            user_samples = self.get_samples(file_path, start_window, end_window)
            if len(user_samples) > 0:
                dataset_id = file_name.replace('.csv', '')
                indices.append(index)
                datasets.append(user_samples)
            del user_samples
        return datasets, indices

    def get_samples(self, file_path: str, start_window: int, end_window: int, add_empty: bool = False):
        samples = []
        with open(file_path) as file:
            intervals = file.readlines()[1:]
            intervals = intervals[start_window:end_window]
            for interval in intervals:
                sample = np.array(interval.split(','), dtype="float64")
                sample = handle_pixels(sample)
                sample = sample.reshape(self.dataset.width, self.dataset.height)
                if sample.max() > sample.min():
                    samples.append(sample)
                elif add_empty:
                    samples.append(sample)
                del sample
        samples = np.array(samples)
        return reshape(samples)

    def samples_as_list(self, start_window: int, end_window: int):
        datasets, indices = self.get_datasets(start_window, end_window)
        samples = []
        for dataset in datasets:
            for sample in dataset:
                samples.append(sample)
                del sample
            del dataset
        del datasets
        return np.array(samples), indices

    def random_dataset(self):
        def get_random():
            total_users = len(sorted_files(self.f3_dm))
            file_name = sorted_files(self.f3_dm)[random.randrange(total_users)]
            file_path = get_file_path(self.f3_dm, file_name)
            single_dataset = self.get_samples(file_path, 0, 1, add_empty=True)
            return single_dataset
        dataset = get_random()
        return dataset
