import sys

import numpy as np
import pandas as pd
import utm

from inner_functions.path import sorted_files, build_path
from inner_types.data import Dataset
from inner_types.path import Path


def get_lines_and_splitter(file_path: str, time_index: int):
    """
    Returns the lines and the char splitter
    :param file_path: path of the file
    :param time_index: index of the timestamp attribute in the dataset file
    :return: lines and splitter
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        splitter = ',' if ',' in lines[0] else ' '
        times = [int(line.split(splitter)[time_index]) for line in lines]
    if times[0] > times[-1]:
        lines = lines[::-1]
    return lines, splitter


def time_unit_dict(epoch_size):
    """
    Uses a dictionary to map the size of epoch time to the expression of one hour
    :param epoch_size: The epoch size (number of chars)
    :return: The expression of one hour
    """
    conversion_factory = {10: 3600, 13: 3600000}
    hour = conversion_factory.get(epoch_size)
    return hour


def min_adjustment(df: pd.DataFrame, min_lat_y: float, min_lon_x: float):
    """
    Adjusts the x and y so their values start from zero, i.e., their min value is zero
    :param df: The dataframe with the load data
    :param min_lat_y: The latitude/y coordinate
    :param min_lon_x: The longitude/x coordinate
    :return: The dataframe with the values adjusted
    """
    df.x -= min_lon_x
    df.y -= min_lat_y
    df.x = df.x.round(decimals=2)
    df.y = df.y.round(decimals=2)
    return df


class CleaningData:

    def __init__(self, dataset: Dataset):
        """
        Cleans the raw data, preparing it to the heatmaps creation
        :param dataset: A Dataset object
        """
        self.dataset = dataset
        self.f1_raw_data = Path.f1_raw_data(dataset.name)
        self.f2_data = Path.f2_data(dataset.name)
        self.header = 'interval,x,y,time\n'
        self.min_lat_y = sys.float_info.max
        self.min_lon_x = sys.float_info.max

    def line_split(self, line: str, splitter: str):
        """
        Splits the line to get the raw data attributes
        :param line: The line to split
        :param splitter: The splitter
        :return: The raw data attributes (lon or x, lat or y, and time)
        """
        split = line.split(splitter)
        lon_x = float(split[self.dataset.attribute_indexes.lon_x])
        lat_y = float(split[self.dataset.attribute_indexes.lat_y])
        time = split[self.dataset.attribute_indexes.time].replace(' ', '').replace('\n', '')
        time = int(time)
        return lon_x, lat_y, time

    def check_last_interval(self, time: int):
        """
        Checks if the dataset last_epoch is not None and the time attribute is smaller than it or if the
        dataset last_epoch is None
        :param time: The time attribute value
        :return: True or False
        """
        return (self.dataset.last_epoch is not None and time <= self.dataset.last_epoch) or \
            self.dataset.last_epoch is None

    def convert_to_utm(self, lat_y, lon_x):
        """
        Converts the lat_lon to y_x coordinates via utm package if the dataset is_lat_lon attribute is True
        :param lat_y: The lat_y attribute
        :param lon_x: The lon_x attribute
        :return: The converted value of lat_y and lon_x
        """
        if self.dataset.is_lat_lon:
            yx = utm.from_latlon(lat_y, lon_x)
            lat_y = yx[0]
            lon_x = yx[1]
        return lat_y, lon_x

    def intervals_by_node(self):
        """
        Cleans the raw data and sets the time interval within which the nodes move
        """
        size_f1 = len(sorted_files(self.f1_raw_data))
        size_f2 = len(sorted_files(self.f2_data))
        if size_f1 != size_f2:
            if self.dataset.time_as_epoch:
                interval_size = self.dataset.hours_per_interval * time_unit_dict(self.dataset.epoch_size)
            else:
                interval_size = self.dataset.hours_per_interval * 3600

            for file_name in sorted_files(self.f1_raw_data):
                file_path = build_path(self.f1_raw_data, file_name)
                lines, splitter = get_lines_and_splitter(file_path, time_index=self.dataset.attribute_indexes.time)

                output_file_path = build_path(self.f2_data, file_name)
                with open(output_file_path, 'a') as file:
                    file.write(self.header)

                    next_interval = self.dataset.first_epoch
                    interval_index = 0

                    for line in lines:
                        lon_x, lat_y, time = self.line_split(line, splitter)

                        if self.dataset.first_epoch <= time and self.check_last_interval(time):

                            while next_interval + interval_size < time:
                                next_interval = next_interval + interval_size
                                interval_index += 1

                            if self.dataset.lat_y_min <= lat_y <= self.dataset.lat_y_max and \
                                    self.dataset.lon_x_min <= lon_x <= self.dataset.lon_x_max:

                                self.set_min_lat_y_lon_x(lat_y, lon_x)
                                lat_y, lon_x = self.convert_to_utm(lat_y, lon_x)
                                file.write('{},{},{},{}\n'.format(interval_index, lon_x, lat_y, time))
            self.lat_y_lon_x_adjustment()

    def set_min_lat_y_lon_x(self, lat_y: float, lon_x: float):
        """
        Sets the min values for min_lat_y and min_lon_x
        :param lat_y: The latitude/y coordinate
        :param lon_x: The longitude/x coordinate
        :return:
        """
        if lat_y < self.min_lat_y:
            self.min_lat_y = lat_y
        if lon_x < self.min_lon_x:
            self.min_lon_x = lon_x

    def lat_y_lon_x_adjustment(self):
        self.min_lat_y, self.min_lon_x = self.convert_to_utm(self.min_lat_y, self.min_lon_x)
        for file_name in sorted_files(self.f2_data):
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path)
            df = min_adjustment(df, self.min_lat_y, self.min_lon_x)
            df.to_csv(file_path, sep=',', index=False)


def logit(normalized_cell_stay_time):
    """
    Runs the Logit Transformation on the normalized stay time in cells
    :param normalized_cell_stay_time: The list of stay time in cells
    :return: The list with transformed values
    """
    epsilon = 1e-15
    arr_clipped = np.clip(normalized_cell_stay_time, epsilon, 1 - epsilon)
    arr_logit = np.log(arr_clipped / (1 - arr_clipped))
    return arr_logit


def normalize_cell_stay_time(cell_stay_time: np.array, time_in_trace: int, apply_logit: bool):
    """
    Normalize the node's stay time in cells by its max time in trace
    :param cell_stay_time: The list of stay time in cells
    :param time_in_trace: Max time of a node in trace
    :param apply_logit: Bool that indicates whether to apply the logit transformation or not
    :return: A string with the transformed stay time in cells to write in file
    """
    time_in_trace = max(1, time_in_trace)
    if apply_logit:
        new_cell_stay_time = logit(cell_stay_time / time_in_trace)
    else:
        new_cell_stay_time = cell_stay_time / time_in_trace
    cells_stay_time_str = ', '.join(['{:.3f}'.format(item) for item in new_cell_stay_time]) + '\n'
    return cells_stay_time_str


class DisplacementMatrix:

    def __init__(self, dataset: Dataset, apply_logit: bool):
        """
        Creates the Displacement Matrix where each row is a heatmap and each row index is an interval
        :param dataset: A Dataset object
        :param apply_logit: Bool that indicates whether to apply the logit transformation or not
        """
        self.dataset = dataset
        self.apply_logit = apply_logit
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.max_interval = self.get_max_interval()
        print(self.dataset.height, self.dataset.width)
        self.columns = np.arange(0, self.dataset.width * self.dataset.height, 1)

    def get_max_interval(self):
        """
        Gets the max interval after check the data of all nodes
        :return: The max interval
        """
        max_interval = 0
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = 0, 0
        for file_name in sorted_files(self.f2_data):
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path)
            if df.interval.max() > max_interval:
                max_interval = df.interval.max()
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
        return max_interval

    def fill_matrix(self, df_interval: pd.DataFrame, output_file_path: str):
        """
        Fills the Displacement Matrix with the node stay time in cells
        :param df_interval: The dataframe with a node mobility information regarding a specific interval
        :param output_file_path: The path to write the Displacement Matrix
        """
        cell_stay_time = np.zeros(len(self.columns))
        time_in_trace = 0

        y_padding = 1 if self.dataset.paddingYX[0] else 0
        x_padding = 1 if self.dataset.paddingYX[1] else 0
        if len(df_interval) > 0:
            min_in_trace = max_in_trace = previous_time = df_interval.time[0] - 1

            for index, row in df_interval.iterrows():
                if row.time > max_in_trace:
                    max_in_trace = row.time

                # cell position calculation
                y_index = int(row.y / self.dataset.resolution[0]) + y_padding
                x_index = int(row.x / self.dataset.resolution[1]) + x_padding
                cell = (x_index * self.dataset.height) + y_index

                # time a node spends in a cell
                delta_time = row.time - previous_time
                new_time = cell_stay_time[cell] + delta_time
                cell_stay_time[cell] = new_time
                previous_time = row.time

            time_in_trace = max_in_trace - min_in_trace
        # appends feature_row at the matrix
        new_row = normalize_cell_stay_time(cell_stay_time, time_in_trace, self.apply_logit)
        output_file = open(output_file_path, 'a')
        output_file.write(new_row)

    def generate(self):
        """
        Creates the Displacement Matrix where each row is a heatmap and each row index is an interval
        """
        size_f2 = len(sorted_files(self.f2_data))
        size_f3 = len(sorted_files(self.f3_dm))
        if size_f2 != size_f3:
            for file_name in sorted_files(self.f2_data):
                file_path = build_path(self.f2_data, file_name)
                output_file_path = build_path(self.f3_dm, file_name)
                df = pd.read_csv(file_path)
                matrix = pd.DataFrame(columns=self.columns)
                matrix.to_csv(output_file_path, index=False)
                for interval in range(0, self.max_interval + 1):
                    df_interval = df[df.interval == interval]
                    copy = df_interval.copy().reset_index()
                    self.fill_matrix(copy, output_file_path)


def pre_processing(dataset: Dataset, apply_logit: bool):
    """
    Runs the CleaningData and DisplacementMatrix scripts for a dataset
    :param dataset: The Dataset object
    :param apply_logit: Bool that indicates whether to apply the logit transformation or not
    """
    data = CleaningData(dataset)
    data.intervals_by_node()

    data = DisplacementMatrix(dataset, apply_logit)
    data.generate()
