import numpy as np


class LatYLonXTimeIndexes:

    def __init__(self, lat_y: int, lon_x: int, time: int):
        """
        Specifies the indexes of the raw dataset attribute inside the file lines
        :param lat_y: Index of the latitude or y attribute
        :param lon_x: Index of the longitude or x attribute
        :param time: Index of the time attribute
        """
        self.lat_y = lat_y
        self.lon_x = lon_x
        self.time = time


class Dataset:

    def __init__(self,
                 name: str,
                 hours_per_interval: float,
                 first_epoch: int,
                 lat_y_min: float,
                 lat_y_max: float,
                 lon_x_min: float,
                 lon_x_max: float,
                 resolution: tuple,
                 attribute_indexes: LatYLonXTimeIndexes,
                 last_epoch: int = None,
                 is_lat_lon: bool = True,
                 paddingYX: tuple = (False, False),
                 k_candidates: int = 20,
                 proximal_term: float = 0.0,
                 time_as_epoch: bool = True):
        """
        Defines the attributes of a data set
        :param name: Dataset name
        :param hours_per_interval: The number of hours of each interval
        :param first_epoch: The dataset initial epoch time
        :param lat_y_min: The min latitude or y
        :param lat_y_max: The max latitude or y
        :param lon_x_min: The min longitude or x
        :param lon_x_max: The max longitude or x
        :param resolution: A int tuple indicating the cell resolution i.e., (height, width)
        :param attribute_indexes: A LatYLonXTimeIndexes object
        :param last_epoch: The dataset final epoch time (Optional)
        :param is_lat_lon: A bool value to indicate if the raw data has lat_lon or y_x geo-information (Optional)
        :param paddingYX: A bool tuple indicating if a padding should be put over the y and x dimensions of the final
        :param k_candidates: Maximum number of communities to test
        :param proximal_term: The parameter of FedProx's regularization term
        :param time_as_epoch: True if the timestamp in the dataset is in epoch format
        """
        self.name = name
        self.hours_per_interval = hours_per_interval
        self.first_epoch = first_epoch
        self.last_epoch = last_epoch
        self.lat_y_min = lat_y_min
        self.lat_y_max = lat_y_max
        self.lon_x_min = lon_x_min
        self.lon_x_max = lon_x_max
        self.resolution = resolution
        self.is_lat_lon = is_lat_lon
        self.paddingYX = paddingYX
        self.attribute_indexes = attribute_indexes
        self.epoch_size = len(str(first_epoch))
        self.height = None
        self.width = None
        self.k_candidates = np.arange(2, k_candidates + 1)
        self.proximal_term = proximal_term
        self.time_as_epoch = time_as_epoch

    def set_height_width(self, float_height: float, float_width: float):
        """
        Sets the height and width of the dataset's heatmap
        :param float_height: The float height resulting from the (max_y - min_y) / resolution[0] calculation
        :param float_width: The float width resulting from the (max_x - min_x) / resolution[1] calculation
        """
        if float_height.is_integer:
            float_height = int(float_height) + 1
        if int(float_height) % 2 != 0:
            float_height += 1
        self.height = int(float_height)

        if float_width.is_integer:
            float_width = int(float_width) + 1
        if int(float_width) % 2 != 0:
            float_width += 1
        self.width = int(float_width)

        if self.paddingYX[0]:
            self.height = self.height + 2
        if self.paddingYX[1]:
            self.width = self.width + 2
