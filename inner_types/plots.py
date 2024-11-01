from enum import Enum


class FigSize(Enum):
    DEFAULT = (5.5, 6.5)
    WIDER = (12.2, 6)
    WIDER_2 = (14, 6)
    SQUARE = (7.5, 6.5)
    SQUARE_2 = (9.55, 5)
    SMALL = (7, 4)


class AxisLabel(Enum):
    LOSS = 'LOSS'
    EPOCH = 'EPOCH'
    ROUND = 'ROUND'

    K = 'CANDIDATE K'
    INTERVAL = 'INTERVAL INDEX'

    CONTACT_TIME = 'CONTACT TIME (IN SECONDS)'
    MSE = 'NORMALIZED MSE'
    SSIM = 'NORMALIZED SSIM'
    ARI = 'NORMALIZED ARI'

    MEAN_CORRELATION = 'MEAN CORRELATION'
    CORRELATION = 'CORRELATION'
    PAIR_INDEX = 'PAIR INDEX'


class FontSize(Enum):
    DEFAULT = 14
    LARGE = 16

    WEIGHT_SBOLD = 'semibold'


class Legend(Enum):
    BEST_LOCATION = 0
    UPPER_RIGHT = 1
    N_COLUMNS_1 = 1
    N_COLUMNS_2 = 2
    N_COLUMNS_3 = 3
