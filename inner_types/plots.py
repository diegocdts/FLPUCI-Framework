from enum import Enum


class FigSize(Enum):
    DEFAULT = (5.5, 6.5)
    WIDER = (8.5, 5)


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


class FontSize(Enum):
    DEFAULT = 13


class Legend(Enum):
    BEST_LOCATION = 0
    N_COLUMNS_2 = 2
    N_COLUMNS_3 = 3
