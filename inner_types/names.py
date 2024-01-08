from enum import Enum


class ExportedFilesName(Enum):
    TRAINING_LOSS = 'Training losses.csv'
    TESTING_LOSS = 'Testing losses.csv'
    LOSSES_CURVE = 'Losses curves.png'
