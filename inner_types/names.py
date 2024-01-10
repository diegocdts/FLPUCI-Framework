from enum import Enum


class ExportedFilesName(Enum):
    TRAINING_LOSS_CSV = 'Training losses.csv'
    TESTING_LOSS_CSV = 'Testing losses.csv'
    LOSSES_PNG = 'Losses curves.png'

    CONTACT_TIME_CSV = 'Contact time.csv'
    MSE_CSV = 'MSE.csv'
    SSIM_CSV = 'SSIM.csv'
    ARI_CSV = 'ARI.csv'
