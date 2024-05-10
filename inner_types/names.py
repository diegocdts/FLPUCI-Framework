from enum import Enum


class ExportedFiles(Enum):
    TRAINING_LOSS_CSV = 'Training losses.csv'
    TESTING_LOSS_CSV = 'Testing losses.csv'
    LOSSES_PNG = 'Losses curves.pdf'

    CONTACT_TIME_CSV = 'Contact time.csv'
    MSE_CSV = 'MSE.csv'
    SSIM_CSV = 'SSIM.csv'
    ARI_CSV = 'ARI.csv'

    CONTACT_TIME_PNG = 'Contact time.pdf'
    MSE_PNG = 'MSE.pdf'
    SSIM_PNG = 'SSIM.pdf'
    ARI_PNG = 'ARI.pdf'

    KS_CHOSEN = 'ks_chosen.csv'

    CORRELATION = 'Samples Correlation.png'
    MCORRELATION = 'Correlation Matrix.png'
