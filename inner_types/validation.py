from enum import Enum


class ImageMetric(Enum):
    MSE = 'MSE'
    SSIM = 'SSIM'
    ARI = 'ARI'

    def __str__(self):
        return str(self.value)
