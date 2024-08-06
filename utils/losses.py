import os

import numpy as np

from inner_functions.path import build_path, path_exists
from inner_types.learning import LearningApproach
from inner_types.names import ExportedFiles
from inner_types.path import Path
from utils.plots import plot_losses


class LossesHandler:

    def __init__(self, path, approach: LearningApproach):
        self.training_path = build_path(path, ExportedFiles.TRAINING_LOSS_CSV.value)
        self.testing_path = build_path(path, ExportedFiles.TESTING_LOSS_CSV.value)
        self.plot_path = build_path(path, ExportedFiles.LOSSES_PNG.value)
        self.approach = approach
        self.training_losses = np.array([])
        self.testing_losses = np.array([])

    def get_losses(self):
        return self.training_losses, self.testing_losses

    def append(self, training, testing):
        if self.approach == LearningApproach.CEN:
            self.training_losses = np.array(training)
            self.testing_losses = np.array(testing)
        else:
            training_loss = training['client_work']['train']['loss']
            self.training_losses = np.append(self.training_losses, training_loss)
            testing_loss = testing['eval']['loss']
            self.testing_losses = np.append(self.testing_losses, testing_loss)

    def load(self, trained_rounds: int = 0):
        if self.approach == LearningApproach.CEN:
            if path_exists(self.training_path) and path_exists(self.testing_path):
                self.training_losses = np.fromfile(self.training_path, sep=',')
                self.testing_losses = np.fromfile(self.testing_path, sep=',')
        else:
            if trained_rounds > 0:
                self.training_losses = np.fromfile(self.training_path, sep=',')
                self.testing_losses = np.fromfile(self.testing_path, sep=',')
            return self.training_losses, self.testing_losses

    def save_losses(self):
        self.training_losses.tofile(self.training_path, sep=',')
        self.testing_losses.tofile(self.testing_path, sep=',')
        plot_losses(self.training_losses, self.testing_losses, self.approach, self.plot_path)


def replot_losses(dataset_name, approach, proximal):
    f8_checkpoints = Path.f8_checkpoints(dataset_name, approach, proximal)
    windows = sorted(os.listdir(f8_checkpoints))

    for window in windows:
        path = build_path(f8_checkpoints, window)
        loss_handler = LossesHandler(path, LearningApproach.FED)
        loss_handler.load(1)
        loss_handler.save_losses()
