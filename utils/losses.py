import numpy as np

from inner_functions.path import get_file_path, path_exists
from inner_types.learning import LearningApproach
from inner_types.names import ExportedFilesName


class LossesHandler:

    def __init__(self, path, approach: LearningApproach):
        self.training_path = get_file_path(path, ExportedFilesName.TRAINING_LOSS.value)
        self.testing_path = get_file_path(path, ExportedFilesName.TESTING_LOSS.value)
        self.plot_path = get_file_path(path, ExportedFilesName.LOSSES_CURVE.value)
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

    def save(self):
        self.training_losses.tofile(self.training_path, sep=',')
        self.testing_losses.tofile(self.testing_path, sep=',')
