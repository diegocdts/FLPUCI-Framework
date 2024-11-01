from components.deep_learning_model import FullConvolutionalAutoEncoder
from components.federated_learning_model import FederatedFullConvolutionalAutoEncoder
from components.validation import Validation
from inner_functions.path import interval_dir, build_path, path_exists
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, FCAEProperties, TrainingParameters, WindowStrategy
from inner_types.names import ExportedFiles
from inner_types.plots import AxisLabel
from utils.plots import plot_existent_result


def adjust_first_interval(first_interval: int):
    if first_interval <= 0:
        first_interval = 1
    return first_interval


def results_exist(results_path, k_candidates):
    names = [ExportedFiles.CONTACT_TIME_CSV, ExportedFiles.MSE_CSV, ExportedFiles.SSIM_CSV, ExportedFiles.ARI_CSV]
    png_names = [ExportedFiles.CONTACT_TIME_PNG, ExportedFiles.MSE_PNG, ExportedFiles.SSIM_PNG, ExportedFiles.ARI_PNG]
    axis_names = [AxisLabel.CONTACT_TIME, AxisLabel.MSE, AxisLabel.SSIM, AxisLabel.ARI]
    for index, name in enumerate(names):
        csv_path = build_path(results_path, name.value)
        if not path_exists(csv_path):
            return False
        plot_existent_result(csv_path, results_path, k_candidates, png_names[index].value, axis_names[index])
    return True


class CommunityIdentification:

    def __init__(self, dataset: Dataset,
                 approach: LearningApproach,
                 properties: FCAEProperties,
                 parameters: TrainingParameters,
                 strategy: WindowStrategy,
                 best_metric: str):
        self.parameters = parameters
        self.strategy = strategy
        self.validation = Validation(dataset, approach, strategy.type, best_metric)

        if approach == LearningApproach.CEN:
            self.model = FullConvolutionalAutoEncoder(dataset, parameters, properties)
        else:
            self.model = FederatedFullConvolutionalAutoEncoder(dataset, parameters, properties)

    def model_training(self, first_interval: int, last_interval: int):
        first_interval = adjust_first_interval(first_interval)

        for end_window in range(first_interval, last_interval + 1):
            start_window = self.strategy.get_start_window(end_window)

            results_path = build_path(self.validation.f9_results, interval_dir(interval=end_window - 1))

            if results_exist(results_path, self.validation.dataset.k_candidates):
                print(f'Results already exist for interval {end_window - 1}')
                continue

            self.model.training(start_window, end_window)

            encodings, user_indexes, user_names = self.model.encoder_prediction(start_window=end_window - 1,
                                                                                end_window=end_window)

            self.validation.generate_communities(interval=end_window - 1,
                                                 input_data=encodings,
                                                 user_indexes=user_indexes,
                                                 user_names=user_names)

    def compare_window_strategies(self):
        self.validation.compare_strategies()

    def time_evolution(self, choice_index):
        self.validation.time_evolution(choice_index)
        self.validation.time_evolution_2(choice_index, is_intra=True)
        self.validation.time_evolution_2(choice_index, is_intra=False)
