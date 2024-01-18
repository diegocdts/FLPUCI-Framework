from components.deep_learning_model import FullConvolutionalAutoEncoder
from components.federated_learning_model import FederatedFullConvolutionalAutoEncoder
from components.validation import Validation
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, FCAEProperties, TrainingParameters, WindowStrategy


def adjust_first_interval(first_interval: int):
    if first_interval <= 0:
        first_interval = 1
    return first_interval


class CommunityIdentification:

    def __init__(self, dataset: Dataset,
                 approach: LearningApproach,
                 properties: FCAEProperties,
                 parameters: TrainingParameters,
                 strategy: WindowStrategy,
                 best_metric: bool):
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

            self.model.training(start_window, end_window)

            encodings, user_indexes = self.model.encoder_prediction(start_window=end_window - 1,
                                                                    end_window=end_window)

            self.validation.generate_communities(interval=end_window - 1,
                                                 input_data=encodings,
                                                 user_indexes=user_indexes)

    def compare_window_strategies(self):
        self.validation.compare_strategies()

    def time_evolution(self, choice_index):
        self.validation.time_evolution(choice_index)
