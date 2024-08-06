from enum import Enum


class LearningApproach(Enum):
    FED = 'FL-based'
    CEN = 'Centralized'

    def __str__(self):
        return str(self.value)


class WindowStrategyType(Enum):
    ACC = 'ACC'
    SLI = 'SLI'

    def __str__(self):
        return str(self.value)


class WindowStrategy:

    def __init__(self,
                 strategy_type: WindowStrategyType,
                 window_size: int = None):
        self.type = strategy_type
        self.size = window_size

    def set_window_size(self, window_size):
        self.size = window_size

    def get_start_window(self, end_window: int):
        if self.type == WindowStrategyType.ACC:
            start_window = 0
        else:
            if end_window - self.size >= 0:
                start_window = end_window - self.size
            else:
                start_window = 0
        return start_window


class FCAEProperties:

    def __init__(self,
                 input_shape: tuple,
                 encode_layers: list,
                 encode_activation: str,
                 decode_activation: str,
                 kernel_size: tuple,
                 encode_strides: list,
                 padding: str,
                 latent_space: int,
                 learning_rate: float):
        self.input_shape = input_shape
        self.encode_layers = encode_layers
        self.decode_layers = encode_layers[::-1]
        self.encode_activation = encode_activation
        self.decode_activation = decode_activation
        self.kernel_size = kernel_size
        self.encode_strides = encode_strides
        self.decode_strides = encode_strides[::-1]
        self.padding = padding
        self.latent_space = latent_space
        self.learning_rate = learning_rate


class TrainingParameters:

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 shuffle_buffer: int = None,
                 prefetch_buffer: int = None,
                 rounds: int = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.rounds = rounds
