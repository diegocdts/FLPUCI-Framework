import asyncio
import collections
import os

import tensorflow as tf
import tensorflow_federated as tff

from components.deep_learning_model import model_build, trained_encoder
from components.sample_generation import SampleHandler, compare_samples_reconstructions
from inner_functions.path import build_path, start_end_window_dir, mkdir
from inner_types.data import Dataset
from inner_types.learning import TrainingParameters, FCAEProperties, LearningApproach
from inner_types.path import Path
from utils.losses import LossesHandler


class FederatedDataHandler:

    def __init__(self, dataset: Dataset, parameters: TrainingParameters):
        self.sample_handler = SampleHandler(dataset=dataset)
        self.parameters = parameters
        self.element_spec = self.element_spec_build()

    def preprocess(self, dataset):

        batch_size = self.parameters.batch_size
        if len(dataset) < batch_size:
            batch_size = len(dataset)

        def batch_format_fn(element):
            return collections.OrderedDict(x=element, y=element)

        return dataset.repeat(self.parameters.epochs).shuffle(self.parameters.shuffle_buffer).batch(
            batch_size).map(batch_format_fn).prefetch(self.parameters.prefetch_buffer)

    def element_spec_build(self):
        single_user_dataset = tf.data.Dataset.from_tensor_slices(self.sample_handler.random_dataset())
        preprocessed = self.preprocess(single_user_dataset)
        del single_user_dataset
        return preprocessed.element_spec

    def users_data(self, start_window: int, end_window: int):
        users_dataset_samples, user_indexes, user_names = self.sample_handler.get_datasets(start_window, end_window)
        federated_dataset_samples = []

        for dataset in users_dataset_samples:
            if len(dataset) > 0:
                federated_dataset = tf.data.Dataset.from_tensor_slices(dataset)
                preprocessed = self.preprocess(federated_dataset)
                federated_dataset_samples.append(preprocessed)
        del users_dataset_samples
        return federated_dataset_samples


class FederatedFullConvolutionalAutoEncoder:

    def __init__(self, dataset: Dataset, parameters: TrainingParameters, properties: FCAEProperties):
        self.proximal_term = dataset.proximal_term
        self.properties = properties
        self.federated_data_handler = FederatedDataHandler(dataset, parameters)
        self.iterative_process, self.state = self.global_model_start()
        self.evaluator = self.build_evaluator()
        self.dataset_name = self.federated_data_handler.sample_handler.dataset.name
        self.f8_checkpoint = Path.f8_checkpoints(self.dataset_name, LearningApproach.FED, dataset.proximal_term)
        self.state_manager = None

    def model_fn(self):
        keras_model = model_build(self.properties)
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=self.federated_data_handler.element_spec,
            loss=tf.keras.losses.MeanSquaredError()
        )

    def global_model_start(self):
        iterative_process = tff.learning.algorithms.build_weighted_fed_prox(
            model_fn=self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=self.properties.learning_rate),
            server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=self.properties.learning_rate),
            proximal_strength=self.proximal_term
        )
        # print(str(iterative_process.initialize.type_signature))
        return iterative_process, iterative_process.initialize()

    def build_evaluator(self):
        return tff.learning.build_federated_evaluation(self.model_fn)

    def model_evaluation(self, testing_data):
        return self.evaluator(self.state.global_model_weights, testing_data)

    def init_state_manager(self, path, rounds):
        _ = mkdir(path)
        self.state_manager = tff.program.FileProgramStateManager(root_dir=path, prefix='round_', keep_total=rounds,
                                                                 keep_first=True)

    def get_next_round(self, loop):
        last_state, last_round = loop.run_until_complete(self.state_manager.load_latest(self.state))
        if last_state is not None:
            self.state = last_state
            return last_round + 1
        else:
            return 0

    def training(self, start_window: int, end_window: int):
        loop = asyncio.get_event_loop()
        rounds = self.federated_data_handler.parameters.rounds
        path = build_path(self.f8_checkpoint, start_end_window_dir(start_window, end_window))
        self.init_state_manager(path, rounds)
        next_round = self.get_next_round(loop)

        loss_handler = LossesHandler(path, LearningApproach.FED)
        loss_handler.load(next_round)

        if next_round < rounds:

            training_data = self.federated_data_handler.users_data(start_window, end_window)
            testing_data = self.federated_data_handler.users_data(end_window, end_window + 1)

            for round_num in range(next_round, rounds):
                print('start: {} | end: {} | round: {}'.format(start_window, end_window, round_num))
                round_iteration = self.iterative_process.next(self.state, training_data)
                self.state = round_iteration[0]
                loop.run_until_complete(self.state_manager.save(self.state, round_num))

                loss_handler.append(round_iteration[1], self.model_evaluation(testing_data))
                loss_handler.save_losses()

            del training_data, testing_data, loss_handler

    def encoder_prediction(self, start_window: int, end_window: int):
        samples, user_indexes, user_names = self.federated_data_handler.sample_handler.samples_as_list(start_window,
                                                                                                       end_window)
        keras_model = model_build(self.properties)
        self.state.global_model_weights.assign_weights_to(keras_model)
        encoder = trained_encoder(keras_model)
        predictions = encoder.predict(samples)
        # reconstructions = keras_model.predict(samples)
        # compare_samples_reconstructions(samples, reconstructions)
        del samples, encoder
        return predictions, user_indexes, user_names
