import tensorflow as tf

from components.sample_generation import SampleHandler
from inner_functions.path import get_file_path, path_exists, start_end_window_dir
from inner_types.data import Dataset
from inner_types.learning import LearningApproach, FCAEProperties, TrainingParameters
from inner_types.path import Path
from utils.losses import LossesHandler


class FullConvolutionalAutoEncoder:

    def __init__(self, dataset: Dataset, parameters: TrainingParameters, properties: FCAEProperties):
        self.sample_handler = SampleHandler(dataset)
        self.parameters = parameters
        self.f8_checkpoint = Path.f8_checkpoints(self.sample_handler.dataset.name, LearningApproach.CEN)
        self.model = model_build(properties)
        self.compile(properties)

    def compile(self, properties):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=properties.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())

    @staticmethod
    def checkpoint(path: str):
        return tf.keras.callbacks.ModelCheckpoint(filepath=get_file_path(path, 'cp.ckpt'),
                                                  save_weights_only=True, verbose=1)

    def training(self, start_window: int, end_window: int):
        path = get_file_path(self.f8_checkpoint, start_end_window_dir(start_window, end_window))
        loss_handler = LossesHandler(path, LearningApproach.CEN)

        if path_exists(path):
            loss_handler.load()
            self.model.load_weights(get_file_path(path, 'cp.ckpt'))
        else:
            training_data, training_indices_list = self.sample_handler.samples_as_list(start_window, end_window)
            testing_data, testing_indices_list = self.sample_handler.samples_as_list(end_window, end_window + 1)
            history = self.model.fit(x=training_data,
                                     y=training_data,
                                     batch_size=self.parameters.batch_size,
                                     epochs=self.parameters.epochs,
                                     verbose=1,
                                     callbacks=[self.checkpoint(path)],
                                     validation_data=(testing_data, testing_data))
            loss_handler.append(history.history['loss'], history.history['val_loss'])
            loss_handler.save_losses()
            del training_data, testing_data, loss_handler

    def encoder_prediction(self, start_window: int, end_window: int):
        samples, indices_list = self.sample_handler.samples_as_list(start_window, end_window)
        encoder = trained_encoder(self.model)
        predictions = encoder.predict(samples)
        del samples, encoder
        return predictions, indices_list


def dense_nodes_width_height(properties: FCAEProperties):
    width, height = properties.input_shape[0], properties.input_shape[1]
    for _ in [stride for stride in properties.encode_strides if stride != 1]:
        if width > 1:
            width = width / 2
        if height > 1:
            height = height / 2
    dense_nodes = int(width * height * properties.encode_layers[-1])
    return dense_nodes, int(width), int(height)


def encoder_build(properties: FCAEProperties):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.InputLayer(input_shape=properties.input_shape))

    for index, layer in enumerate(properties.encode_layers):
        encoder.add(tf.keras.layers.Conv2D(filters=layer,
                                           kernel_size=properties.kernel_size,
                                           activation=properties.encode_activation,
                                           strides=properties.encode_strides[index],
                                           padding=properties.padding))
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(units=properties.latent_space,
                                      activation=properties.encode_activation))
    return encoder


def decoder_build(properties: FCAEProperties):
    dense_layer, width, height = dense_nodes_width_height(properties)

    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.InputLayer(input_shape=(properties.latent_space,)))

    decoder.add(tf.keras.layers.Dense(units=dense_layer,
                                      activation=properties.decode_activation))
    decoder.add(tf.keras.layers.Reshape((width, height, properties.decode_layers[0])))

    for index, layer in enumerate(properties.decode_layers):
        decoder.add(tf.keras.layers.Conv2DTranspose(filters=layer,
                                                    kernel_size=properties.kernel_size,
                                                    activation=properties.decode_activation,
                                                    strides=properties.decode_strides[index],
                                                    padding=properties.padding))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=1,
                                                kernel_size=properties.kernel_size,
                                                activation=properties.decode_activation,
                                                padding=properties.padding))
    return decoder


def model_build(properties: FCAEProperties):
    encoder = encoder_build(properties)
    print(encoder.summary())
    decoder = decoder_build(properties)
    print(decoder.summary())
    return tf.keras.models.Model(inputs=encoder.input,
                                 outputs=decoder(encoder.outputs))


def trained_encoder(model: tf.keras.Model):
    return tf.keras.models.Model(model.input, model.layers[-2].output)
