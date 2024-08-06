from inner_types.learning import FCAEProperties, TrainingParameters

sfc_rt_properties = FCAEProperties(input_shape=(40, 40, 1),
                                   encode_layers=[128, 64, 32],
                                   encode_activation='relu',
                                   decode_activation='relu',
                                   kernel_size=(3, 3),
                                   encode_strides=[2, 2, 2],
                                   padding='same',
                                   latent_space=100,
                                   learning_rate=0.0005)

ngsim_properties = FCAEProperties(input_shape=(4, 56, 1),
                                  encode_layers=[16, 8],
                                  encode_activation='relu',
                                  decode_activation='relu',
                                  kernel_size=(5, 5),
                                  encode_strides=[2, 2],
                                  padding='same',
                                  latent_space=20,
                                  learning_rate=0.002)

sumo_ipanema_properties = FCAEProperties(input_shape=(48, 24, 1),
                                         encode_layers=[128, 64, 32],
                                         encode_activation='relu',
                                         decode_activation='relu',
                                         kernel_size=(3, 3),
                                         encode_strides=[2, 2, 2],
                                         padding='same',
                                         latent_space=100,
                                         learning_rate=0.0005)

fed_parameters = TrainingParameters(epochs=3, batch_size=2, shuffle_buffer=10, prefetch_buffer=-1, rounds=20)
cen_parameters = TrainingParameters(epochs=150, batch_size=5)
