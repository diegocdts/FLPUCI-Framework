from components.community_identification import CommunityIdentification
from instances.arguments import arguments

dataset, approach, properties, parameters, strategy, first_interval, last_interval = arguments()

framework = CommunityIdentification(dataset,
                                    approach,
                                    properties,
                                    parameters,
                                    strategy)
framework.model_training(first_interval, last_interval)
framework.compare_window_strategies()
