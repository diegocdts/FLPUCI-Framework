from components.community_identification import CommunityIdentification
from instances.arguments import arguments

dataset, approach, properties, parameters, strategy, first_interval, last_interval, best_metric = arguments()

framework = CommunityIdentification(dataset,
                                    approach,
                                    properties,
                                    parameters,
                                    strategy,
                                    best_metric)
framework.model_training(first_interval, last_interval)
framework.compare_window_strategies()
framework.time_evolution()
