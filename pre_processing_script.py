from components.baseline_computation import compute_baseline
from components.pre_processing import pre_processing
from instances.arguments import arguments

dataset, approach, properties, parameters, strategy, first_interval, last_interval, best_metric, choice = arguments()

pre_processing(dataset)

compute_baseline(dataset)
