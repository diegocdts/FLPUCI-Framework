from components.baseline_computation import compute_baseline
from components.pre_processing import pre_processing
from instances.arguments import arguments

args = arguments()

pre_processing(args.dataset, args.apply_logit)

compute_baseline(args.dataset)
