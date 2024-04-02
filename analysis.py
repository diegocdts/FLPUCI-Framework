from data_analysis.iid_analysis import get_sample_correlations
from instances.arguments import arguments

args = arguments()

get_sample_correlations(args.dataset, args.last_interval)
