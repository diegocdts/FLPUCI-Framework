from components.baseline_computation import compute_baseline
from components.pre_processing import pre_processing
from instances.data import rt

pre_processing(rt)

compute_baseline(rt)
