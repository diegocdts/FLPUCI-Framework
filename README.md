# FLPUCI-Framework

This repository includes the scripts of the Federated Learning for Periodic User Community Identification project.

To execute the project, one first needs to pre-process one of the datasets available at the 
[FLPUCI-Datasets repository](https://github.com/diegocdts/FLPUCI-Datasets).

To do so, one can run the [pre_processing_script.py](pre_processing_script.py) and pass the argument (--dataset) 
referring to the identifier of the chosen dataset.

After that, one can start identifying communities by running the [experiment_script.py](experiment_script.py).
The complete list of arguments that may be passed to this script is available at 
[instances/arguments.py](instances/arguments.py)