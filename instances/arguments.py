import argparse

from inner_types.learning import LearningApproach, WindowStrategy, WindowStrategyType
from instances.data import sfc, rt, ngsim
from instances.learning import cen_parameters, fed_parameters, sfc_rt_properties, ngsim_properties


def arguments():
    parser = argparse.ArgumentParser(description='Arguments to be used in the FLPUCI-Framework')

    parser.add_argument('--dataset',
                        type=str,
                        default='rt',
                        help='The dataset identifier. It may be sfc (San Francisco Cabs),'
                             'rt (Roma Taxi) or ngsim (Next Generation Simulation). Default rt')

    parser.add_argument('--first_interval',
                        type=int,
                        default=0,
                        help='The index of the first interval to run experiments. Default 0.')

    parser.add_argument('--last_interval',
                        type=int,
                        default=5,
                        help='The index of the last interval to run experiments. Default 5')

    parser.add_argument('--approach',
                        type=str,
                        default='cen',
                        help='The learning approach. It may be cen (Centralized) or fed (FL-based). Default cen')

    parser.add_argument('--strategy',
                        type=str,
                        default='sli',
                        help='The window strategy. It may be sli (Sliding) or acc (Accumulated). Default sli')

    parser.add_argument('--sli_size',
                        type=int,
                        default=3,
                        help='The size of the sli window strategy. Default 3')

    parser.add_argument('--best_metric',
                        type=bool,
                        default=True,
                        help='If True, the contact time is used as metric to generate the Best candidate. '
                             'Otherwise, the SSIM will be used. Default True')

    parsed = parser.parse_args()

    properties = sfc_rt_properties
    if parsed.dataset == 'sfc':
        dataset = sfc
    elif parsed.dataset == 'rt':
        dataset = rt
    else:
        dataset = ngsim
        properties = ngsim_properties

    if parsed.approach == 'cen':
        approach = LearningApproach.CEN
        parameters = cen_parameters
    else:
        approach = LearningApproach.FED
        parameters = fed_parameters

    if parsed.strategy == 'sli':
        strategy = WindowStrategy(WindowStrategyType.SLI, window_size=parsed.sli_size)
    else:
        strategy = WindowStrategy(WindowStrategyType.ACC)

    first_interval = parsed.first_interval
    last_interval = parsed.last_interval

    return dataset, approach, properties, parameters, strategy, first_interval, last_interval
