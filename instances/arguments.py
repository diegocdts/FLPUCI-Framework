import argparse
from collections import namedtuple

from inner_types.learning import LearningApproach, WindowStrategy, WindowStrategyType
from instances.data import sfc, rt, ngsim, helsinki
from instances.learning import cen_parameters, fed_parameters, sfc_rt_properties, helsinki_properties, ngsim_properties


def arguments():
    parser = argparse.ArgumentParser(description='Arguments to be used in the FLPUCI-Framework')

    parser.add_argument('--dataset',
                        type=str,
                        default='sfc',
                        help='The dataset identifier. It may be sfc (San Francisco Cabs),'
                             'rt (Roma Taxi) or ngsim (Next Generation Simulation). Default rt')

    parser.add_argument('--apply_logit',
                        type=str,
                        default='True',
                        help='If True, the logit transformation is applied to the DisplacementMatrix. Default True')

    parser.add_argument('--first_interval',
                        type=int,
                        default=0,
                        help='The index of the first interval to run experiments. Default 0.')

    parser.add_argument('--last_interval',
                        type=int,
                        default=5,
                        help='The total number of intervals for the experiment. Default 5')

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
                        type=str,
                        default='contact_time',
                        help='The metric used to generate the Best candidate. It can be contact_time or ssim.'
                             'Default contact_time')

    parser.add_argument('--choice',
                        type=str,
                        default='best',
                        help='The K choice to plot in the time evolution. It may be aic, bic or best. Default best.')

    parser.add_argument('--proximal_term',
                        type=float,
                        default=0.0,
                        help='The parameter of FedProx\'s regularization term. When set to 0.0, the algorithm reduces '
                             'to FedAvg Default 0.0.')

    parsed = parser.parse_args()

    properties = sfc_rt_properties
    if parsed.dataset == 'sfc':
        dataset = sfc
    elif parsed.dataset == 'rt':
        dataset = rt
    elif parsed.dataset == 'helsinki':
        dataset = helsinki
        properties = helsinki_properties
    else:
        dataset = ngsim
        properties = ngsim_properties
    dataset.proximal_term = parsed.proximal_term

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
    best_metric = parsed.best_metric
    apply_logit = parsed.apply_logit == 'True'

    if parsed.choice == 'aic':
        choice = 0
    elif parsed.choice == 'bic':
        choice = 1
    else:
        choice = 2

    args = namedtuple('args',
                      ['dataset', 'apply_logit', 'approach', 'properties', 'parameters', 'strategy', 'first_interval',
                       'last_interval', 'best_metric', 'choice'])

    return args(dataset, apply_logit, approach, properties, parameters, strategy, first_interval, last_interval,
                best_metric, choice)
