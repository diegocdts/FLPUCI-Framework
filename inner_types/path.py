import os

from inner_types.learning import LearningApproach, WindowStrategyType

root_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'FLPUCI-Datasets/')


def dir_exists_create(dir_name: str):
    """
    Checks if a directory inside the root_dir path exists; if not, then creates it
    :param dir_name: The directory name
    :return: The directory path
    """
    path = os.path.join(root_dir, '{}'.format(dir_name))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def dir_proximal(proximal):
    if proximal == 0.0:
        return 'FED_AVG'
    else:
        return f'FED_PROX_{proximal}'


def labels_for_k():
    return 'labels for k'


class Path:
    @staticmethod
    def f1_raw_data(dataset: str):
        return dir_exists_create('{}/f1_raw_data'.format(dataset))

    @staticmethod
    def f2_data(dataset: str):
        return dir_exists_create('{}/f2_data/'.format(dataset))

    @staticmethod
    def f3_dm(dataset: str):
        return dir_exists_create('{}/f3_logit/'.format(dataset))

    @staticmethod
    def f4_entry_exit(dataset: str):
        return dir_exists_create('{}/f4_entry_exit/'.format(dataset))

    @staticmethod
    def f5_interval_entry_exit(dataset: str):
        return dir_exists_create('{}/f5_interval_entry_exit/'.format(dataset))

    @staticmethod
    def f6_contact_time(dataset: str):
        return dir_exists_create('{}/f6_contact_time/'.format(dataset))

    @staticmethod
    def f7_metrics(dataset: str):
        return dir_exists_create('{}/f7_metrics/'.format(dataset))

    @staticmethod
    def f8_checkpoints(dataset: str, approach: LearningApproach, proximal: float):
        if approach == LearningApproach.CEN:
            return dir_exists_create('{}/f8_checkpoints/{}/'.format(dataset, approach.value))
        else:
            return dir_exists_create('{}/f8_checkpoints/{}/{}/'
                                     .format(dataset, approach.value, dir_proximal(proximal)))

    @staticmethod
    def f9_results(dataset: str, approach: LearningApproach, strategy_type: WindowStrategyType, proximal):
        if approach == LearningApproach.CEN:
            return dir_exists_create('{}/f9_results/{}/{}/'.format(dataset, approach.value, strategy_type))
        else:
            return dir_exists_create('{}/f9_results/{}/{}/{}/'
                                     .format(dataset, approach.value, dir_proximal(proximal), strategy_type))

    @staticmethod
    def f9_results_compare_strategies(dataset: str, approach: LearningApproach, proximal):
        if approach == LearningApproach.CEN:
            return dir_exists_create('{}/f9_results/{}/strategy_comparison/'.format(dataset, approach.value))
        else:
            return dir_exists_create('{}/f9_results/{}/{}/strategy_comparison/'
                                     .format(dataset, approach.value, dir_proximal(proximal)))

    @staticmethod
    def f9_results_time_evolution(dataset: str):
        return dir_exists_create('{}/f9_results/time_evolution/'.format(dataset))

    @staticmethod
    def f9_results_analysis(dataset: str):
        return dir_exists_create('{}/f9_results/analysis/'.format(dataset))

    @staticmethod
    def f9_base(dataset: str):
        return dir_exists_create('{}/f9_results/'.format(dataset))

    @staticmethod
    def f9_community_info(dataset: str, approach: LearningApproach, strategy_type: WindowStrategyType, proximal):
        info = '_community_info'
        if approach == LearningApproach.CEN:
            return dir_exists_create('{}/f9_results/{}/{}{}/'.format(dataset, approach.value, strategy_type, info))
        else:
            return dir_exists_create('{}/f9_results/{}/{}/{}{}/'
                                     .format(dataset, approach.value, dir_proximal(proximal), strategy_type, info))

    @staticmethod
    def f9_community_id_maps(f9_info: str):
        return dir_exists_create('{}/community_id_maps'.format(f9_info))

    @staticmethod
    def f9_previous_community_count(f9_info: str):
        return dir_exists_create('{}/previous_community_count'.format(f9_info))
