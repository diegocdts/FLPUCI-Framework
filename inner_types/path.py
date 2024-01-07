import os

from inner_types.learning import LearningApproach, WindowStrategy

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
    def f8_results(dataset: str, approach: LearningApproach, strategy: WindowStrategy):
        return dir_exists_create('{}/f8_results/{}/{}/'.format(dataset, approach.value, strategy))

    @staticmethod
    def f8_results_match(dataset: str, approach: LearningApproach):
        return dir_exists_create('{}/f8_results/{}/strategies_match/'.format(dataset, approach.value))

    @staticmethod
    def f9_checkpoints(dataset: str, approach: LearningApproach):
        return dir_exists_create('{}/f9_checkpoints/{}/'.format(dataset, approach.value))
