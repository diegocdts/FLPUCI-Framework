
def sources():
    return ['all_pairs', 'intra_community', 'inter_community']


def curves():
    return ['lower', 'mean', 'upper']


def choice_method():
    return ['AIC', 'BIC', 'Best']


def column_k(k: int):
    return f'k={k}'


def sort_name(directory):
    return int(directory.split('_')[1])
