import os

import pandas as pd

from inner_functions.path import build_path
from inner_types.path import root_dir


def fix_contents():
    def find_lfk_folder(root_directory):
        for dirpath, dirnames, filenames in os.walk(root_directory):
            if 'labels for' in dirpath:
                rewrite_file(dirpath)

    for directory in os.listdir(root_dir):
        find_lfk_folder(build_path(root_dir, directory))


def rewrite_file(dirpath):
    for file in os.listdir(dirpath):
        file_path = build_path(dirpath, file)
        df = pd.read_csv(file_path, names=['label', 'node'], delimiter=' ')
        node_column = df['node'].str.replace('.txt', '', regex=False)
        node_column = node_column.str.replace('.csv', '', regex=False).str[1:]
        df['node'] = node_column

        df.to_csv(file_path, sep=' ', header=False, index=False)
