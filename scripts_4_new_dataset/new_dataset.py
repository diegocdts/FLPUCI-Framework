import pandas as pd

from inner_functions.path import build_path
from inner_types.path import Path


class NewDataset:

    def __init__(self, name, raw_file_path):
        self.name = name
        self.raw_file_path = raw_file_path
        self.f1_raw_data = Path.f1_raw_data(name)

    def export_raw_data(self):
        df = pd.read_csv(self.raw_file_path, names=['x', 'y', 'id', 'time'])

        df['time'] = df['time'].round(2)
        ids = df['id'].unique()

        for node_id in ids:
            node_trace = df[df.id == node_id]
            node_trace.to_csv(build_path(self.f1_raw_data, f'{node_trace.id.min()}.txt'), sep=' ', header=False,
                              index=False)
