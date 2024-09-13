import pandas as pd

from inner_functions.path import build_path, sorted_files
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


class ExportTrace:

    def __init__(self, dataset):
        self.dataset = dataset
        self.f2_data = Path.f2_data(dataset.name)
        self.mapping = dict()
        self.set_mapping()

    def set_mapping(self):
        for index, name in enumerate(sorted_files(self.f2_data)):
            self.mapping[name] = index

    def rewrite_info(self, path):
        for file_name in sorted_files(path):
            file_path = build_path(path, file_name)
            new_content = ''
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    split = line.split(' ')
                    label = split[0]
                    node_name = split[1][1:].replace('\n', '')
                    new_content = f'{new_content}{label} {self.mapping[node_name]}\n'
            with open(file_path, 'w') as new_file:
                new_file.write(new_content)

    def write_trace(self):
        df_final = pd.DataFrame(columns=['time', 'id', 'x', 'y'])
        aux = pd.DataFrame(columns=['interval', 'x', 'y', 'time'])
        initial_x = 100
        initial_y = 12000
        initial_increment = 400
        initial_in_line = 0

        final_x = 100
        final_y = 12000
        final_increment = 400
        final_in_line = 0
        for index, file_name in enumerate(sorted_files(self.f2_data)):
            # time id xPos yPos
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path, sep=',')

            df['id'] = self.mapping[file_name]
            min_time_line = df[df['time'] == df['time'].min()].copy()
            min_time_line['time'] = 0.0
            min_time_line['x'] = initial_x
            min_time_line['y'] = initial_y
            if initial_in_line < 60:
                initial_x += initial_increment
                initial_in_line += 1
            else:
                initial_y += initial_increment
                initial_x = 100
                initial_in_line = 0
            aux = pd.concat([aux, min_time_line])

            max_time_line = df[df['time'] == df['time'].max()].copy()
            max_time_line['time'] = df['time'].max()
            max_time_line['x'] = final_x
            max_time_line['y'] = final_y
            if final_in_line < 60:
                final_x += final_increment
                final_in_line += 1
            else:
                final_y += final_increment
                final_x = 100
                final_in_line = 0
            aux = pd.concat([aux, max_time_line])

            df_final = pd.concat([df_final, df[['time', 'id', 'x', 'y']]])

        df_final['time'] = df_final['time'] - df_final['time'].min()
        df_final = pd.concat([df_final, aux[['time', 'id', 'x', 'y']]])
        df_final = df_final.sort_values(by=['time', 'id'], ascending=[True, True]).reset_index(drop=True)
        df_final = df_final[['time', 'id', 'x', 'y']]
        df_final.loc[df_final['y'] < 0, 'y'] = 0
        df_final['id'] = df_final['id'].astype(int)

        with open(build_path(self.f2_data, 'TRACE_SFC.txt'), 'w') as file:
            file.write(f'{df_final["time"].min()} {df_final["time"].max()} {df_final["x"].min()} {df_final["x"].max()} '
                       f'{df_final["y"].min()} {df_final["y"].max()} 0.0 0.0\n')

            df_final.to_csv(file, sep=' ', index=False, header=False)
