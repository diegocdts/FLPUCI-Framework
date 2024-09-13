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
        aux_initial = pd.DataFrame(columns=['id', 'x', 'y', 'time'])

        initial_x = 100.0
        initial_y = 12000.0
        initial_increment = 400
        initial_in_line = 0

        final_x = 100
        final_y = 12000.0
        final_increment = 400
        final_in_line = 0

        # organiza todos os nós numa posição inicial
        for index, file_name in enumerate(sorted_files(self.f2_data)):
            min_time_line = pd.DataFrame(columns=['id', 'x', 'y', 'time'], data=[[index, initial_x, initial_y, 0.0]])
            if initial_in_line < 60:
                initial_x += initial_increment
                initial_in_line += 1
            else:
                initial_y += initial_increment
                initial_x = 100
                initial_in_line = 0
            aux_initial = pd.concat([aux_initial, min_time_line])

        for index, file_name in enumerate(sorted_files(self.f2_data)):
            # time id xPos yPos
            file_path = build_path(self.f2_data, file_name)
            df = pd.read_csv(file_path, sep=',')
            df = df.sort_values(by='time', ascending=True).reset_index(drop=True)

            df['id'] = self.mapping[file_name]
            df_final = pd.concat([df_final, df[['time', 'id', 'x', 'y']]])

        df_final['time'] = df_final['time'] - df_final['time'].min()
        df_final = df_final[df_final['time'] <= 86400]
        df_final = df_final.sort_values(by=['time', 'id'], ascending=[True, True]).reset_index(drop=True)

        last_rows = df_final.groupby('id').tail(1)
        for i, row in last_rows.iterrows():
            new_row = pd.DataFrame({'time': [row.time], 'id': [row.id], 'x': [final_x], 'y': [final_y]})
            df_final = pd.concat([df_final, new_row], ignore_index=True)
            if final_in_line < 60:
                final_x += final_increment
                final_in_line += 1
            else:
                final_y += final_increment
                final_x = 100
                final_in_line = 0

        df_final = pd.concat([df_final, aux_initial[['time', 'id', 'x', 'y']]])
        df_final = df_final.sort_values(by=['time', 'id'], ascending=[True, True]).reset_index(drop=True)
        df_final = df_final[['time', 'id', 'x', 'y']]
        df_final.loc[df_final['y'] < 0, 'y'] = 0
        df_final['id'] = df_final['id'].astype(int)

        with open(build_path(self.f2_data, 'TRACE_SFC.txt'), 'w') as file:
            file.write(f'{df_final["time"].min()} {df_final["time"].max()} {df_final["x"].min()} {df_final["x"].max()} '
                       f'{df_final["y"].min()} {df_final["y"].max()} 0.0 0.0\n')

            df_final.to_csv(file, sep=' ', index=False, header=False)
