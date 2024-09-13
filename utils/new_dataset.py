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
        self.root_dataset = Path.root_dataset(dataset.name)
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

        for index, file_name in enumerate(sorted_files(self.f2_data)):
            min_time_line = pd.DataFrame(columns=['time', 'id', 'x', 'y'], data=[[0.0, index, initial_x, initial_y]])
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
        df_final = pd.concat([df_final, aux_initial[['time', 'id', 'x', 'y']]])

        df_final.loc[df_final['x'] < 0, 'x'] = 0
        df_final.loc[df_final['y'] < 0, 'y'] = 0
        df_final['id'] = df_final['id'].astype(int)

        df_final = df_final.sort_values(by=['id', 'time'])
        df_final = df_final.groupby('id', group_keys=False).apply(filter_blocked_nodes)
        
        df_final = insert_final_position(df_final)

        df_final = df_final.sort_values(by=['time', 'id'], ascending=[True, True]).reset_index(drop=True)

        with open(build_path(self.root_dataset, 'trace.txt'), 'w') as file:
            file.write(f'{df_final["time"].min()} {df_final["time"].max()} '
                       f'{df_final["x"].min()} {df_final["x"].max()} '
                       f'{df_final["y"].min()} {df_final["y"].max()} '
                       f'0.0 0.0\n')

            df_final.to_csv(file, sep=' ', index=False, header=False)

inter_record_threshold = 180

def insert_final_position(df):
    final_x = 100
    final_y = 12000.0
    final_increment = 400
    final_in_line = 0

    new_rows = []
    for i in range(len(df) - 1):
        new_rows.append(df.iloc[i])

        time_n = df.iloc[i]['time']
        time_n1 = df.iloc[i + 1]['time']

        id_n = df.iloc[i]['id']
        id_n1 = df.iloc[i + 1]['id']

        if time_n1 - time_n > inter_record_threshold and id_n == id_n1:
            new_row = pd.Series({'time': time_n + 2, 'id': id_n, 'x': final_x, 'y': final_y})

            new_rows.append(new_row)
            if final_in_line < 60:
                final_x += final_increment
                final_in_line += 1
            else:
                final_y += final_increment
                final_x = 100
                final_in_line = 0
    new_rows.append(df.iloc[-1])

    return pd.DataFrame(new_rows)

def filter_blocked_nodes(group):
    delta_xy = 300
    indexes_to_remove = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if group['time'].iloc[j] - group['time'].iloc[i] > inter_record_threshold:
                break
            if (abs(group['x'].iloc[j] - group['x'].iloc[i]) < delta_xy
                    or abs(group['y'].iloc[j] - group['y'].iloc[i]) < delta_xy):
                indexes_to_remove.append(group.index[j])
    return group.drop(indexes_to_remove)
