import os


def sorted_files(dir_path: str):
    return sorted(os.listdir(dir_path))


def get_file_path(dir_path: str, file_name: str):
    return os.path.join(dir_path, file_name)
