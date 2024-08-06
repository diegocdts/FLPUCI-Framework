import json


def read_json(path):
    with open(path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary
