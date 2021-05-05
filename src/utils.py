# import
from ruamel.yaml import safe_load
from torchaudio import transforms
import torch.nn as nn

# def


def load_yaml(file_path):
    with open(file=file_path, mode='r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(file_path):
    transform_dict = {}
    transform_config = load_yaml(file_path=file_path)
    for stage in transform_config.keys():
        transform_dict[stage] = []
        for name, value in transform_config[stage].items():
            if value is None:
                transform_dict[stage].append(
                    eval('transforms.{}()'.format(name)))
            else:
                if type(value) is dict:
                    value = ('{},'*len(value)).format(*
                                                      ['{}={}'.format(a, b) for a, b in value.items()])
                transform_dict[stage].append(
                    eval('transforms.{}({})'.format(name, value)))
        transform_dict[stage] = nn.Sequential(*transform_dict[stage])
    return transform_dict
