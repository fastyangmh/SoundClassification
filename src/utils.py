# import
from ruamel.yaml import safe_load
import torchaudio
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchaudio.functional import lowpass_biquad, highpass_biquad
import torch
from os.path import isfile, join
from glob import glob
import random

# def


def get_files(filepath, file_type):
    files = []
    if type(file_type) != list:
        file_type = [file_type]
    for v in file_type:
        files += sorted(glob(join(filepath, '*.{}'.format(v))))
    return files


def calculate_data_weight(classes, data_path):
    data_weight = {}
    for c in classes.keys():
        files = get_files(filepath=join(
            data_path, 'train/{}'.format(c)), file_type=['wav'])
        data_weight[c] = len(files)
    data_weight = {c: 1-(data_weight[c]/sum(data_weight.values()))
                   for c in data_weight.keys()}
    return data_weight


def load_checkpoint(model, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    if model.loss_function.weight is None:
        # delete the loss_function.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['loss_function.weight']
    else:
        # assign the new loss_function weight to the checkpoint
        checkpoint['state_dict']['loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model


def digital_filter(waveform, filter_type, sample_rate, cutoff_freq):
    if filter_type == 'bandpass':
        waveform = lowpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=max(cutoff_freq), Q=1)
        waveform = highpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=min(cutoff_freq), Q=1)
    elif filter_type == 'lowpass':
        waveform = lowpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=max(cutoff_freq), Q=1)
    elif filter_type == 'highpass':
        waveform = highpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=min(cutoff_freq), Q=1)
    return waveform


def pad_waveform(waveform, max_waveform_length):
    diff = max_waveform_length-len(waveform)
    pad = (int(np.ceil(diff/2)), int(np.floor(diff/2)))
    waveform = F.pad(input=waveform, pad=pad)
    return waveform


def load_yaml(filepath):
    with open(file=filepath, mode='r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = {}
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
            for transform_type in transform_config[stage].keys():
                temp = []
                for name, value in transform_config[stage][transform_type].items():
                    if transform_type not in ['audio', 'vision']:
                        assert False, 'please check the transform config.'
                    module_name = 'torchaudio.transforms' if transform_type == 'audio' else 'torchvision.transforms'
                    if value is None:
                        temp.append(eval('{}.{}()'.format(module_name, name)))
                    else:
                        if type(value) is dict:
                            value = ('{},'*len(value)).format(*
                                                              ['{}={}'.format(a, b) for a, b in value.items()])
                        temp.append(
                            eval('{}.{}({})'.format(module_name, name, value)))
                if transform_type == 'audio':
                    transform_dict[stage][transform_type] = nn.Sequential(
                        *temp)
                elif transform_type == 'vision':
                    transform_dict[stage][transform_type] = torchvision.transforms.Compose(
                        temp)
        return transform_dict
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)


def get_sox_effect_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        effect_dict = {}
        effect_config = load_yaml(filepath=filepath)
        for stage in effect_config.keys():
            effect_dict[stage] = []
            if type(effect_config[stage]) != dict:
                effect_dict[stage] = None
                continue
            for effect_type, values in effect_config[stage].items():
                effect_dict[stage].append([effect_type, '{}'.format(
                    random.uniform(min(values), max(values)))])
        return effect_dict
    else:
        assert False, 'please check the sox effect config path: {}'.format(
            filepath)
