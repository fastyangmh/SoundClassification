# import
from ruamel.yaml import safe_load
import torchaudio
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchaudio.functional import lowpass_biquad, highpass_biquad

# def


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


def load_yaml(file_path):
    with open(file=file_path, mode='r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(file_path):
    transform_dict = {}
    transform_config = load_yaml(file_path=file_path)
    for stage in transform_config.keys():
        transform_dict[stage] = {}
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
                transform_dict[stage][transform_type] = nn.Sequential(*temp)
            elif transform_type == 'vision':
                transform_dict[stage][transform_type] = torchvision.transforms.Compose(
                    temp)
    return transform_dict
