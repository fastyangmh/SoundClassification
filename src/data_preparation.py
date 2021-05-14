# import
from os.path import join
from torch.utils.data.dataset import random_split
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import digital_filter, get_transform_from_file
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from src.utils import pad_waveform
from copy import copy
from torchvision.datasets import DatasetFolder
import torch
import random
import warnings
warnings.filterwarnings('ignore')


# class


class MosaicAudio:
    def __init__(self, samples) -> None:
        self.samples = samples

    def get_data(self, index):
        filepath, label = self.samples[index]
        samples = copy(self.samples)
        samples.remove((filepath, label))
        samples = np.array(samples)
        samples = samples[samples[:, 1] == str(label), 0]
        temp_wav = []
        temp_sample_rate = []
        for f in [filepath]+list(samples[random.sample(population=range(len(samples)), k=4)]):
            w, s = self._vad(filepath=f)
            temp_wav.append(w)
            temp_sample_rate.append(s)
        return torch.cat(temp_wav, 1), label, np.mean(temp_sample_rate)

    def _vad(self, filepath):
        wav, sample_rate = torchaudio.load(filepath)
        threshold = torch.abs(wav).mean()
        voiced_index = torch.where(wav > threshold)[1]
        wav = wav[:, voiced_index[0]:voiced_index[-1]][:, :sample_rate]
        if len(wav[0]) < sample_rate:
            wav = pad_waveform(waveform=wav, max_waveform_length=sample_rate)
        return wav, sample_rate


class AudioFolder(DatasetFolder):
    def __init__(self, root: str, project_parameters, stage, transform=None, loader=None):
        super().__init__(root, loader, extensions=('.wav'), transform=transform)
        self.project_parameters = project_parameters
        self.transform = transform
        self.mosaic_audio = MosaicAudio(
            samples=self.samples) if project_parameters.use_mosaic and stage == 'train' else None

    def __getitem__(self, index: int):
        if self.mosaic_audio is not None and random.random() > 0.5:
            data, label, sample_rate = self.mosaic_audio.get_data(index=index)
        else:
            filepath, label = self.samples[index]
            data, sample_rate = torchaudio.load(filepath=filepath)
        assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate. the sample_rate: {}'.format(
            sample_rate)
        if self.project_parameters.filter_type is not None:
            data = digital_filter(waveform=data, filter_type=self.project_parameters.filter_type,
                                  sample_rate=sample_rate, cutoff_freq=self.project_parameters.cutoff_freq)
        if len(data[0]) < self.project_parameters.max_waveform_length:
            data = pad_waveform(
                waveform=data[0], max_waveform_length=self.project_parameters.max_waveform_length)[None]
        else:
            data = data[:, :self.project_parameters.max_waveform_length]
        if self.transform is not None:
            data = self.transform['audio'](data)
            if 'vision' in self.transform:
                data = self.transform['vision'](data)
        return data, label


class SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, root, download: bool, subset, project_parameters, transform=None):
        super().__init__(root, download=download, subset=subset)
        self.project_parameters = project_parameters
        self.transform = transform
        self.class_to_idx = {c: idx for idx, c in enumerate(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
                                                             'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])}

    def __getitem__(self, n: int):
        data, sample_rate, label = super().__getitem__(n)[:3]
        assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate. the sample_rate: {}'.format(
            sample_rate)
        if self.project_parameters.filter_type is not None:
            data = digital_filter(waveform=data, filter_type=self.project_parameters.filter_type,
                                  sample_rate=sample_rate, cutoff_freq=self.project_parameters.cutoff_freq)
        if len(data[0]) < self.project_parameters.max_waveform_length:
            data = pad_waveform(
                waveform=data[0], max_waveform_length=self.project_parameters.max_waveform_length)[None]
        else:
            data = data[:, :self.project_parameters.max_waveform_length]
        if self.transform is not None:
            data = self.transform['audio'](data)
            if 'vision' in self.transform:
                data = self.transform['vision'](data)
        return data, self.class_to_idx[label]


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = AudioFolder(root=join(self.project_parameters.data_path, stage), stage=stage,
                                                  transform=self.transform_dict[stage], project_parameters=self.project_parameters)
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files, len(
                        self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
            if self.project_parameters.max_files is not None:
                assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset['train'].dataset.class_to_idx, self.project_parameters.classes)
            else:
                assert self.dataset['train'].class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset[stage].class_to_idx, self.project_parameters.classes)
        else:
            train_set = SPEECHCOMMANDS(root=self.project_parameters.data_path, download=True, subset='training',
                                       project_parameters=self.project_parameters, transform=self.transform_dict['train'])
            val_set = SPEECHCOMMANDS(root=self.project_parameters.data_path, download=True, subset='validation',
                                     project_parameters=self.project_parameters, transform=self.transform_dict['val'])
            test_set = SPEECHCOMMANDS(root=self.project_parameters.data_path, download=True, subset='testing',
                                      project_parameters=self.project_parameters, transform=self.transform_dict['test'])
            # modify the maximum number of files
            for v in [train_set, val_set, test_set]:
                v._walker = list(np.random.permutation(v._walker))[
                    :self.project_parameters.max_files]
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}
            # get the classes from the train_set
            self.project_parameters.classes = self.dataset['train'].class_to_idx

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()
