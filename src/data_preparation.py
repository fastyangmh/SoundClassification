# import
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import digital_filter, get_transform_from_file
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
from torch.utils.data import DataLoader
from src.utils import pad_waveform
import warnings
warnings.filterwarnings('ignore')

# def


# class


class SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, root, download: bool, subset, project_parameters, transform=None) -> None:
        super().__init__(root, download=download, subset=subset)
        self.project_parameters = project_parameters
        self.transform = transform
        self.class_to_idx = {c: idx for idx, c in enumerate(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
                                                             'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])}

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, n: int):
        data, sample_rate, label = super().__getitem__(n)[:3]
        assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate. the sample_rate: {}'.format(
            sample_rate)
        if self.project_parameters.filter_type is not None:
            data = digital_filter(waveform=data, filter_type=self.project_parameters.filter_type,
                                  sample_rate=sample_rate, cutoff_freq=self.project_parameters.cutoff_freq)
        if len(data[0]) < self.project_parameters.max_waveform_length:
            data = pad_waveform(
                waveform=data[0], max_waveform_length=project_parameters.max_waveform_length)[None]
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
            file_path=project_parameters.transform_config_path)

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            pass
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self) -> DataLoader:
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
