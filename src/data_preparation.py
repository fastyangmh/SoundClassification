# import
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import get_transform_from_file
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np

# class


class SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, root, download: bool, subset, transform=None) -> None:
        super().__init__(root, download=download, subset=subset)
        self.transform = transform
        self.class_to_idx = {c: idx for idx, c in enumerate(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
                                                             'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])}

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, n: int):
        data, sample_rate, label = super().__getitem__(n)[:3]
        if self.transform is not None:
            data = self.transform(data)
        return data, label


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
            train_set = SPEECHCOMMANDS(root=self.project_parameters.data_path,
                                       download=True, subset='training', transform=self.transform_dict['train'])
            val_set = SPEECHCOMMANDS(root=self.project_parameters.data_path, download=True,
                                     subset='validation', transform=self.transform_dict['val'])
            test_set = SPEECHCOMMANDS(root=self.project_parameters.data_path,
                                      download=True, subset='testing', transform=self.transform_dict['test'])
            # modify the maximum number of files
            for v in [train_set, val_set, test_set]:
                v._walker = list(np.random.permutation(v._walker))[
                    :self.project_parameters.max_files]
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}
            # get the classes from the train_set
            self.project_parameters.classes = self.dataset['train'].class_to_idx


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()
