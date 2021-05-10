# import
from glob import glob
from os import makedirs
from os.path import join
from src.train import train
import numpy as np
from src.utils import get_files
from src.project_parameters import ProjectParameters
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from shutil import copy2, copytree, rmtree

# def


def _train_val_dataset_from_data_path(project_parameters):
    data, label = [], []
    for stage in ['train', 'val']:
        for c in project_parameters.classes.keys():
            files = get_files(file_path=join(
                project_parameters.data_path, '{}/{}'.format(stage, c)), file_type=['wav'])
            data += sorted(files)
            label += [project_parameters.classes[c]]*len(files)
    return {'data': np.array(data), 'label': np.array(label)}


def _copy_files_to_destination_path(files, destination_path, project_parameters):
    for c in project_parameters.classes.keys():
        makedirs(name=join(destination_path, c), exist_ok=True)
        for f in files:
            if c in f:
                copy2(src=f, dst=join(destination_path, c))


def _create_k_fold_data(project_parameters, dataset):
    skf = StratifiedKFold(n_splits=project_parameters.n_splits, shuffle=True)
    for idx, (train_index, val_index) in tqdm(enumerate(skf.split(X=dataset['data'], y=dataset['label'])), total=project_parameters.n_splits):
        x_train = dataset['data'][train_index]
        x_val = dataset['data'][val_index]
        destination_path = join(
            project_parameters.k_fold_data_path, 'k_fold_{}'.format(idx+1))
        makedirs(name=join(destination_path, 'train'), exist_ok=True)
        makedirs(name=join(destination_path, 'val'), exist_ok=True)
        _copy_files_to_destination_path(files=x_train, destination_path=join(
            destination_path, 'train'), project_parameters=project_parameters)
        _copy_files_to_destination_path(files=x_val, destination_path=join(
            destination_path, 'val'), project_parameters=project_parameters)
        copytree(src=join(project_parameters.data_path, 'test'),
                 dst=join(destination_path, 'test'))


def _get_k_fold_result(project_parameters):
    print('start k-fold cross-validation')
    results = {}
    directories = sorted(glob(join(project_parameters.k_fold_data_path, '*/')))
    for idx, directory in enumerate(directories):
        print('-'*30)
        print('\nk-fold cross-validation: {}/{}'.format(idx +
                                                        1, project_parameters.n_splits))
        project_parameters.data_path = directory
        results[idx+1] = train(project_parameters=project_parameters)
    return results


def _parse_k_fold_result(results):
    train_loss, val_loss, test_loss = [], [], []
    train_acc, val_acc, test_acc = [], [], []
    for result in results.values():
        each_stage_result = {stage: list(result[stage][0].values()) for stage in [
            'train', 'val', 'test']}
        train_loss.append(each_stage_result['train'][0])
        train_acc.append(each_stage_result['train'][1])
        val_loss.append(each_stage_result['val'][0])
        val_acc.append(each_stage_result['val'][1])
        test_loss.append(each_stage_result['test'][0])
        test_acc.append(each_stage_result['test'][1])
    return {'train': (train_loss, train_acc),
            'val': (val_loss, val_acc),
            'test': (test_loss, test_acc)}


def _calculate_mean_and_error(arrays):
    return np.mean(arrays), (max(arrays)-min(arrays)/2)


def evaluate(project_parameters):
    train_val_dataset = _train_val_dataset_from_data_path(
        project_parameters=project_parameters)
    _create_k_fold_data(project_parameters=project_parameters,
                        dataset=train_val_dataset)
    results = _get_k_fold_result(project_parameters=project_parameters)
    results = _parse_k_fold_result(results=results)
    print('-'*30)
    print('k-fold cross-validation training loss mean:\t{} ± {}'.format(*
                                                                        _calculate_mean_and_error(arrays=results['train'][0])))
    print('k-fold cross-validation training accuracy mean:\t{} ± {}'.format(*
                                                                            _calculate_mean_and_error(arrays=results['train'][1])))
    print('k-fold cross-validation validation accuracy mean:\t{} ± {}'.format(*
                                                                              _calculate_mean_and_error(arrays=results['val'][1])))
    print('k-fold cross-validation test accuracy mean:\t{} ± {}'.format(*
                                                                        _calculate_mean_and_error(arrays=results['test'][1])))
    rmtree(path=project_parameters.k_fold_data_path)
    return results


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # k-fold cross validation
    if project_parameters.predefined_dataset is not None:
        print('temporarily does not support predefined dataset.')
    else:
        result = evaluate(project_parameters=project_parameters)
