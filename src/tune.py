# import
from os.path import join
from src.utils import load_yaml
from src.project_parameters import ProjectParameters
import ray.tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import init, shutdown
from functools import partial
from copy import copy
from src.train import train
import numpy as np

# def


def _get_hyperparameter_space(project_parameters):
    hyperparameter_space_config = load_yaml(
        filepath=project_parameters.hyperparameter_config_path)
    assert hyperparameter_space_config is not None, 'the hyperparameter space config has not any content.'
    hyperparameter_space = {}
    for parameter_type in hyperparameter_space_config.keys():
        assert parameter_type in ['int', 'float', 'choice'], 'the type is wrong, please check it. the type: {}'.format(
            parameter_type)
        for parameter_name, parameter_value in hyperparameter_space_config[parameter_type].items():
            if parameter_name == 'cutoff_freq_low' or parameter_name == 'cutoff_freq_high':
                pass
            else:
                assert parameter_name in project_parameters, 'the parameter name is wrong, please check it. the parameter name: {}'.format(
                    parameter_name)
            if parameter_type == 'int':
                hyperparameter_space[parameter_name] = ray.tune.randint(
                    lower=min(parameter_value), upper=max(parameter_value))
            elif parameter_type == 'float':
                hyperparameter_space[parameter_name] = ray.tune.uniform(
                    lower=min(parameter_value), upper=max(parameter_value))
            elif parameter_type == 'choice':
                hyperparameter_space[parameter_name] = ray.tune.choice(
                    categories=parameter_value)
    return hyperparameter_space


def _set_tune_project_parameters(hyperparameter, project_parameters):
    temp = []
    for k, v in hyperparameter.items():
        if k == 'cutoff_freq_low' or k == 'cutoff_freq_high':
            temp.append(v)
            continue
        if type(v) == str:
            exec('project_parameters.{}="{}"'.format(k, v))
        else:
            exec('project_parameters.{}={}'.format(k, v))
        if k == 'train_iter' and not project_parameters.use_early_stopping:
            project_parameters.val_iter = v
    if sum(temp):
        project_parameters.cutoff_freq = sorted(temp)
    return project_parameters


def _parse_tune_result(result):
    loss_dict, accuracy_dict = {}, {}
    for stage in ['train', 'val', 'test']:
        loss, accuracy = result[stage][0].values()
        loss_dict[stage] = loss
        accuracy_dict[stage] = accuracy
    return loss_dict, accuracy_dict


def _tune_function(hyperparameter, project_parameters):
    if project_parameters.tune_debug:
        ray.tune.report(accuracy_difference=sum(
            [value for value in hyperparameter.values() if type(value) is not str]))
    else:
        tune_project_parameters = _set_tune_project_parameters(
            hyperparameter=hyperparameter, project_parameters=copy(project_parameters))
        result = train(project_parameters=tune_project_parameters)
        loss_dict, accuracy_dict = _parse_tune_result(result=result)
        accuracy_difference = sum([1-v for v in accuracy_dict.values()])
        ray.tune.report(train_loss=loss_dict['train'],
                        val_loss=loss_dict['val'],
                        test_loss=loss_dict['test'],
                        train_accuracy=accuracy_dict['train'],
                        val_accuracy=accuracy_dict['val'],
                        test_accuracy=accuracy_dict['test'],
                        accuracy_difference=accuracy_difference)


def tune(project_parameters):
    project_parameters.mode = 'train'
    hyperparameter_space = _get_hyperparameter_space(
        project_parameters=project_parameters)
    tune_scheduler = ASHAScheduler(metric='accuracy_difference', mode='min')
    reporter = CLIReporter(metric_columns=['train_loss', 'val_loss', 'test_loss',
                                           'train_accuracy', 'val_accuracy', 'test_accuracy', 'accuracy_difference'])
    init(dashboard_host='0.0.0.0')
    tuning_result = ray.tune.run(run_or_experiment=partial(
        _tune_function, project_parameters=project_parameters),
        resources_per_trial={
            'cpu': project_parameters.tune_cpu, 'gpu': project_parameters.tune_gpu},
        config=hyperparameter_space,
        num_samples=project_parameters.tune_iter,
        scheduler=tune_scheduler,
        local_dir=join(project_parameters.save_path, 'tuning_logs'),
        progress_reporter=reporter)
    best_trial = tuning_result.get_best_trial(
        'accuracy_difference', 'min', 'last')
    if not project_parameters.tune_debug:
        project_parameters = _set_tune_project_parameters(
            hyperparameter=best_trial.config, project_parameters=project_parameters)
        result = train(project_parameters=project_parameters)
        result['tune'] = tuning_result
    else:
        result = {'tune': tuning_result}
    print('best trial name: {}'.format(best_trial))
    print('best trial result: {}'.format(
        best_trial.last_result['accuracy_difference']))
    print('best trial config: {}'.format(best_trial.config))
    if 'cutoff_freq_low' in best_trial.config or 'cutoff_freq_high' in best_trial.config:
        best_trial.config['cutoff_freq'] = []
        for key in ['cutoff_freq_low', 'cutoff_freq_high']:
            value = best_trial.config.pop(key, None)
            if value is not None:
                best_trial.config['cutoff_freq'].append(value)
        best_trial.config['cutoff_freq'] = (
            '{},'*len(best_trial.config['cutoff_freq']))[:-1].format(*best_trial.config['cutoff_freq'])
    if 'parameters_config_path' in project_parameters:
        output = 'num_workers: {}'.format(project_parameters.num_workers)
        for k, v in best_trial.config.items():
            if k == 'cutoff_freq':
                output += '\n{}:\n  - {}\n  - {}'.format(
                    k, v.split(',')[0], v.split(',')[1])
            else:
                output += '\n{}: {}'.format(k, v)
        print('best trial config command:\n{}'.format(output))
    else:
        print('best trial config command: --num_workers {}{}'.format(project_parameters.num_workers, (' --{} {}' *
                                                                                                      len(best_trial.config)).format(*np.concatenate(list(zip(best_trial.config.keys(), best_trial.config.values()))))))
    shutdown()
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # tune
    result = tune(project_parameters=project_parameters)
