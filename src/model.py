# import
import timm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningModule
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix
import pandas as pd
import numpy as np
from src.utils import load_checkpoint, load_yaml
import torch.optim as optim
from os.path import basename, dirname
import torch.nn.functional as F

# def


def _get_backbone_model_from_file(filepath):
    import sys
    sys.path.append('{}'.format(dirname(filepath)))
    class_name = basename(filepath).split('.')[0]
    exec('from {} import {}'.format(*[class_name]*2))
    return eval('{}()'.format(class_name))


def _get_backbone_model(project_parameters):
    if project_parameters.backbone_model in timm.list_models():
        backbone_model = timm.create_model(model_name=project_parameters.backbone_model,
                                           pretrained=False, num_classes=project_parameters.num_classes, in_chans=1)
    elif '.py' in project_parameters.backbone_model:
        backbone_model = _get_backbone_model_from_file(
            filepath=project_parameters.backbone_model)
    else:
        assert False, 'please check the backbone model. the backbone model: {}'.format(
            project_parameters.backbone_model)
    return backbone_model


def _get_loss_function(project_parameters):
    if 'data_weight' in project_parameters:
        weight = torch.Tensor(list(project_parameters.data_weight.values()))
    else:
        weight = None
    return nn.CrossEntropyLoss(weight=weight)


def _get_optimizer(model_parameters, project_parameters):
    optimizer_config = load_yaml(
        filepath=project_parameters.optimizer_config_path)
    optimizer_name = list(optimizer_config.keys())[0]
    if optimizer_name in dir(optim):
        for name, value in optimizer_config.items():
            if value is None:
                optimizer = eval('optim.{}(params=model_parameters, lr={})'.format(
                    optimizer_name, project_parameters.lr))
            elif type(value) is dict:
                value = ('{},'*len(value)).format(*['{}={}'.format(a, b)
                                                    for a, b in value.items()])
                optimizer = eval('optim.{}(params=model_parameters, lr={}, {})'.format(
                    optimizer_name, project_parameters.lr, value))
            else:
                assert False, '{}: {}'.format(name, value)
        return optimizer
    else:
        assert False, 'please check the optimizer. the optimizer config: {}'.format(
            optimizer_config)


def _get_lr_scheduler(project_parameters, optimizer):
    if project_parameters.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=project_parameters.step_size, gamma=project_parameters.gamma)
    elif project_parameters.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=project_parameters.step_size)
    else:
        assert False, 'please check the lr scheduler. the lr scheduler: {}'.format(
            project_parameters.lr_scheduler)
    return lr_scheduler


def create_model(project_parameters):
    model = Net(project_parameters=project_parameters)
    if project_parameters.checkpoint_path is not None:
        model = load_checkpoint(model=model, use_cuda=project_parameters.use_cuda,
                                checkpoint_path=project_parameters.checkpoint_path)
    return model

# class


class Net(LightningModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.backbone_model = _get_backbone_model(
            project_parameters=project_parameters)
        self.activation_function = nn.Softmax(dim=-1)
        self.loss_function = _get_loss_function(
            project_parameters=project_parameters)
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(
            num_classes=project_parameters.num_classes)

    def forward(self, x):
        return self.activation_function(self.backbone_model(x))

    def get_progress_bar_dict(self):
        # don't show the loss value
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def _parse_outputs(self, outputs, calculate_confusion_matrix):
        epoch_loss = []
        epoch_accuracy = []
        if calculate_confusion_matrix:
            y_true = []
            y_pred = []
        for step in outputs:
            epoch_loss.append(step['loss'].item())
            epoch_accuracy.append(step['accuracy'].item())
            if calculate_confusion_matrix:
                y_pred.append(step['y_hat'])
                y_true.append(step['y'])
        if calculate_confusion_matrix:
            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            confmat = pd.DataFrame(self.confusion_matrix(y_pred, y_true).tolist(
            ), columns=self.project_parameters.classes.keys(), index=self.project_parameters.classes.keys()).astype(int)
            return epoch_loss, epoch_accuracy, confmat
        else:
            return epoch_loss, epoch_accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        train_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': train_step_accuracy}

    def training_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('training loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('training accuracy', np.mean(epoch_accuracy))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        val_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': val_step_accuracy}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('validation loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('validation accuracy', np.mean(epoch_accuracy))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        test_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': test_step_accuracy, 'y_hat': F.softmax(y_hat), 'y': y}

    def test_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_accuracy, confmat = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=True)
        self.log('test loss', np.mean(epoch_loss))
        self.log('test accuracy', np.mean(epoch_accuracy))
        print(confmat)

    def configure_optimizers(self):
        optimizer = _get_optimizer(model_parameters=self.parameters(
        ), project_parameters=self.project_parameters)
        if self.project_parameters.step_size > 0:
            lr_scheduler = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    model.summarize()

    # create input data
    x = torch.ones(project_parameters.batch_size, 1, 224, 224)

    # get model output
    y = model.forward(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)
