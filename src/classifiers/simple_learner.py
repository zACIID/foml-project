import dataclasses
from collections import OrderedDict
from typing import Type, Tuple

import torch
import torch.nn as nn
from torch import Tensor, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers.base import TrainingResults, TrainingValidationResults
from datasets.custom_coco_dataset import ItemType, BatchType
from layers.fire_layer import FireLayer
from loss_functions.base_weighted_loss import WeightedBaseLoss, PredictionMap
from loss_functions.weighted_cross_entropy import WeightedCrossEntropy


@dataclasses.dataclass
class WeakLearnerTrainingResults(TrainingResults):
    last_epoch_prediction_map: PredictionMap | None = None


@dataclasses.dataclass
class WeakLearnerValidationResults(TrainingValidationResults, WeakLearnerTrainingResults):
    pass


# TODO(pierluigi): maybe merge this and the WeakLearner classes into one since they are dependent on each other
class SimpleLearner(nn.Module):
    def __init__(
            self,
            k_classes: int = 2,
            act_fun: Type[nn.Module] = nn.SiLU,
            device: torch.device = None
    ):
        super().__init__()

        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._k_classes = k_classes

        self._layers: nn.Sequential = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels=3, out_channels=96,
                                    kernel_size=7, stride=2, device=device)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('fire1', FireLayer(in_channels_sqz=96, out_channels_sqz=32,
                                    out_channels_exp_ones=64,
                                    out_channels_exp_threes=64, act_fun=act_fun,
                                    device=device)),
                ('act_fun1', act_fun()),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('fire2', FireLayer(in_channels_sqz=128, out_channels_sqz=64,
                                    out_channels_exp_ones=256,
                                    out_channels_exp_threes=256, act_fun=act_fun,
                                    device=device)),
                ('act_fun2', act_fun()),
                ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(in_channels=512, out_channels=k_classes,
                                    kernel_size=1, device=device)),
                ('averagepool', nn.AvgPool2d(kernel_size=12, stride=1)),
                ('flatten', nn.Flatten()),
            ])
        )
        self._layers = self._layers.to(self._device)

    def forward(self, batch: Tensor) -> Tensor:
        return self._layers(batch)

    def fit(
            self,
            data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            loss_weights: Tensor,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> WeakLearnerTrainingResults:
        loss = WeightedCrossEntropy() if loss is None else loss
        loss = loss.to(device=self._device)
        loss_weights = loss_weights.to(self._device)

        results = WeakLearnerTrainingResults()
        for epoch in range(epochs):
            avg_loss_train, accuracy_train, prediction_map = self._train_epoch(
                data_loader=data_loader,
                optimizer=optimizer,
                loss_weights=loss_weights,
                loss=loss,
                save_prediction_map=True if epoch == epochs - 1 else False,
            )
            results.avg_train_loss.append(avg_loss_train)
            results.last_epoch_prediction_map = prediction_map

            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} train loss is {avg_loss_train}\033[0m")
                print(f"\033[32mEpoch:{epoch} train accuracy is {accuracy_train}\033[0m")

        return results

    def fit_and_validate(
            self,
            train_data_loader: DataLoader[ItemType],
            validation_data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            train_loss_weights: Tensor,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> WeakLearnerValidationResults:
        loss = WeightedCrossEntropy() if loss is None else loss
        loss = loss.to(device=self._device)
        train_loss_weights = train_loss_weights.to(self._device)

        results = WeakLearnerValidationResults()
        for epoch in range(epochs):
            avg_loss_train, accuracy_train, prediction_map = self._train_epoch(
                data_loader=train_data_loader,
                optimizer=optimizer,
                loss_weights=train_loss_weights,
                loss=loss,
                save_prediction_map=True if epoch == epochs - 1 else False,
            )
            results.avg_train_loss.append(avg_loss_train)
            results.train_accuracy.append(accuracy_train)
            results.last_epoch_prediction_map = prediction_map

            avg_loss_val, accuracy_val = self._validation_epoch(
                data_loader=validation_data_loader,
                loss=loss,
            )
            results.avg_validation_loss.append(avg_loss_val)
            results.validation_accuracy.append(accuracy_val)

            # TODO(pierluigi): Maybe create a shared function in the base module,
            #   especially for AdaBoost stuff that is closely related and tends to just
            #   forward stuff to Simplelearner?

            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} train loss is {avg_loss_train}\033[0m")
                print(f"\033[32mEpoch:{epoch} validation loss is {avg_loss_val}\033[0m")
                print(f"\033[32mEpoch:{epoch} train accuracy is {accuracy_train}\033[0m")
                print(f"\033[32mEpoch:{epoch} validation accuracy is {accuracy_val}\033[0m")

        return results

    def _train_epoch(
            self,
            data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            loss_weights: Tensor,
            loss: WeightedBaseLoss,
            save_prediction_map,
    ) -> Tuple[float, float, PredictionMap | None]:
        """
        :param data_loader:
        :param optimizer:
        :param loss_weights:
        :param loss:
        :param save_prediction_map:
        :return: (average loss per sample, accuracy, prediction map from loss function)
        """

        avg_loss = .0
        accuracy = .0

        for batch in tqdm(data_loader):
            batch: BatchType
            ids, x_batch, y_batch, wgt_batch = batch
            ids, x_batch, y_batch, wgt_batch = (
                ids.to(self._device),
                x_batch.to(self._device),
                y_batch.to(self._device),
                wgt_batch.to(self._device)
            )
            batch_length = x_batch.shape[0]

            y_logits: Tensor = self(x_batch)

            weights: Tensor = loss_weights[ids] * wgt_batch
            batch_avg_loss: Tensor = loss(
                y_true=y_batch,
                y_pred=y_logits,
                weights=weights,
                ids=ids,
                save=save_prediction_map
            )
            avg_loss += (batch_length / len(data_loader.dataset)) * batch_avg_loss.item()

            optimizer.zero_grad()  # initialize gradient to zero
            batch_avg_loss.backward()  # compute gradient
            optimizer.step()  # backpropagation

            # torch.max(x, dim=1) returns a tuple (values, indices)
            scores, predictions = torch.max(y_logits, dim=1)
            predictions: Tensor

            # noinspection PyUnresolvedReferences
            accuracy += (predictions == y_batch).sum().item() / len(data_loader.dataset)

        prediction_map = loss.get_prediction_map()

        return avg_loss, accuracy, prediction_map

    def _validation_epoch(
            self,
            data_loader: DataLoader[ItemType],
            loss: WeightedBaseLoss,
    ):
        avg_loss = .0
        accuracy = .0

        self.eval()
        with torch.no_grad():

            for batch in tqdm(data_loader):
                batch: BatchType
                ids, x_batch, y_batch, wgt_batch = batch
                ids, x_batch, y_batch, wgt_batch = (
                    ids.to(self._device),
                    x_batch.to(self._device),
                    y_batch.to(self._device),
                    wgt_batch.to(self._device)
                )
                batch_length = x_batch.shape[0]

                y_logits: Tensor = self(x_batch)

                weights: Tensor = wgt_batch
                batch_avg_loss: Tensor = loss(
                    y_true=y_batch,
                    y_pred=y_logits,
                    weights=weights,
                    ids=ids,
                    save=False
                )
                avg_loss += (batch_length / len(data_loader.dataset)) * batch_avg_loss.item()

                # torch.max(x, dim=1) returns a tuple (values, indices)
                scores, predictions = torch.max(y_logits, dim=1)
                predictions: Tensor

                # noinspection PyUnresolvedReferences
                accuracy += (predictions == y_batch).sum().item() / len(data_loader.dataset)

        return avg_loss, accuracy

    def predict(self, samples: Tensor) -> Tensor:
        """
        This function either returns:
        a) a tensor of shape: (k_classes, 1) if the number of samples is 1
        b) a tensor of shape: (n_samples, k_classes) otherwise
        """
        dim: int = 0 if samples.dim() == 3 else 1
        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            raw_pred = self.__call__(samples)
            return nn.functional.softmax(input=raw_pred, dim=dim)

    def get_modules(self) -> nn.Sequential:
        return self._layers

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._layers = new_modules
        self._layers.to(self._device)
