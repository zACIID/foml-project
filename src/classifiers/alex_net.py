from collections import OrderedDict
from typing import Type, Tuple

import torch
import torch.nn as nn
from torch import Tensor, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers.base import TrainingResults, TrainingValidationResults
from datasets.custom_coco_dataset import ItemType, BatchType
from loss_functions.base_weighted_loss import WeightedBaseLoss, PredictionMap
from loss_functions.weighted_cross_entropy import WeightedCrossEntropy


class AlexNet(nn.Module):
    """
    - 3x224x224
    - Conv 11x11, stride 4, x96 -> output 96x54x54
    - MaxPooling 3x3, stride 2 -> output 96x26x26
    - Conv 5x5, padding 2, x256 -> output 256x26x26
    - MaxPooling 3x3, stride 2 -> output 256x12x12
    - Conv 3x3, padding 1, x384 -> output 384x12x12
    - Conv 3x3, padding 1, x384 -> output 384x12x12
    - Conv 3x3, padding 1, x256 -> output 256x12x12
    - MaxPooling 3x3, stride 2 -> output 256x5x5
    - Dense 4096, dropout 0.5
    - Dense 4096, dropout 0.5
    - Dense k_classes
    - Softmax
    """

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

        self._layers: nn.Sequential = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels=3, out_channels=96,
                                    kernel_size=11, stride=4,
                                    device=device)),
                ('act_fun1', act_fun()),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(in_channels=96, out_channels=256,
                                    kernel_size=5, padding=2,
                                    device=device)),
                ('act_fun2', act_fun()),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', nn.Conv2d(in_channels=256, out_channels=384,
                                    kernel_size=3, padding=1,
                                    device=device)),
                ('act_fun3', act_fun()),
                ('conv4', nn.Conv2d(in_channels=384, out_channels=384,
                                    kernel_size=3, padding=1,
                                    device=device)),
                ('act_fun4', act_fun()),
                ('conv5', nn.Conv2d(in_channels=384, out_channels=256,
                                    kernel_size=3, padding=1,
                                    device=device)),
                ('act_fun5', act_fun()),
                ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('avgpool1', nn.AdaptiveAvgPool2d((6, 6))),
                ('flatten1', nn.Flatten()),
                ('dense1', nn.Linear(in_features=256*6*6, out_features=4096)),
                ('act_fun6', act_fun()),
                ('dropout1', nn.Dropout(p=0.5)),
                ('dense2', nn.Linear(in_features=4096, out_features=4096)),
                ('act_fun7', act_fun()),
                ('dropout2', nn.Dropout(p=0.5)),
                ('dense3', nn.Linear(in_features=4096, out_features=k_classes)),
            ])
        )
        self._layers = self._layers.to(self._device)

    def forward(self, batch: Tensor) -> Tensor:
        return self._layers(batch)

    def fit(
            self,
            data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> TrainingResults:
        loss = WeightedCrossEntropy() if loss is None else loss
        loss = loss.to(device=self._device)

        results = TrainingResults()
        for epoch in range(epochs):
            avg_loss_train, accuracy_train, prediction_map = self._train_epoch(
                data_loader=data_loader,
                optimizer=optimizer,
                loss=loss,
            )
            results.avg_train_loss.append(avg_loss_train)
            results.prediction_map = prediction_map

            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} train loss is {avg_loss_train}\033[0m")
                print(f"\033[32mEpoch:{epoch} train accuracy is {accuracy_train}\033[0m")

        return results

    def fit_and_validate(
            self,
            train_data_loader: DataLoader[ItemType],
            validation_data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> TrainingValidationResults:
        loss = WeightedCrossEntropy() if loss is None else loss
        loss = loss.to(device=self._device)

        results = TrainingValidationResults()
        for epoch in range(epochs):
            avg_loss_train, accuracy_train, prediction_map = self._train_epoch(
                data_loader=train_data_loader,
                optimizer=optimizer,
                loss=loss,
            )
            results.avg_train_loss.append(avg_loss_train)
            results.train_accuracy.append(accuracy_train)
            results.prediction_map = prediction_map

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
            loss: WeightedBaseLoss,
    ) -> Tuple[float, float, PredictionMap | None]:
        """
        :param data_loader:
        :param optimizer:
        :param loss:
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

            weights: Tensor = wgt_batch
            batch_avg_loss: Tensor = loss(
                y_true=y_batch,
                y_pred=y_logits,
                weights=weights,
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
                )
                avg_loss += (batch_length / len(data_loader.dataset)) * batch_avg_loss.item()

                # torch.max(x, dim=1) returns a tuple (values, indices)
                scores, predictions = torch.max(y_logits, dim=1)
                predictions: Tensor

                # noinspection PyUnresolvedReferences
                accuracy += (predictions == y_batch).sum().item() / len(data_loader.dataset)

        return avg_loss, accuracy

    def predict(self, samples: Tensor) -> Tensor:
        dim: int = 0 if samples.dim() == 3 else 1
        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            raw_pred = self.__call__(samples)
            return torch.argmax(nn.functional.softmax(input=raw_pred, dim=dim), dim=dim)

    def get_modules(self) -> nn.Sequential:
        return self._layers

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._layers = new_modules
