from collections import OrderedDict
from typing import Type

import torch
import torch.nn as nn
from torch import Tensor, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers.base import TrainingResults
from datasets.custom_coco_dataset import ItemType, BatchType
from loss_functions.base_weighted_loss import WeightedBaseLoss
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
                ('dropout1', nn.Dropout(p=0.5)),
                ('dense2', nn.Linear(in_features=4096, out_features=4096)),
                ('dropout2', nn.Dropout(p=0.5)),
                ('dense3', nn.Linear(in_features=4096, out_features=k_classes)),
            ])
        )
        self._layers = self._layers.to(self._device)

    def forward(self, batch: Tensor) -> Tensor:
        batch_ft_map: Tensor = self._layers(batch)

        return batch_ft_map

    def fit(
            self,
            data_loader: DataLoader[ItemType],
            optimizer: torch.optim.Optimizer,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> TrainingResults:
        # optimizer = SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-03)
        # optimizer = Adam(self.parameters(), lr=learning_rate,) #weight_decay=5e-03)
        loss = WeightedCrossEntropy() if loss is None else loss

        self.train()
        results = TrainingResults()
        for epoch in range(epochs):
            avg_loss = .0

            for batch in tqdm(data_loader):
                batch: BatchType
                _, x_batch, y_batch, wgt_batch = batch
                _, x_batch, y_batch, wgt_batch = (
                    _.to(self._device),
                    x_batch.to(self._device),
                    y_batch.to(self._device),
                    wgt_batch.to(self._device)
                )

                y_pred: Tensor = self(x_batch)

                batch_loss: Tensor = loss(
                    y_true=y_batch,
                    y_pred=y_pred,
                    weights=wgt_batch
                )
                avg_loss += (x_batch.shape[0] / len(data_loader)) * batch_loss.item()

                optimizer.zero_grad()  # initialize gradient to zero
                batch_loss.backward()  # compute gradient
                optimizer.step()  # backpropagation

            results.train_loss.append(avg_loss)
            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} loss is {avg_loss}\033[0m")
                pass

        return results

    def predict(self, samples: Tensor) -> Tensor:
        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            raw_pred = self.__call__(samples)

            dim: int = 0 if samples.dim() == 3 else 1
            return nn.functional.softmax(input=raw_pred, dim=dim)

    def get_modules(self) -> nn.Sequential:
        return self._layers

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._layers = new_modules
