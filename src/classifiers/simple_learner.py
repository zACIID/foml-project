from collections import OrderedDict
from typing import Type, Tuple

import torch
import torch.nn as nn
from torch import Tensor, no_grad
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from datasets.custom_coco_dataset import ItemType, BatchType
from layers.fire_layer import FireLayer
from loss_functions.base_weighted_loss import WeightedBaseLoss, ErrorMap
from loss_functions.weighted_cross_entropy import WeightedCrossEntropy


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
        dim: int = 0 if batch.dim() == 3 else 1
        batch_ft_map: Tensor = self._layers(batch)

        return nn.functional.softmax(input=batch_ft_map, dim=dim)

    def fit(
            self,
            dataset: Dataset[ItemType],
            adaboost_weights: Tensor,
            loss: WeightedBaseLoss = None,
            batch_size: int = 32,
            epochs: int = 10,
            verbose: int = 0
    ) -> Tuple[ErrorMap, float]:
        adam_opt: Adam = Adam(self.parameters())

        loss = WeightedCrossEntropy() if loss is None else loss
        loss.to(device=self._device)
        cum_loss: float = .0

        adaboost_weights = adaboost_weights.to(self._device)

        self.train()
        for epoch in range(epochs):
            cum_loss = .0
            for batch in DataLoader(dataset, batch_size, shuffle=True):
                batch: BatchType
                ids, x_batch, y_batch, wgt_batch = batch

                y_pred: Tensor = self(x_batch)
                mixed_weights: Tensor = adaboost_weights[ids] * wgt_batch

                batch_loss: Tensor = loss(
                    y_true=y_batch,
                    y_pred=y_pred,
                    weights=mixed_weights,
                    ids=ids,
                    save=True if epoch == epochs - 1 else False
                )
                cum_loss += batch_loss.item()

                adam_opt.zero_grad()  # initialize gradient to zero
                batch_loss.backward()  # compute gradient
                adam_opt.step()  # backpropagation

            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} loss is {cum_loss}\033[0m")

        return loss.get_error_map(), cum_loss

    def predict(self, samples: Tensor) -> Tensor:
        """
            This function either returns:
            a) a tensor of shape: (k_classes, 1) if the number of samples is 1
            b) a tensor of shape: (n_samples, k_classes) otherwise
        """
        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            return self.__call__(samples)

    def get_modules(self) -> nn.Sequential:
        return self._layers

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._layers = new_modules
        self._layers.to(self._device)
