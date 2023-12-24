from typing import Type

import torch.nn as nn
from torch import Tensor, no_grad
from torch.optim import Adam

from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader

from datasets.custom_coco_dataset import ItemType, BatchType
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
    
    def __init__(self, k_classes: int = 2, act_fun: Type[nn.Module] = nn.SiLU, device: str = None):

        super().__init__()

        self._pipe_line: nn.Sequential = nn.Sequential(
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
                ('flatten', nn.Flatten()),
                ('dense1', nn.Linear(in_features=256*5*5, out_features=4096)),
                ('dropout1', nn.Dropout(p=0.5)),
                ('dense2', nn.Linear(in_features=4096, out_features=4096)),
                ('dropout2', nn.Dropout(p=0.5)),
                ('dense3', nn.Linear(in_features=4096, out_features=k_classes)),
                ('softmax', nn.Softmax(dim=0))
            ])
        )

    def forward(self, batch: Tensor) -> Tensor:
        return self._pipe_line(batch)

    def fit(self, dataset: Dataset[ItemType], batch_size: int = 32,
            epochs: int = 10, loss: nn.Module = None,
            verbose: int = 0) -> float:

        adam_opt: Adam = Adam(self.parameters())
        loss = WeightedCrossEntropy() if loss is None else loss
        cum_loss: float = .0

        self.train()
        for epoch in range(epochs):
            cum_loss = .0
            for batch in DataLoader(dataset, batch_size, shuffle=True):
                batch: BatchType
                _, x_batch, y_batch, wgt_batch = batch

                y_pred: Tensor = self(x_batch)

                batch_loss: Tensor = loss(
                    # TODO rivedere il tipo della loss
                    y_true=y_batch, y_pred=y_pred, weights=wgt_batch
                )
                cum_loss += batch_loss.item()

                adam_opt.zero_grad()  # initialize gradient to zero
                batch_loss.backward()  # compute gradient
                adam_opt.step()  # backpropagation

            if verbose >= 1:
                print(f"\033[32mEpoch:{epoch} loss is {cum_loss}\033[0m")

        return cum_loss

    def predict(self, samples: Tensor) -> Tensor:
        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            return self.__call__(samples)

    def get_modules(self) -> nn.Sequential:
        return self._pipe_line

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._pipe_line = new_modules
