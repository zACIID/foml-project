from typing import Callable

import torch.nn as nn
from torch import Tensor, tensor, cat


class WeightedBaseLoss(nn.Module):
    def __init__(self, sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor]):
        super().__init__()
        self._pred: Tensor = tensor([])
        self._ids: Tensor = tensor([])
        self._sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor] = sub_loss

    def forward(self, y_true: Tensor, y_pred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        if save:
            self._pred = cat((self._pred, y_pred))
            self._ids = cat((self._ids, ids))

        return self._sub_loss(y_true, y_pred, weights)

    def get_error_map(self) -> tuple[Tensor, Tensor]:
        return self._pred, self._ids
