from typing import Callable

import torch.nn as nn
from numpy import ndarray, array, append
from torch import Tensor


class WeightedBaseLoss(nn.Module):
    def __init__(self, sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor]):
        super().__init__()
        self._error_map: ndarray[dict[str, Tensor]] = array([])
        self._sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor] = sub_loss

    def forward(self, y_true: Tensor, y_pred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        if save:
            # TODO(pierluigi): maybe use two separate np.ndarrays
            self._error_map = append(self._error_map, {
                "y_pred": y_pred,
                "ids": ids
            })

        return self._sub_loss(y_true, y_pred, weights)

    def getErrorMap(self) -> ndarray[dict[str, Tensor]]:
        return self._error_map


