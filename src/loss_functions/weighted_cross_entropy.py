import torch.nn as nn
from torch import Tensor

from .base_weighted_loss import WeightedBaseLoss


class WeightedCrossEntropy(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            loss = cross_entropy(y_true, y_pred)

            # TODO(pierluigi): weighted loss significa che vogliamo moltiplicare la loss finale, non y_true prima di darlo in pasto alla loss, corretto?
            weighted_loss = loss * weights
            return weighted_loss

        super().__init__(sub_loss)
