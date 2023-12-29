import torch.nn as nn
from torch import Tensor

from .base_weighted_loss import WeightedBaseLoss


class WeightedCrossEntropy(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
            loss = cross_entropy(y_pred, y_true)
            loss = loss * weights
            loss = loss.mean()
            return loss

        super().__init__(sub_loss)
