import torch.nn as nn
from torch import Tensor, inner
from torch import linalg as euclidean
from base_weighted_loss import WeightedBaseLoss


class WeightedCrossEntropy(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            cross_entropy: Tensor = nn.CrossEntropyLoss()
            weighted_y_true: Tensor = y_true * (weights.unsqueeze(dim=1))
            return cross_entropy(weighted_y_true, y_pred)

        super().__init__(sub_loss)
