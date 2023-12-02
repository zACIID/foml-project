import torch.nn as nn
from torch import Tensor, inner
from torch import linalg as euclidean


class WeightedDistance(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            distances_tensor: Tensor = y_true - y_pred
            euclidean_distances: Tensor = euclidean.norm(distances_tensor, dim=1)
            return inner(weights, euclidean_distances)

        super().__init__(sub_loss)
