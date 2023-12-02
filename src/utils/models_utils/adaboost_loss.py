import torch.nn as nn
from torch import Tensor, inner
from torch import linalg as euclidean


class WeightedKLD(nn.Module):
    def __init__(self):
        super().__init__()
        self._errorMap: list[dict[str, Tensor]] = []

    def forward(self, y_true: Tensor, y_pred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        distancesTensor: Tensor = y_true - y_pred
        euclideanDistances: Tensor = euclidean.norm(distancesTensor, dim=1)
        weightedDistances: Tensor = inner(weights, euclideanDistances)

        if save:
            self._errorMap.append({
                "yPred": y_pred,
                "ids": ids
            })

        return weightedDistances

    def getErrorMap(self) -> list[dict[str, Tensor]]:
        return self._errorMap


class WeightedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
