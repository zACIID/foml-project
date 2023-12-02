import torch as th
from numpy import ndarray, full
from torch import tensor, Tensor, ones, float32


class WeakLearner:
    def __init__(
            self, dataset: list[Tensor], labels_set: Tensor, weights: Tensor,
            epochs: int = 10, verbose: int = 0
    ):
        # TODO self._base_learner: BaseLearner
        self._errorRate: float = .0
        self._beta: float = .0
        self._accuracy: float = .0
        self._dataset: list[Tensor] = dataset
        self._labels: Tensor = labels_set
        self._weights: Tensor = weights
        self._weightsMap: Tensor = ones(self._weights.numel(), dtype=th.bool)
        self._fit(epochs=epochs, verbose=verbose)

    def _fit(self, epochs: int = 5, verbose: int = 0) -> None:
        pass

    def _updateWeightsMap(self, data: list[dict[str, Tensor]]) -> None:
        pass

    def predictAll(self, samples: Tensor) -> Tensor:
        return self._baseLearner.predictAll(samples)

    def predictOne(self, sample: Tensor) -> Tensor:
        return self._baseLearner.predict(sample)

    def getErrorRate(self) -> float:
        return self._errorRate

    def getBeta(self) -> float:
        return self._beta

    def getWeights(self) -> Tensor:
        return self._weights.detach()

    def getWeightsMap(self) -> Tensor:
        return self._weightsMap


