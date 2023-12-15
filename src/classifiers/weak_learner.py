import math

import torch
from torch import Tensor, ones, argmax, int32

from classifiers.simple_learner import SimpleLearner
from datasets.custom_coco_dataset import CocoDataset


class WeakLearner:
    def __init__(
            self, dataset: CocoDataset, weights: Tensor,
            epochs: int = 10, verbose: int = 0
    ):
        self._dataset: CocoDataset = dataset
        self._weights: Tensor = weights
        self._weights.requires_grad_()
        self._simple_learner: SimpleLearner = SimpleLearner()
        self._error_rate: float = .0
        self._beta: float = .0
        self._accuracy: float = .0
        self._weights_map: Tensor = ones(self._weights.shape[0], dtype=torch.bool)

        self._fit(epochs=epochs, verbose=verbose)

    def _fit(self, epochs: int = 5, verbose: int = 0) -> None:
        training_result: tuple[tuple[Tensor, Tensor], float] = self._simple_learner.fit(
            dataset=self._dataset, adaboost_wgt=self._weights,
            epochs=epochs, verbose=verbose
        )

        # sigmoid used to make sure that the error rate is a number \in [0, 1]
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        self._error_rate = training_result[1]
        self._beta = sigmoid(self._error_rate)
        self._update_weights_map(training_result[0])

    def _update_weights_map(self, data: tuple[Tensor, Tensor]) -> None:
        classes_mask: Tensor = self._dataset.get_labels
        preds, ids = data
        for pred, _id in zip(preds, ids):
            model_pred: int = argmax(pred).item()
            weight_flag: bool = model_pred == classes_mask[_id].value
            self._weights_map[_id.to(int32)] = weight_flag

    def predict(self, samples: Tensor) -> Tensor:
        return self._simple_learner.predict(samples)

    def get_error_rate(self) -> float:
        return self._error_rate

    def get_beta(self) -> float:
        return self._beta

    def get_weights(self) -> Tensor:
        return self._weights.detach()

    def get_weights_map(self) -> Tensor:
        return self._weights_map
