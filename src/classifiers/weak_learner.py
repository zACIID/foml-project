import math
from typing import Tuple

import torch
from torch import Tensor, ones, argmax, int32

from classifiers.simple_learner import SimpleLearner
from datasets.custom_coco_dataset import CocoDataset
from loss_functions.base_weighted_loss import ErrorMap


class WeakLearner:
    def __init__(
            self,
            dataset: CocoDataset,
            weights: Tensor,
            k_classes: int = 2,
            epochs: int = 10,
            verbose: int = 0,
            device: torch.device = None
    ):
        """
        :param dataset:
        :param weights: AdaBoost weights - vector of dataset length that weighs the loss of each instance
        :param epochs:
        :param verbose:
        :param device:
        """

        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._dataset: CocoDataset = dataset
        self._k_classes: int = k_classes
        self._weights: Tensor = weights

        # Make sure weights device is the same as the one assigned to this learner
        self._weights = self._weights.to(device)

        self._weights.requires_grad_()

        self._simple_learner: SimpleLearner = SimpleLearner(k_classes=self._k_classes, device=device)
        self._error_rate: float = .0
        self._accuracy: float = .0

        self._beta: float = .0
        """AdaBoost coefficient used to weigh each prediction"""

        self._weights_map: Tensor = ones(self._weights.shape[0], dtype=torch.bool, device=device)
        """
        Basically a prediction mask: for each sample contains either 1, 
        if this learner predicts it correctly, or 0, otherwise.
        Used by AdaBoost in conjunction with the Beta param to decrease the 
        weights of the samples predicted correctly
        """

        # TODO(pierluigi): not a fan of performing stuff in the constructor because constructors should not have side effects
        self._fit(epochs=epochs, verbose=verbose)

    def _fit(self, epochs: int = 5, verbose: int = 0) -> None:
        training_result: Tuple[ErrorMap, float] = self._simple_learner.fit(
            dataset=self._dataset,
            adaboost_weights=self._weights,
            epochs=epochs,
            verbose=verbose
        )
        error_map, cum_loss = training_result

        # sigmoid used to make sure that the error rate is a number \in [0, 1]
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        self._error_rate = cum_loss
        self._beta = sigmoid(self._error_rate)
        self._update_weights_map(error_map)

    def _update_weights_map(self, data: ErrorMap) -> None:
        classes_mask: Tensor = self._dataset.get_labels()
        preds, ids = data
        ids = ids.to(torch.int)
        max_preds: Tensor = torch.squeeze(torch.argmax(preds, dim=1), dim=1)
        self._weights_map[ids] = max_preds == classes_mask[ids]

    def predict(self, samples: Tensor) -> Tensor:
        """
            This function returns:
            a) a tensor of shape: (n_samples, k_classes)
            Where the n_samples is the number of input images
        """
        # Recall that predict here returns (k_classes, n_samples) shape tensors
        #   if `samples` is just one image (not a batch)
        #   but we want them transposed
        pred: Tensor = self._simple_learner.predict(samples)
        if samples.dim() == 3:  # the input is a single image (CxHxW)
            pred = torch.transpose(pred, dim0=0, dim1=1)

        return pred

    def get_error_rate(self) -> float:
        return self._error_rate

    def get_beta(self) -> float:
        return self._beta

    def get_weights(self) -> Tensor:
        return self._weights.detach()

    def get_weights_map(self) -> Tensor:
        return self._weights_map
