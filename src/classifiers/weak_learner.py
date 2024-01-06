import math
from typing import Callable, Iterator

import torch
from torch import Tensor, ones
from torch.utils.data import DataLoader

from classifiers.simple_learner import SimpleLearner, WeakLearnerTrainingResults, WeakLearnerValidationResults
from datasets.custom_coco_dataset import ItemType
from loss_functions.base_weighted_loss import PredictionMap, WeightedBaseLoss


class WeakLearner:
    def __init__(
            self,
            weights: Tensor,
            k_classes: int = 2,
            device: torch.device = None
    ):
        """
        :param weights: AdaBoost weights - vector of dataset length that weighs the loss of each instance
        """

        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._k_classes: int = k_classes
        self._weights: Tensor = weights

        # Make sure weights device is the same as the one assigned to this learner
        self._weights = self._weights.to(device)

        self._weights.requires_grad_()

        self._simple_learner: SimpleLearner = SimpleLearner(k_classes=self._k_classes, device=device)
        self._error_rate: float = .0

        self._beta: float = .0
        """AdaBoost coefficient used to weigh each prediction"""

        self._weights_map: Tensor = ones(self._weights.shape[0], dtype=torch.bool, device=device)
        """
        Basically a prediction mask: for each sample contains either 1, 
        if this learner predicts it correctly, or 0, otherwise.
        Used by AdaBoost in conjunction with the Beta param to decrease the 
        weights of the samples predicted correctly
        """

    def fit(
            self,
            data_loader: DataLoader[ItemType],
            classes_mask: Tensor,
            optimizer: torch.optim.Optimizer,
            loss: WeightedBaseLoss = None,
            epochs: int = 5,
            verbose: int = 0,
    ) -> WeakLearnerTrainingResults:
        """
        :param data_loader:
        :param classes_mask: tensor where the i-th entry contains the class label for the i-th sample
        :param optimizer:
        :param loss:
        :param epochs:
        :param verbose:
        :return:
        """

        training_result = self._simple_learner.fit(
            data_loader=data_loader,
            optimizer=optimizer,
            loss_weights=self._weights,
            loss=loss,
            epochs=epochs,
            verbose=verbose
        )

        self._update_params(
            training_results=training_result,
            classes_mask=classes_mask,
        )

        return training_result

    def fit_and_validate(
            self,
            train_data_loader: DataLoader[ItemType],
            validation_data_loader: DataLoader[ItemType],
            classes_mask: Tensor,
            optimizer_builder: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> WeakLearnerValidationResults:
        """
        :param train_data_loader:
        :param validation_data_loader:
        :param classes_mask: tensor where the i-th entry contains the class label for the i-th sample
        :param optimizer_builder:
        :param loss:
        :param verbose:
        :param epochs:
        """
        classes_mask = classes_mask.to(self._device)

        train_validate_result = self._simple_learner.fit_and_validate(
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            optimizer=optimizer_builder(self._simple_learner.parameters()),
            loss=loss,
            train_loss_weights=self._weights,
            epochs=epochs,
            verbose=verbose
        )

        self._update_params(
            training_results=train_validate_result,
            classes_mask=classes_mask,
        )

        return train_validate_result

    def _update_params(
            self,
            training_results: WeakLearnerTrainingResults,
            classes_mask: Tensor,
    ):
        # sigmoid used to make sure that the error rate is a number \in [0, 1]

        # TODO(pierluigi): old approach
        # def sigmoid(x):
        #     return 1 / (1 + math.exp(-x))
        #
        # Take loss related just to last epoch,
        #   because we need to use it with the prediction map from only the last epoch
        # Also, just take avg loss instead of cumulative because loss is unbounded,
        #   which saturates the sigmoid
        # last_epoch_avg_loss = training_results.avg_train_loss[-1]
        # self._error_rate = last_epoch_avg_loss
        # self._beta = min(sigmoid(self._error_rate), 0.9)

        preds, pred_ids = training_results.last_epoch_prediction_map
        pred_ids = pred_ids.to(torch.int)
        error_mask = (torch.argmax(preds, dim=1) != classes_mask[pred_ids])

        # TODO see if this works
        # Weighted misclassification (0-1) loss
        self._error_rate = (self._weights[pred_ids])[error_mask].sum().item()
        print(f"ERROR RATE DEBUG: {self._error_rate}")  # TODO debug
        print(f"ERROR CLASSES: {torch.unique(classes_mask[pred_ids][error_mask], return_counts=True)}")
        self._beta = self._error_rate / (1 - self._error_rate) if self._error_rate < 0.5 else 0.99

        self._update_weights_map(classes_mask=classes_mask, data=training_results.last_epoch_prediction_map)

    def _update_weights_map(self, classes_mask: Tensor, data: PredictionMap) -> None:
        preds, ids = data

        # Recall that preds are the raw logits for each sample,
        #   so we need to pass them through softmax before doing argmax
        preds = torch.nn.functional.softmax(input=preds, dim=1)
        ids = ids.to(torch.int)

        max_preds: Tensor = torch.argmax(preds, dim=1)
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
