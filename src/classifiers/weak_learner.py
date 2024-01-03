import math

import numpy as np
import torch
from torch import Tensor, ones
from torch.utils.data import DataLoader

from classifiers.simple_learner import SimpleLearner, WeakLearnerTrainingResults, WeakLearnerTrainingValidationResults
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
            dataset_length=len(data_loader)
        )

        return training_result

    def fit_and_validate(
            self,
            train_data_loader: DataLoader[ItemType],
            validation_data_loader: DataLoader[ItemType],
            classes_mask: Tensor,
            optimizer: torch.optim.Optimizer,
            loss: WeightedBaseLoss = None,
            epochs: int = 10,
            verbose: int = 0,
    ) -> WeakLearnerTrainingValidationResults:
        """
        :param train_data_loader:
        :param validation_data_loader:
        :param classes_mask: tensor where the i-th entry contains the class label for the i-th sample
        :param optimizer:
        :param loss:
        :param verbose:
        :param epochs:
        """

        train_validate_result = self._simple_learner.fit_and_validate(
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader,
            optimizer=optimizer,
            loss=loss,
            train_loss_weights=self._weights,
            epochs=epochs,
            verbose=verbose
        )

        self._update_params(
            training_results=train_validate_result,
            classes_mask=classes_mask,
            dataset_length=len(train_data_loader)
        )

        return train_validate_result

    def _update_params(
            self,
            training_results: WeakLearnerTrainingResults,
            classes_mask: Tensor,
            dataset_length: int
    ):
        # sigmoid used to make sure that the error rate is a number \in [0, 1]
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        # TODO(pierluigi): ask biagio or check adaboost paper because I dont remember, what should error rate be? cumulative loss over every epoch or just over the last epoch?
        # Multiplying by dataloader length because loss is the per-sample average of each epoch
        cumulative_loss = (np.array(training_results.avg_train_loss) * dataset_length).sum(axis=0)
        self._error_rate = cumulative_loss
        self._beta = sigmoid(self._error_rate)

        self._update_weights_map(classes_mask=classes_mask, data=training_results.prediction_map)

    def _update_weights_map(self, classes_mask: Tensor, data: PredictionMap) -> None:
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
