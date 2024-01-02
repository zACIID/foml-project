from typing import List, Sequence, Tuple

import torch
import torch as th
from torch import Tensor, sum
from torch.utils.data import DataLoader

from classifiers.simple_learner import WeakLearnerTrainingResults
from loss_functions.base_weighted_loss import WeightedBaseLoss
from src.classifiers.strong_learner import StrongLearner
from src.classifiers.weak_learner import WeakLearner
from src.datasets.custom_coco_dataset import Labels, ItemType


class AdaBoost:
    """
        Class wrapping all the functionalities needed to make a training algorithm based
        on an ensemble approach
    """

    def __init__(
            self,
            n_classes: int,
            device: torch.device = None
    ):
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._k_classes: int = n_classes
        self._weak_learners: List[WeakLearner] = []

        self._weights: Tensor | None = None
        """
        Weights associated to each sample of the previous training run. 
        None if fit() has never been called
        """

    def fit(
            self,
            eras: int,
            data_loader: DataLoader[ItemType],
            classes_mask: Tensor,
            class_cardinalities: Tensor,
            weak_learner_optimizer: torch.optim.Optimizer,
            weak_learner_loss: WeightedBaseLoss = None,
            weak_learner_epochs: int = 5,
            verbose: int = 0,
    ) -> Tuple[StrongLearner, Sequence[WeakLearnerTrainingResults]]:
        self._weights = _initialize_weights(classes_mask=classes_mask, class_cardinalities=class_cardinalities)

        weak_learner_results: List[WeakLearnerTrainingResults] = []
        for era in range(eras):
            normalized_weights = _normalize_weights(self._weights)

            weak_learner: WeakLearner = WeakLearner(
                weights=normalized_weights,
                device=self._device
            )

            results = weak_learner.fit(
                data_loader=data_loader,
                classes_mask=classes_mask,
                optimizer=weak_learner_optimizer,
                loss=weak_learner_loss,
                epochs=weak_learner_epochs,
                verbose=verbose
            )

            _update_weights_(
                weights=normalized_weights,
                weak_learner_beta=weak_learner.get_beta(),
                weak_learner_weights_map=weak_learner.get_weights_map()
            )

            self._weak_learners.append(weak_learner)
            weak_learner_results.append(results)

            if verbose > 1:
                print(f"\033[31mEras left: {eras - (era + 1)}\033[0m")

        return StrongLearner(weak_learners=self._weak_learners, device=self._device), weak_learner_results

    def get_weights(self) -> Tensor:
        return self._weights


def _initialize_weights(classes_mask: Tensor, class_cardinalities: Tensor) -> Tensor:
    # TODO(pierluigi): capire se anche weights ha senso che sia messo nella cpu
    weights: Tensor = th.zeros(classes_mask.shape[0])

    for lbl in Labels:
        cardinality = class_cardinalities[lbl]
        weights[classes_mask == int(lbl)] = 1 / (2 * cardinality)

    return weights


def _normalize_weights(weights: Tensor) -> Tensor:
    return weights / sum(weights)


def _update_weights_(
        weights: Tensor,
        weak_learner_beta: float,
        weak_learner_weights_map: Tensor
) -> None:
    """Update is performed in-place"""

    # TODO(pierluigi): I think in place update has to access the data field else the following error occurs:
    #   https://stackoverflow.com/questions/73616963/runtimeerror-a-view-of-a-leaf-variable-that-requires-grad-is-being-used-in-an
    weights.data[weak_learner_weights_map] *= weak_learner_beta

