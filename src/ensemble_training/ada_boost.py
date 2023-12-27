from typing import Callable, List

import torch
import torch as th
from torch import Tensor, sum

from classifiers.strong_learner import StrongLearner
from classifiers.weak_learner import WeakLearner
from datasets.custom_coco_dataset import CocoDataset, Labels

"""
    Class wrapping all the functionalities needed to make a training algorithm based
    on an ensemble approach
"""


class AdaBoost:
    def __init__(
            self,
            dataset: CocoDataset,
            n_eras: int,
            n_classes: int,
            weak_learner_epochs: int = 10,
            device: torch.device = None
    ):
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._n_eras: int = n_eras
        self._dataset: CocoDataset = dataset
        self._n_classes: int = n_classes
        self._weak_learner_epochs: int = weak_learner_epochs
        self._weak_learners: List[WeakLearner] = []
        self._weights: Tensor = self._initialize_weights()

    def _initialize_weights(self) -> Tensor:
        classes_mask: Tensor = self._dataset.get_labels()

        # TODO(pierluigi): capire se anche weights ha senso che sia messo nella cpu
        weights: Tensor = th.zeros(classes_mask.shape[0])

        for lbl in Labels:
            class_cardinality: int = self._dataset.get_class_cardinality(label=lbl)
            weights[classes_mask == int(lbl)] = 1 / (2 * class_cardinality)

        return weights

    @staticmethod
    def normalize_weights(weights: Tensor) -> Tensor:
        return weights / sum(weights)

    @staticmethod
    def update_weights(
            weights: Tensor,
            weak_learner_beta: float,
            weak_learner_weights_map: Tensor
    ) -> None:
        """Update is performed in-place"""

        # TODO(pierluigi): I think in place update has to access the data field else the following error occurs:
        #   https://stackoverflow.com/questions/73616963/runtimeerror-a-view-of-a-leaf-variable-that-requires-grad-is-being-used-in-an
        weights.data[weak_learner_weights_map] *= weak_learner_beta

    def start_generator(
            self,
            update_weights: bool = True,
            verbose: int = 0
    ) -> Callable[[Tensor], StrongLearner]:
        # Returning a function here is maybe useful if multiprocessing is to be used
        def detached_start(weights: Tensor) -> StrongLearner:
            for era in range(self._n_eras):
                weights = AdaBoost.normalize_weights(weights)

                weak_learner: WeakLearner = WeakLearner(
                    dataset=self._dataset,
                    weights=self._weights,
                    epochs=self._weak_learner_epochs,
                    verbose=verbose,
                    device=self._device
                )

                if update_weights:
                    AdaBoost.update_weights(
                        weights=weights,
                        weak_learner_beta=weak_learner.get_beta(),
                        weak_learner_weights_map=weak_learner.get_weights_map()
                    )

                self._weak_learners.append(weak_learner)

                if verbose > 1:
                    print(f"\033[31mEras left: {self._n_eras - (era + 1)}\033[0m")

            return StrongLearner(weak_learners=self._weak_learners, device=self._device)

        return detached_start

    def start(self, verbose: int = 0) -> StrongLearner:
        start: Callable[[Tensor], StrongLearner] = self.start_generator(True, verbose)
        return start(self._weights)

    def get_weights(self) -> Tensor:
        return self._weights
