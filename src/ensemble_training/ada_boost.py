from typing import Callable

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
            n_eras: int,
            dataset: CocoDataset,
            n_classes: int,
            weak_learner_epochs: int = 10,
    ):
        self._n_eras: int = n_eras
        self._dataset: CocoDataset = dataset
        self._n_classes: int = n_classes
        self._weak_learner_epochs: int = weak_learner_epochs
        self._weak_learners: Tensor = th.tensor([])
        self._weights: Tensor = self._initialize_weights()

    def _initialize_weights(self) -> Tensor:
        classes_mask: Tensor = self._dataset.get_labels()
        weights: Tensor = th.zeros(classes_mask.shape[0])

        for lbl in Labels:
            card: int = self._dataset.get_class_cardinality(label=lbl)
            weights[classes_mask == int(lbl)] = 1 / (2 * card)

        return weights

    @staticmethod
    def normalize_weights(weights: Tensor) -> None:
        weights /= sum(weights)

    @staticmethod
    def update_weights(weights: Tensor, weak_learner_beta: float,
                       weak_learner_weights_map: Tensor) -> None:

        weights[weak_learner_weights_map] *= weak_learner_beta

    def start_generator(
            self,
            update_weights: bool = True,
            verbose: int = 0
    ) -> Callable[[Tensor], StrongLearner]:

        def detached_start(weights: Tensor) -> StrongLearner:

            for era in range(self._n_eras):
                AdaBoost.normalize_weights(weights)

                weak_learner: WeakLearner = WeakLearner(
                    self._dataset, self._weights,
                    self._weak_learner_epochs, verbose
                )

                if update_weights:
                    AdaBoost.update_weights(
                        weights, weak_learner.get_beta(),
                        weak_learner.get_weights_map()
                    )

                self._weak_learners = th.cat((
                    self._weak_learners,
                    th.tensor([weak_learner])
                ))

                if verbose > 1:
                    print(f"\033[31mEras left: {self._n_eras - (era + 1)}\033[0m")

            return StrongLearner(self._weak_learners)

        return detached_start

    def start(self, verbose: int = 0) -> StrongLearner:
        start: Callable[[Tensor], StrongLearner] = self.start_generator(True, verbose)
        return start(self._weights)

    def get_weights(self) -> Tensor:
        return self._weights
