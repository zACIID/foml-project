from typing import Callable
from numpy import ndarray, array, append
from torch import tensor, Tensor, ones, zeros, sum
from src.classifiers.weak_learner import WeakLearner
from src.classifiers.strong_learner import StrongLearner


def getTheBestWeakLearner(weak_learners: ndarray[WeakLearner]) -> WeakLearner:
    min_err: float = weak_learners[0].getErrorRate()
    best_idx: int = 0
    for idx, weak_learner in enumerate(weak_learners):
        wl_error_rate: float = weak_learner.getErrorRate()
        if wl_error_rate < min_err:
            min_err = wl_error_rate
            best_idx = idx

    return weak_learners[best_idx]


"""
    Class wrapping all the functionalities needed to make a training algorithm based 
    on an ensemble approach
"""


class AdaBoost:
    def __init__(
            self, n_eras: int, dataset: list[Tensor], labels_set: Tensor,
            n_classes: int, img_size: tuple[int, int], weak_learner_epochs: int = 10,
            weak_learners_ratio: float = 0.05
    ):
        self._n_eras: int = n_eras
        self._dataset: list[Tensor] = dataset
        self._labels_set: Tensor = labels_set
        self._n_classes: int = n_classes
        self._img_size: tuple[int, int] = img_size
        self._weak_learner_epochs: int = weak_learner_epochs
        self._weak_learner_inst: int = int(self._dataset.size().item() * weak_learners_ratio)
        self._weak_learners: ndarray[WeakLearner] = array([])
        self._weights: Tensor = AdaBoost.initializeWeights(self._labels_set, self._n_classes)

    @staticmethod
    def initializeWeights(labels_set: Tensor, n_classes: int) -> Tensor:
        weights: Tensor = ones(labels_set.numel())

        for idx in range(n_classes):
            n_occ: int = sum(labels_set[labels_set == idx + 1]).item() / (idx + 1)
            weights[labels_set == idx + 1] = 1 / (2 * n_occ)

        return weights

    @staticmethod
    def normalizeWeights(weights: Tensor) -> None:
        weights /= sum(weights)

    @staticmethod
    def updateWeights(weights: Tensor, weak_learner_beta: float, weak_learner_weights_map: Tensor) -> None:
        weights[weak_learner_weights_map] *= weak_learner_beta

    def startGenerator(self, update_weights: bool = True, verbose: int = 0) -> Callable[[Tensor], StrongLearner]:

        def detachedStart(weights: Tensor) -> StrongLearner:
            for era in range(self._n_eras):
                # normalize the weights
                AdaBoost.normalizeWeights(weights)

                # find the weak learner with the lowest loss
                best_weak_learner: WeakLearner = self._getBestWeakLearner(verbose)

                if update_weights:
                    AdaBoost.updateWeights(weights, best_weak_learner.getBeta(), best_weak_learner.getWeightsMap())

                # store the best weak learner at the t-th era
                self._weak_learners = append(self._weak_learners, [best_weak_learner])

                if verbose > 1:
                    print(f"\033[31mTime left: {-1}\033[0m")

            return StrongLearner(self._weak_learners)

        return detachedStart

    def start(self, verbose: int = 0) -> StrongLearner:
        start: Callable[[Tensor], StrongLearner] = self.startGenerator(True, verbose)
        return start(self._weights)

    def _getBestWeakLearner(self, verbose: int = 0) -> WeakLearner:
        weakLearners: ndarray[WeakLearner] = array([])

        for idx in range(self._weak_learner_inst):
            if verbose > 1:
                print(
                    f"\033[33mWeak learner number: {idx}. \n\033[0m"
                    f"\033[33mWeak learner left: {self._weak_learner_inst - (idx + 1)}\033[0m"
                )

            weakLearner: WeakLearner = WeakLearner(
                self._dataset, self._labels_set, self._weights,
                self._weak_learner_epochs, verbose
            )

            weakLearners = append(weakLearners, [weakLearner])

        return getTheBestWeakLearner(weakLearners)

    def getWeights(self) -> Tensor:
        return self._weights
