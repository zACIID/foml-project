import dataclasses
from typing import List, Sequence, Tuple, Callable, Iterator

import torch
from torch import Tensor, sum
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers.simple_learner import WeakLearnerValidationResults, WeakLearnerTrainingResults
from loss_functions.base_weighted_loss import WeightedBaseLoss
from src.classifiers.strong_learner import StrongLearner
from src.classifiers.weak_learner import WeakLearner
from src.datasets.custom_coco_dataset import ItemType, BatchType


@dataclasses.dataclass
class StrongLearnerTrainingResults:
    weak_learner_results: List[WeakLearnerTrainingResults] | List[WeakLearnerValidationResults] = dataclasses.field(default_factory=lambda: [])
    train_accuracy: List[float] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class StrongLearnerValidationResults(StrongLearnerTrainingResults):
    val_accuracy: List[float] = dataclasses.field(default_factory=lambda: [])


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
            actual_train_dataset_length: int,
            weak_learner_optimizer: torch.optim.Optimizer,
            weak_learner_loss: WeightedBaseLoss = None,
            weak_learner_epochs: int = 5,
            verbose: int = 0,
    ) -> Tuple[StrongLearner, StrongLearnerTrainingResults]:
        self._weights = _initialize_weights(
            actual_train_dataset_length=actual_train_dataset_length,
            device=self._device
        )

        strong_learner_results = StrongLearnerTrainingResults()
        for era in range(eras):
            weak_learner: WeakLearner = WeakLearner(
                weights=self._weights,
                device=self._device
            )

            res = weak_learner.fit(
                data_loader=data_loader,
                classes_mask=classes_mask,
                optimizer=weak_learner_optimizer,
                loss=weak_learner_loss,
                epochs=weak_learner_epochs,
                verbose=verbose
            )

            _update_weights_(
                weights=self._weights,
                weak_learner_beta=weak_learner.get_beta(),
                weak_learner_weights_map=weak_learner.get_weights_map()
            )

            self._weak_learners.append(weak_learner)
            strong_learner_results.weak_learner_results.append(res)

            strong_learner = StrongLearner(weak_learners=self._weak_learners, device=self._device)
            accuracy = _test_results(data_loader=data_loader, model=strong_learner)
            strong_learner_results.train_accuracy.append(accuracy)

            if verbose > 1:
                print(f"StrongLearner accuracy: {accuracy}")
                print(f"Eras left: {eras - (era + 1)}")

        return StrongLearner(weak_learners=self._weak_learners, device=self._device), strong_learner_results

    def fit_and_validate(
            self,
            eras: int,
            train_data_loader: DataLoader[ItemType],
            validation_data_loader: DataLoader[ItemType],
            classes_mask: Tensor,
            actual_train_dataset_length: int,
            weak_learner_optimizer_builder: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
            weak_learner_loss: WeightedBaseLoss = None,
            weak_learner_epochs: int = 5,
            verbose: int = 0,
    ) -> Tuple[StrongLearner, StrongLearnerValidationResults]:
        self._weights = _initialize_weights(
            actual_train_dataset_length=actual_train_dataset_length,
            device=self._device
        )

        strong_learner_results = StrongLearnerValidationResults()
        for era in range(eras):
            weak_learner: WeakLearner = WeakLearner(
                weights=self._weights,
                device=self._device
            )

            res = weak_learner.fit_and_validate(
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
                classes_mask=classes_mask,
                optimizer_builder=weak_learner_optimizer_builder,
                loss=weak_learner_loss,
                epochs=weak_learner_epochs,
                verbose=verbose
            )

            _update_weights_(
                weights=self._weights,
                weak_learner_beta=weak_learner.get_beta(),
                weak_learner_weights_map=weak_learner.get_weights_map()
            )

            self._weak_learners.append(weak_learner)
            strong_learner_results.weak_learner_results.append(res)

            strong_learner = StrongLearner(weak_learners=self._weak_learners, device=self._device)
            train_accuracy = _test_results(data_loader=train_data_loader, model=strong_learner)
            val_accuracy = _test_results(data_loader=validation_data_loader, model=strong_learner)
            strong_learner_results.train_accuracy.append(train_accuracy)
            strong_learner_results.val_accuracy.append(val_accuracy)

            if verbose > 1:
                print(f"StrongLearner train accuracy: {train_accuracy}")
                print(f"StrongLearner validation accuracy: {val_accuracy}")
                print(f"Eras left: {eras - (era + 1)}")

        return StrongLearner(weak_learners=self._weak_learners, device=self._device), strong_learner_results

    def get_weights(self) -> Tensor:
        return self._weights


def _initialize_weights(
        actual_train_dataset_length: int,
        device: torch.device
) -> Tensor:
    # NOTE: since the dataset used may be a torch.utils.data.Subset,
    #   we have a mismatch between the indices of the samples in the dataset and
    #   the indices of the samples in [0, dataset_length]
    #  For example, Subset may use indices [0, 8, 12], hence a dataset of length 3,
    #   but the whole dataset would be of length n.
    #  What we must do therefore is to initialize the weights vector with the total
    #   length of the dataset, so that the indices returned by the dataloader can be used.
    #  Even if this results in space being wasted, that's fine
    #   (unless the vector becomes so big it is unusable)
    weights = torch.ones(actual_train_dataset_length, device=device)
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

def _test_results(
        data_loader: torch.utils.data.DataLoader[ItemType],
        model: StrongLearner
):
    accuracy = .0

    for batch in tqdm(data_loader):
        batch: BatchType
        _, x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        predictions: torch.Tensor = model(x_batch)

        # noinspection PyUnresolvedReferences
        accuracy += (predictions == y_batch).sum().item() / len(data_loader.dataset)

    return accuracy
