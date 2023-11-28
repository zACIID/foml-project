import time
from dataclasses import dataclass
from typing import Iterable, TypeVar, Generic, Optional

from classifiers.base import BaseClassifier
from .dataset_providers.base import Dataset

T = TypeVar("T", bound=BaseClassifier)


@dataclass
class ModelData(Generic[T]):
    training_dataset: Dataset
    """Original, non-engineered training dataset"""

    testing_dataset: Dataset
    """Original, non-engineered testing dataset"""

    model: T

    training_time: float
    """Time that it took to train the model, in seconds"""

    training_dataset_engineered: Optional[Dataset] = None
    testing_dataset_engineered: Optional[Dataset] = None


class ModelBuilder:
    """
    Class that handles the training of the provided model
    """

    def __init__(self, untrained_models: Iterable[BaseClassifier]):
        self.models: Iterable[BaseClassifier] = untrained_models

    # TODO see TODO in model_evaluator.py -> that's the place to provide original datasets
    #   here just keep training_dataset and testing_dataset, assuming that those are the ones used
    #   to create the model
    def build(self, training_dataset: Dataset, testing_dataset: Dataset) -> Iterable[ModelData]:
        for m in self.models:
            t1 = time.perf_counter()
            m.fit(training_dataset.X, training_dataset.y)
            t2 = time.perf_counter()

            yield ModelData(
                model=m,
                training_dataset=training_dataset,
                testing_dataset=testing_dataset,
                training_time=t2 - t1,
            )
