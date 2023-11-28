from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray

    id: Optional[str] = None
    """Optional string to identify the dataset"""


class DatasetProvider(abc.ABC):
    @abc.abstractmethod
    def get_training_dataset(self) -> Dataset:
        pass

    @abc.abstractmethod
    def get_testing_dataset(self) -> Dataset:
        pass
