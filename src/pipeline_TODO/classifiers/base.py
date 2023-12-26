from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    @property
    @abstractmethod
    def y_(self) -> np.ndarray | None:
        """Labels used to train the model"""
        return

    @property
    @abstractmethod
    def X_(self) -> np.ndarray | None:
        """Samples used to train the model"""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        pass
