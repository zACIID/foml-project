from typing import Tuple

from abc import ABC, abstractmethod

from pipeline.dataset_providers.base import Dataset


class FeatureEngineering(ABC):
    @abstractmethod
    def engineer(self, train: Dataset, test: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Applies the feature engineering to the provided train and test datasets,
            and returns them

        :param train:
        :param test:
        :return: tuple of engineered datasets (train, test)
        """
        pass
