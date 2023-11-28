import os
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from pipeline.dataset_providers.base import DatasetProvider, Dataset
from utils.constants import RND_SEED


class MNISTDigitsProvider(DatasetProvider):
    _OPENML_DATASET_NAME = "mnist_784"

    def __init__(
            self,
            storage_path: str | os.PathLike = None,
            data_fraction: float = 1.0,
            test_size: float = 0.25
    ):
        """
        :param storage_path: directory to store the dataset at.
            If left None, the dataset isn't stored after being fetched.
            Two csv files (X, y) are created for each dataset, data_fraction and test_size setting
        :param data_fraction: fraction of the dataset, in (0, 1], that should be kept.
            Useful if one wants to use less data.
        :param test_size: fraction of data, after data_fraction has been applied, to use as testing set
        """

        self.storage_path = storage_path

        assert 0 < data_fraction <= 1.0, "Fraction of the dataset to keep must be in the (0, 1] interval"
        self.data_fraction = data_fraction

        assert 0 <= test_size <= 1.0, "Fraction of the dataset to use as test must be in the (0, 1] interval"
        self.test_size = test_size

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

    def get_training_dataset(self) -> Dataset:
        assert self._train_dataset is not None, f"Must call fetch_dataset() before attempting to retrieve it"

        return self._train_dataset

    def get_testing_dataset(self) -> Dataset:
        assert self._test_dataset is not None, f"Must call fetch_dataset() before attempting to retrieve it"

        return self._test_dataset

    def fetch_datasets(self):
        """
        Fetch locally/remotely stored MNIST train and test dataset
        """

        # Try to fetch locally before downloading
        dataset = self._local_fetch_dataset()
        locally_fetched = True
        if dataset is None:
            dataset = self._remote_fetch_dataset()
            locally_fetched = False

        # Apply data_fraction only when fetched from remote the first time
        if not locally_fetched:
            # Use train_test_split to fraction the dataset in a stratified way
            removed_fraction = 1-self.data_fraction
            X_fraction, _, y_fraction, _ = train_test_split(
                dataset.X,
                dataset.y,

                # 1 as integer means just 1 sample, which is the minimum allowed by this method apparently
                test_size=removed_fraction if removed_fraction > 0 else 1,

                random_state=RND_SEED,
                stratify=dataset.y
            )
            dataset = Dataset(X=X_fraction, y=y_fraction)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=self.test_size,
            random_state=RND_SEED
        )

        self._train_dataset = Dataset(X=X_train, y=y_train, id=f"{self._OPENML_DATASET_NAME}_train")
        self._test_dataset = Dataset(X=X_test, y=y_test, id=f"{self._OPENML_DATASET_NAME}_test")

        # Call this after setting train and test dataset fields
        if not locally_fetched:
            self._store_dataset()

    def _local_fetch_dataset(self) -> Dataset:        # Try to fetch locally before downloading
        local_dataset = self._read_dataset()
        if local_dataset is not None:
            logger.info(f"Dataset `{self._OPENML_DATASET_NAME}` fetched from local storage at {self.storage_path}")

            return local_dataset

    def _remote_fetch_dataset(self) -> Dataset:

        # Downloading data
        dataset_name = self._OPENML_DATASET_NAME
        logger.info(f"Fetching dataset `{dataset_name}` remotely")

        X, y = fetch_openml(dataset_name, version=1, return_X_y=True)
        X: pd.DataFrame = X / 255.0  # scale grey color channel into [0, 1]
        y: pd.Series = y.astype(int)

        # Storing data
        dataset = Dataset(X=X.to_numpy(), y=y.to_numpy(), id=dataset_name)

        logger.info(f"Dataset `{dataset_name}` fetched")
        return dataset

    def _read_dataset(self, X_name: str = 'X', y_name: str = 'y') -> Optional[Dataset]:
        """
        Reads the dataset from the provided storage directory, only if the latter was provided

        :param X_name: name of feature file
        :param y_name: name of labels file
        """
        if self.storage_path is None:
            logger.debug("Storage path was not provided, dataset won't be read from local storage")
            return None

        X_file = self._get_X_file_path(X_name=X_name)
        y_file = self._get_y_file_path(y_name=y_name)

        # Dataset not previously stored, return
        if not os.path.exists(X_file) or not os.path.exists(y_file):
            return None

        logger.debug(f"Reading file at {X_file} ")
        X: pd.DataFrame = pd.read_csv(X_file)

        logger.debug(f"Reading file at {y_file} ")
        y: pd.DataFrame = pd.read_csv(y_file)
        y: pd.Series = y[y.columns[0]]  # df should be 1D, just one column

        return Dataset(X=X.to_numpy(), y=y.to_numpy(), id=self._OPENML_DATASET_NAME)

    def _store_dataset(self, X_name: str = 'X', y_name: str = 'y'):
        """
        Stores the dataset in the provided storage directory, only if the latter was provided

        :param X_name: name of feature file
        :param y_name: name of labels file
        """

        if self.storage_path is None:
            logger.debug("Storage path was not provided, dataset won't be stored")
            return

        X_file = self._get_X_file_path(X_name=X_name)
        y_file = self._get_y_file_path(y_name=y_name)

        # Reconstruct the full dataset by merging train + test
        logger.debug(f"Saving {X_file} ")
        X = np.concatenate((self._train_dataset.X, self._test_dataset.X))
        pd.DataFrame(X).to_csv(X_file, index=False)

        logger.debug(f"Saving {y_file} ")
        y = np.concatenate((self._train_dataset.y, self._test_dataset.y))
        pd.DataFrame(y).to_csv(y_file, index=False)

    def _get_X_file_path(self, X_name: str) -> str | os.PathLike:
        return os.path.join(self.storage_path, f"{X_name}_frac-{self.data_fraction}_test-{self.test_size}.csv")

    def _get_y_file_path(self, y_name: str) -> str | os.PathLike:
        return os.path.join(self.storage_path, f"{y_name}_frac-{self.data_fraction}_test-{self.test_size}.csv")
