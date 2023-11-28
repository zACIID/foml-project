from typing import Tuple, Optional

import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

import src.visualization.feature_engineering as vis
from src.pipeline.feature_engineering.base import FeatureEngineering, Dataset


class PCAEngineering(FeatureEngineering):
    def __init__(
            self,
            n_components: int,
            standardize_data: bool = True
    ):
        """
        :param n_components: number of components to keep after applying PCA
        :param standardize_data: if True, data is standardized before being fed to PCA
        """

        super().__init__()

        assert n_components > 0, "Provide a positive number of components"

        self.n_components: int = n_components

        self.pca: Optional[PCA] = None
        self.standardize_data: bool = standardize_data

    def engineer(self, train: Dataset, test: Dataset) -> Tuple[Dataset, Dataset]:
        self.pca = PCA(n_components=self.n_components)

        check_X_y(train.X, train.y)
        check_X_y(test.X, test.y)

        if self.standardize_data:
            scaler = StandardScaler()
            scaler.fit(train.X)
            train = Dataset(X=scaler.transform(train.X), y=train.y, id=train.id)
            test = Dataset(X=scaler.transform(test.X), y=test.y, id=train.id)

        # TODO(pierluigi): fortran-ordered or use np.ascontiguous array?
        self.pca.fit(train.X)
        train_pca = Dataset(X=self.pca.transform(train.X), y=train.y, id=f"{train.id}_PCA-{self.n_components}")
        test_pca = Dataset(X=self.pca.transform(test.X), y=test.y, id=f"{test.id}_PCA-{self.n_components}")

        logger.debug(f"Applied PCA {'(and StandardScaler)' if self.standardize_data else ''}"
                     f" with n_components={self.n_components} to train and test datasets")

        return train_pca, test_pca

    def plot_pca_explained_variances(
            self,
            figsize: Tuple[float, float] = (8, 5)
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = vis.pca_variances_chart(pca=self.pca, figsize=figsize)

        return fig, ax
