import numpy as np
from loguru import logger
from sklearn.cluster import SpectralClustering

import utils.time_utils
from classifiers.base import BaseClassifier
from utils.constants import RND_SEED


class NCut(BaseClassifier):
    def __init__(self, n_components: int, rnd_state: int = RND_SEED):
        self.n_mixtures = n_components
        self.rnd_state = rnd_state

        self._X: np.ndarray = None

        self._is_fitted: bool = False
        self.inner_model = SpectralClustering(
            n_clusters=n_components,
            affinity="nearest_neighbors",
            random_state=rnd_state,
            assign_labels="kmeans",
            eigen_solver="arpack",
            n_jobs=-1
        )

    @property
    def X_(self) -> np.ndarray:
        return self._X

    @property
    def y_(self) -> np.ndarray:
        """
        Doesn't return anything because this is an unsupervised mode.
        Present for consistency.
        """
        return None

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    @utils.time_utils.print_time_perf
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> BaseClassifier:
        """
        :param X: training samples
        :param y: not used because model is unsupervised, present for api consistency
        :return:
        """
        logger.debug(f"Training {self.__class__.__name__}...")

        self.inner_model.fit(X)
        self._is_fitted = True

        return self

    @utils.time_utils.print_time_perf
    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.debug(f"Making predictions with {self.__class__.__name__}...")

        return self.inner_model.fit_predict(X)

    def get_params(self, deep=True):
        # Return inner gmm params + current BaseEstimator params
        #   (extracted from __init__ signature)
        return self.inner_model.get_params().update(super().get_params())

