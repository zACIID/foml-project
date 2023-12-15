import abc
import enum
from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt

from pipeline.dataset_providers.base import Dataset


class AnalysisPlotTypes(enum.Enum):
    FEATURE_DISTRIBUTIONS = 0
    BIVARIATE_PLOTS = 1


@dataclass(frozen=True)
class DatasetAnalysis:
    dataset: Dataset
    plots: Mapping[AnalysisPlotTypes, plt.Figure]


class DatasetAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, dataset: Dataset) -> DatasetAnalysis:
        pass
