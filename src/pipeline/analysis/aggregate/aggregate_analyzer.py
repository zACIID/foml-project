import abc
import enum
from dataclasses import dataclass
from typing import Sequence, Generic, TypeVar, Mapping, Type

import matplotlib.pyplot as plt

from classifiers.base import BaseClassifier
from pipeline.analysis.models.model_evaluator import ModelEvaluation

P = TypeVar("P", bound=enum.Enum)


@dataclass(frozen=True)
class AggregateAnalysis(Generic[P]):
    plots: Mapping[P, plt.Figure]
    best_models: Mapping[Type[BaseClassifier], ModelEvaluation]
    worst_models: Mapping[Type[BaseClassifier], ModelEvaluation]


class AggregateAnalyzer(Generic[P], abc.ABC):
    @abc.abstractmethod
    def analyze(self, evaluations: Sequence[ModelEvaluation]) -> AggregateAnalysis[P]:
        pass
