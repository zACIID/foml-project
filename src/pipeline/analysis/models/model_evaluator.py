from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, Optional, Mapping

import matplotlib.pyplot as plt
import numpy as np

from classifiers.base import BaseClassifier
from pipeline.model_builder import ModelData


class ScoreTypes(Enum):
    RAND_INDEX = 0
    # TODO silhouette score?


class PlotType(Enum):
    MEAN_CLASSIFICATIONS_PLOT = 0
    LABEL_COUNTS = 1
    CONFUSION_MATRIX = 2
    INTERNAL_REPRESENTATION = 3

    # TODO silhouette plot?


T = TypeVar("T", bound=BaseClassifier)


@dataclass(frozen=True)
class ModelEvaluation(Generic[T]):
    model_data: ModelData[T]
    predictions: np.ndarray
    scores: Mapping[ScoreTypes, float]
    plots: Mapping[PlotType, plt.Figure]


class ModelEvaluator(Generic[T], ABC):
    """Class that handles the logic behind the evaluation of a model of type T"""

    def __init__(self):
        pass

    # TODO I think this is the correct place to pass the original training and testing datasets
    #   add two optional params both here and tw optional fields inside model evaluation
    #   The idea is that it is the evaluator that might need the original dataset for stats/plots,
    #       not the model builder whose only responsibility should be training the model

    @abstractmethod
    def evaluate(self, model_data: ModelData[T]) -> Optional[ModelEvaluation]:
        """
        Evaluate the provided model.
        The result is optional because models that are not of type T (should) be discarded.
        """
        pass
