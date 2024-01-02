import dataclasses
from typing import List


@dataclasses.dataclass
class TrainingResults:
    """
    Collection of train results.
    Each item is a list of values, one for each epoch
    """
    train_loss: List[float] = dataclasses.field(default_factory=lambda: [])


# TODO(pierluigi): not sure if this will actually be useful
@dataclasses.dataclass
class TrainValidationResults:
    """
    Collection of train and validation results.
    Each item is a list of values, one for each epoch
    """

    train_loss: List[float]
    validation_loss: List[float]
    train_accuracy: List[float]
    validation_accuracy: List[float]
