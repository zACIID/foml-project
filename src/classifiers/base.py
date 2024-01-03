import dataclasses
from typing import List


@dataclasses.dataclass
class TrainingResults:
    """
    Collection of train results.
    Each item is a list of values, one for each epoch
    """
    avg_train_loss: List[float] = dataclasses.field(default_factory=lambda: [])


# TODO(pierluigi): not sure if this will actually be useful
@dataclasses.dataclass
class TrainingValidationResults(TrainingResults):
    """
    Collection of train and validation results.
    Each item is a list of values, one for each epoch
    """

    avg_validation_loss: List[float] = dataclasses.field(default_factory=lambda: [])
    train_accuracy: List[float] = dataclasses.field(default_factory=lambda: [])
    validation_accuracy: List[float] = dataclasses.field(default_factory=lambda: [])
