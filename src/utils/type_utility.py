from typing import NewType
from numpy import ndarray

image = NewType('image', ndarray[ndarray[float]])
ensamble_batch = NewType('ensamble batch', tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]])
