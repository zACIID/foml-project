from typing import NewType
from numpy import ndarray

image = NewType('image', ndarray[ndarray[float]])
