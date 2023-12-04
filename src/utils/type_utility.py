from numpy import ndarray
from torch import Tensor

Image = ndarray[ndarray[float]]
EnsembleBatch = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
