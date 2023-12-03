from torch.utils.data import Dataset
from torch import Tensor, tensor, float32, int32
from numpy import ndarray, array
import src.utils.type_utility as th


class EnsambleDataset(Dataset):

    def __init__(self, data: list[Tensor], labels: Tensor, weights: Tensor):
        self._x_train: list[Tensor] = data
        self._y_train: Tensor = labels
        self._weights: Tensor = weights
        self._ids: Tensor = tensor([idx for idx in range(len(data))])

    def __len__(self) -> int:
        return len(self._x_train)

    def __getitem__(self, idx: int) -> th.ensamble_batch:
        self._x_train[idx].requires_grad_()
        self._y_train[idx].requires_grad_()
        self._weights[idx].requires_grad_()
        return (self._x_train[idx], self._ids[idx]), (self._y_train[idx], self._weights[idx])
