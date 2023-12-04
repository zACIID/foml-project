from torch import Tensor, tensor
from torch.utils.data import Dataset

import src.utils.type_utility as th


class EnsembleDataset(Dataset):

    def __init__(self, data: list[Tensor], labels: Tensor, weights: Tensor):
        self._x_train: list[Tensor] = data
        self._y_train: Tensor = labels
        self._weights: Tensor = weights
        self._ids: Tensor = tensor([idx for idx in range(len(data))])

    def __len__(self) -> int:
        return len(self._x_train)

    def __getitem__(self, idx: int) -> th.EnsembleBatch:
        self._x_train[idx].requires_grad_()
        self._y_train[idx].requires_grad_()
        self._weights[idx].requires_grad_()
        return (self._x_train[idx], self._ids[idx]), (self._y_train[idx], self._weights[idx])
