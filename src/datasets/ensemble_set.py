import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset


class EnsembleDataset(Dataset):

    def __init__(self, data: list[Tensor], labels: Tensor, weights: Tensor):
        super().__init__()
        self._x_train: list[Tensor] = data
        self._y_train: Tensor = tensor([
            (1, 0) if elem != 2 else (0, 1) for elem in labels
        ], dtype=torch.float32)
        self._weights: Tensor = weights
        self._ids: Tensor = tensor([idx for idx in range(len(data))])

    def __len__(self) -> int:
        return len(self._x_train)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._x_train[idx].requires_grad_()
        self._y_train[idx].requires_grad_()
        self._weights[idx].requires_grad_()

        return (
            self._ids[idx], self._x_train[idx],
            self._y_train[idx], self._weights[idx]
        )
