from typing import Sequence
from torch import linalg as euclidean
import torch
import torch as th
from torch import Tensor

from classifiers.weak_learner import WeakLearner


class StrongLearner:
    def __init__(
            self,
            weak_learners: Sequence[WeakLearner],
            k_classes: int = 2,
            device: torch.device = None
    ):
        """
        :param weak_learners: collection of weak learners (ideally trained via AdaBoost)
        :param k_classes: number of classes to predict
        :param device: device to create tensors on; must be same device used for weak learners
        """
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._weak_learners: Sequence[WeakLearner] = weak_learners
        self._k_classes: int = k_classes

        self._alphas: Tensor = th.ones(len(self._weak_learners), dtype=th.float32)
        self._initialize_alphas()

    def predict(self, images: Tensor) -> Tensor:
        Pred: Tensor = self._get_preds(images)
        result_mat: Tensor = Pred @ self._alphas
        return th.argmax(result_mat, dim=0)

    def _get_betas(self) -> Tensor:
        betas: Tensor = th.tensor([], dtype=th.float32)
        for wk_l in self._weak_learners:
            betas = th.cat((betas, th.tensor([wk_l.get_beta()])))

        return betas.to(self._device)

    def _get_preds(self, samples: Tensor) -> Tensor:
        col_size: int = samples.shape[0] if samples.dim() > 3 else 1
        pred_mat: Tensor = th.zeros((
            len(self._weak_learners), col_size, self._k_classes
        ))

        for idx, wk_l in enumerate(self._weak_learners):
            pred_mat[idx] = wk_l.predict(samples)

        return th.transpose(pred_mat, dim0=0, dim1=2).to(self._device)

    def _initialize_alphas(self) -> None:
        betas: Tensor = self._get_betas()
        self.alphas = th.log(1 / betas)
        self.alphas = self.alphas / euclidean.norm(self.alphas, dim=0)

    def get_weak_learners(self) -> Sequence[WeakLearner]:
        return self._weak_learners
