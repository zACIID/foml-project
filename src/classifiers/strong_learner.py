from typing import Sequence

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

    def predict_image(self, img: Tensor) -> Tensor:
        final_distribution: Tensor = th.zeros(self._k_classes, device=self._device)
        for weak_learner, alpha in zip(self._weak_learners, self._alphas):
            weak_learner: WeakLearner
            alpha: Tensor = alpha.to(self._device)

            wk_pred: Tensor = th.squeeze(weak_learner.predict(img), 1)
            final_distribution += alpha * wk_pred

        # Get class with the highest probability
        # Need to re-normalize to 1 after multiplying by the alpha weights
        return th.argmax(final_distribution / th.euclidean.norm(final_distribution, 1))

    def predict_images(self, images: Tensor) -> Tensor:
        preds: Tensor = th.tensor([], dtype=th.int32, device=self._device)

        images = images.cpu()
        for img in images:
            # Note(pierluigi): unsqueeze is done so that cat can work
            pred: Tensor = th.unsqueeze(self.predict_image(img), dim=0)
            preds = th.cat((preds, pred))

        return preds

    def _initialize_alphas(self) -> None:
        for idx, weak_learner in enumerate(self._weak_learners):
            # TODO(pierluigi): si sbrega qui se tensore e in gpu e uso math.log? Oppure si sbrega cosi?
            self._alphas[idx] = torch.log(1 / weak_learner.get_beta())

    def get_weak_learners(self) -> Sequence[WeakLearner]:
        return self._weak_learners

