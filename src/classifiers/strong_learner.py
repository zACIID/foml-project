import math
from typing import Sequence

import torch as th
from torch import Tensor

from classifiers.weak_learner import WeakLearner


class StrongLearner(object):
    def __init__(self, weak_learners: Sequence[WeakLearner], k_classes: int = 2):
        self._weak_learners: Sequence[WeakLearner] = weak_learners
        self._k_classes: int = k_classes

        self._alphas: Tensor = th.ones(len(self._weak_learners), dtype=th.float32)
        self._initialize_alphas()

    def predict_image(self, img: Tensor) -> Tensor:
        final_dist: Tensor = th.zeros(self._k_classes)
        for weak_learner, alpha in zip(self._weak_learners, self._alphas):
            weak_learner: WeakLearner
            wk_pred: Tensor = th.squeeze(weak_learner.predict(img), 1)
            final_dist += alpha * wk_pred

        return th.argmax(final_dist / th.euclidean.norm(final_dist, 1))

    def predict_images(self, images: Tensor) -> Tensor:
        preds: Tensor = th.tensor([], dtype=th.int32)

        for img in images:
            pred: Tensor = th.unsqueeze(self.predict_image(img), dim=0)
            preds = th.cat((preds, pred))

        return preds

    def _initialize_alphas(self) -> None:
        for idx, weak_learner in enumerate(self._weak_learners):
            self._alphas[idx] = math.log(1 / weak_learner.get_beta())

    def get_weak_learners(self) -> Sequence[WeakLearner]:
        return self._weak_learners

