# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="NoaC2o-Nt6bS"
# # Weak learners based on squueze net

# %% [markdown] id="DwLNRccmgQpk"
# ## Fire layer implementation

# %% [markdown] id="2H3pVZnX0vc6"
#

# %% id="45pgNRGo-Rld"
from torch import Tensor, cat
import torch.nn as nn
from typing import Union

class FireLayer(nn.Module):
  def __init__(self, in_channels_sqz: int, out_channels_sqz: int,
               out_channels_exp_ones: int, out_channels_exp_threes: int,
               act_fun: nn.Module, n_dims: int = 3, device: str = None
               ):

    super().__init__()

    self._squueze_layer: nn.Conv2d = nn.Conv2d(
        in_channels=in_channels_sqz, out_channels=out_channels_sqz,
        kernel_size=1, device=device
    )
    self._expand_layer_ones: nn.Conv2d  = nn.Conv2d(
        in_channels=out_channels_sqz, out_channels=out_channels_exp_ones,
        kernel_size=1, device=device
    )
    self._expand_layer_threes: nn.Conv2d  = nn.Conv2d(
        in_channels=out_channels_sqz, out_channels=out_channels_exp_threes,
        padding=1, kernel_size=3, device=device
    )

    self._n_dims: int = n_dims
    self._act_fun: nn.Module = act_fun()

  def forward(self, ft_map: Tensor) -> Tensor:
    # apply squeeze layer and non linearity
    sqz_ft_map: Tensor = self._act_fun(self._squueze_layer(ft_map))

    # apply expand layer
    exp_ones_ft_map: Tensor = self._expand_layer_ones(sqz_ft_map)
    exp_threes_ft_map: Tensor = self._expand_layer_threes(sqz_ft_map)

    # choose on which dimension to concat
    dim: int = int(self._n_dims != exp_ones_ft_map.dim())

    # concatenate the 1x1 and 3x3 filters features maps
    return cat((exp_ones_ft_map, exp_threes_ft_map), dim=dim)


# %% [markdown] id="VM27KKMPWHxh"
# #### Test

# %% colab={"base_uri": "https://localhost:8080/"} id="TVxSCM0ehO2R" outputId="542290e2-ac30-4d3b-e93b-8afcc7f79da3"
import torch as th

img = th.rand((65, 55, 55))

fire = FireLayer(in_channels_sqz=65, out_channels_sqz=32, out_channels_exp_ones=79,
                 out_channels_exp_threes=79, act_fun=nn.SiLU)

fire(img).shape

# %% [markdown] id="f8Zj7nIlIVSn"
# ## Temporary dataset implementation

# %% id="pvcUZ5LGIbx7"
from torch.utils.data import Dataset, DataLoader

class EnsembleDataset(Dataset):

    def __init__(self, data: list[Tensor], labels: Tensor, weights: Tensor):
        super().__init__()
        self._x_train: list[Tensor] = data
        self._y_train: Tensor = th.tensor([
            (1, 0) if elem != 2 else (0, 1) for elem in labels
        ], dtype=th.float32)
        self._weights: Tensor = weights
        self._ids: Tensor = th.tensor([idx for idx in range(len(data))])

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



# %% [markdown] id="u2cu9hQfx6Or"
# ## Loss Implementation

# %% [markdown] id="k2gDNT5cyPV4"
# #### Base class

# %% id="5zZ27gwvx5ob"
from typing import Callable
import torch.nn as nn
from numpy import ndarray, array, append
from torch import Tensor


class WeightedBaseLoss(nn.Module):
    def __init__(self, sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor]):
        super().__init__()
        self._pred: Tensor = th.tensor([])
        self._ids: Tensor = th.tensor([])
        self._sub_loss: Callable[[Tensor, Tensor, Tensor], Tensor] = sub_loss

    def forward(self, y_true: Tensor, y_pred: Tensor, weights: Tensor,
                ids: Tensor, save: bool = False) -> Tensor:

        if save:
            self._pred = th.cat((self._pred, y_pred))
            self._ids = th.cat((self._ids, ids))

        return self._sub_loss(y_true, y_pred, weights)

    def get_error_map(self) -> tuple[Tensor, Tensor]:
        return self._pred, self._ids




# %% [markdown] id="DpJbdaiIyT5q"
# #### Weighted cross entropy

# %% id="ga5jU_OnyXPM"
class WeightedCrossEntropy(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            weighted_y_true: Tensor = y_true * (weights.unsqueeze(dim=1))
            return cross_entropy(weighted_y_true, y_pred)

        super().__init__(sub_loss)



# %% [markdown] id="Itazbsygz1aW"
# #### Weighted $norm_2$ distance
#
#
#

# %% id="7jDXCyI80Bc9"
from torch import linalg as euclidean

class WeightedDistance(WeightedBaseLoss):
    def __init__(self):

        def sub_loss(y_true: Tensor, y_pred: Tensor, weights: Tensor) -> Tensor:
            distances_tensor: Tensor = y_true - y_pred
            euclidean_distances: Tensor = euclidean.norm(distances_tensor, dim=1)
            return th.inner(weights, euclidean_distances)

        super().__init__(sub_loss)


# %% [markdown] id="ZCvHWoFxgePk"
# ## SimpleLearner

# %% id="5rZqwW05rUxX"
from collections import OrderedDict
from torch.optim import Adam

class SimpleLearner(nn.Module):
  def __init__(self, k_classes: int = 2, act_fun: nn.Module = nn.SiLU, device: str = None):
    super().__init__()
    self._pipe_line: nn.Sequential = nn.Sequential(
        OrderedDict([
          ('conv1', nn.Conv2d(in_channels=3, out_channels=96,
                              kernel_size=7, stride=2, device=device)),
          ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
          ('fire1', FireLayer(in_channels_sqz=96, out_channels_sqz=32,
                              out_channels_exp_ones=64,
                              out_channels_exp_threes=64, act_fun=act_fun,
                              device=device)),
          ('act_fun1', act_fun()),
          ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
          ('fire2', FireLayer(in_channels_sqz=128, out_channels_sqz=64,
                              out_channels_exp_ones=256,
                              out_channels_exp_threes=256, act_fun=act_fun,
                              device=device)),
          ('act_fun2', act_fun()),
          ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2)),
          ('conv2', nn.Conv2d(in_channels=512, out_channels=k_classes,
                              kernel_size=1, device=device)),
          ('averagepool', nn.AvgPool2d(kernel_size=12, stride=1)),
          ('flatten', nn.Flatten()),
          ('softmax', nn.Softmax(dim=0))
        ])
    )

  def forward(self, batch: Tensor) -> Tensor:
    return self._pipe_line(batch)

  def fit(self, dataset: Dataset, adaboost_wgt: Tensor, batch_size: int = 32,
          epochs: int = 10, loss: WeightedBaseLoss = WeightedCrossEntropy,
          verbose: int = 0) -> tuple[tuple[Tensor, Tensor], float]:

    adam_opt: Adam = Adam(self.parameters())
    wgtd_loss: WeightedBaseLoss = loss()
    cum_loss: float = .0

    self.train()
    for epoch in range(epochs):
      cum_loss = .0
      for batch in DataLoader(dataset, batch_size, shuffle=True):

        ids, x_batch, y_batch, wgt_batch = batch
        y_pred: Tensor = self(x_batch)
        mixed_weights: Tensor = adaboost_wgt[ids] * wgt_batch

        batch_loss: Tensor = wgtd_loss(
            y_true=y_batch, y_pred=y_pred, weights=mixed_weights,
            ids=ids, save=True if epoch == epochs - 1 else False
        )
        cum_loss += batch_loss.item()

        adam_opt.zero_grad()  # initialize gradient to zero
        batch_loss.backward()  # compute gradient
        adam_opt.step()  # backpropagation

      if verbose >= 1:
        print(f"\033[32mEpoch:{epoch} loss is {cum_loss}\033[0m")

    return wgtd_loss.get_error_map(), cum_loss


  def predict(self, samples: Tensor) -> Tensor:
    with th.no_grad():
      self.eval()  # Set the model to evaluation mode (if applicable)
      return self(samples)

  def get_modules(self) -> nn.Sequential:
    return self._pipe_line

  def set_modules(self, new_modules: nn.Sequential) -> None:
    self._pipe_line = new_modules

# %% [markdown] id="0Ewe5rYVy2UI"
# #### Tests

# %% id="a-wY-BOUyzDX" colab={"base_uri": "https://localhost:8080/"} outputId="477e80a8-efab-470d-f0cd-5c4a8dad5a87"
images = []

for idx in range(1000):
  images.append(th.rand(3, 214, 214))

dataset = EnsembleDataset(images, th.tensor(
    [0 if idx % 2 else 1 for idx in range(1000)]
), th.ones(1000))

model = SimpleLearner()

training_result = model.fit(dataset, epochs=3, verbose=1, adaboost_wgt=th.ones(1000))

# %% id="obU9FzcNWlxK" colab={"base_uri": "https://localhost:8080/"} outputId="a562ec5e-ccf0-4fc2-acbd-a4040209cd80"
model.predict(images[474])


# %% [markdown] id="ySIOxNqNYwqZ"
# ## SimpleLearner wrapper

# %% id="T6ctpOgGEA3Z"
import enum

class Labels(enum.IntEnum):
    PERSON = 0
    OTHER = 1


# %% [markdown] id="saRRFLTRmNA6"
# **Big ass problem:         (UPDATE: SOLVED CHOSE SIGMOID)**
#
# A weak learner sets its $\beta$ to $\frac{\epsilon}
# {1-\epsilon }$. Where $\epsilon$ is the error computed throughout the training phase. Given our `WeightedCrossEntropy` the return value is a value $\geq1$ . So the $\beta$ either becomes $\inf$ or a negative value. We need to remind ourseleves that each adaboost weight is updated as:   
#
# *   $w_{t+1, i}=w_{t, i}\cdot \beta ^{1-e_i}$
# *   $e_i=0$ if the i-th sample was classified correctly 1 otherwise
#
# So here we would not only makes a weight of a sample (wich was classified correctly) negative but we would reduce the loss every time a weak learner classify a sample correctly.
# The reason the creator of adaboost normalize the weights at each iteration is beacuse their loss function compute the distance between the true label and the prediction such as:
#
# *   $\epsilon=\sum_{i}{w_i\cdot|pred(x_i)-label(x_i)|}$.
#
# So in their case given the fact that the weights are normalized the error can at most be 1 (if all the sample in the dataset are uncorrectly classified).
#
# Thus this imply that in their case $\epsilon_i\in [0, 1]\text{ }\forall i$. This means that in order to have $\beta< 1$, we need:
#
# *   $\frac{\epsilon}{1-\epsilon }<1\rightarrow \epsilon<1-\epsilon\rightarrow \epsilon < \frac{1}{2}$
#
# This because when viola and jones made this project they started with the assumption that a weak learner would just do better than a chance ($\epsilon < 0.5$).
# The assumption behind how $\beta$ is that, the greater the error the grater the value of $\beta$. This implies that when a model fuck up a lot, the amount of changing that it will apply on the correctly classfied samples is less than another model with lower $\epsilon$. This with a possible assumption that if a model is not precise enough then we have no insurace that the samples classified correctly by it will be classfied (w.h.p) correctly again in the future.
#
# The reason why i'm concerned about the value of $\beta$ is beacuse if $\beta\geq 1$ then the weights related to the samples clasified correctly are going to be incremented and not decremented. This would fuck up all the assumption behind learning from the mistakes of the past learners.
#
# **Possible solutions**:
#
# 1. Normalize the `cum_loss` inside the `fit` method of a `SimpleLearner` with the sum of all the `cum_loss` for each epoch. The cons of this approach is that in presence of just one epoch this would not work (i.e $\epsilon=1$ thus $\beta= \inf$)
#
# 2.  Given the fact that the loss is always > 1 we could set $\beta=\frac{\epsilon}{\omega -\epsilon }$ where $\omega \in (1, 100] $ and yes $\omega$ unluckly must be computed given the datset used.
#
# 3. Another possibility is that we set $\beta=\frac{1}{1+e^{-\epsilon}}=\text{sigmoid}(\epsilon)$ this would makes $0<\beta<1$
#
# Furthermore option 3) would have the same assumption of how the normal $\beta$: `the greater the error the grater the value of`$\beta$
#
# Further analisys:  we need also to remember that the final classification (`StrongLearner`) is performed as:
# ![image.png](data:image/png;base64,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)
#
# So we need to be aware of $\alpha$ this is the reason why I did not propose to set $\beta=\frac{1}{\epsilon}$.
#
# **Conclusions**:
#
# To me either soulution 1 and 3 are the best, 3 is more robust and would works anyway whereas 1 would not work with just one epoch. But if we chose to use 3) we would still like to normalize the weights because otherwise the other weights which was never or rarely updated would be be too impactful on the loss. Lastly if we chose option 3) we have two options, either we just say $\alpha=\ln{\frac{1}{\beta}}$ or $\alpha=\ln{\beta}$ (I used the $log_e$ instead of $log_2$ so I can juggle more with it)  
#
# *   $\alpha=\ln{(\frac{1}{\beta})} =\ln{(1+e^{-\epsilon})}$
# *   $\alpha=\ln{(\beta)}=\ln{(\frac{1}{1+e^{-\epsilon}})}=\ln{(\frac{e^{\epsilon}}{e^{\epsilon} + 1})}$
#
# Now we need to introduce a new problem, the $\alpha_i>0 \forall i$ this means that the first option is suitable because $\ln{(1+e^{-\epsilon})} > 0\text{  }\forall \epsilon \in R$.
# Whereas the second one:
#
# *  $\ln{(\frac{e^{\epsilon}}{e^{\epsilon} + 1})}>0 \text{     }→\text{     }\ln{e^{\epsilon}} - \ln{(e^{\epsilon} +1)} > 0\text{     }→\text{     }(\frac{e^{\epsilon}}{e^{\epsilon} + 1}) > 1\text{     }→\text{     }e^{\epsilon} > e^{\epsilon} + 1   \text{          }\nexists \epsilon \in R $
#
#
# So second option is unsuitable for us whereas the first one atleast holds the constraint we set ($\alpha > 0$). The assumption behind this value: $\alpha=\ln{(1+e^{-\epsilon})}$ is the greater the error the smaller will be the final value of $\alpha$.
#
#
#
#

# %% id="6IRy6o5WZJbV"
import math

class WeakLearner:
    def __init__(
            self, dataset: Dataset, weights: Tensor,
            epochs: int = 10, verbose: int = 0
    ):
        self._dataset: Dataset= dataset
        self._weights: Tensor = weights
        self._weights.requires_grad_()
        self._simple_learner: SimpleLearner = SimpleLearner()
        self._error_rate: float = .0
        self._beta: float = .0
        self._accuracy: float = .0
        self._weights_map: Tensor = th.ones(self._weights.shape[0], dtype=th.bool)

        self._fit(epochs=epochs, verbose=verbose)

    def _fit(self, epochs: int = 5, verbose: int = 0) -> None:
       training_result: tuple[tuple[Tensor, Tensor], float] = self._simple_learner.fit(
            dataset=self._datset, adaboost_wgt=self._weights,
            epochs=epochs, verbose=verbose
        )

       self._error_rate = training_result[1]
       self._beta = 1 / (1 + math.exp(-self._error_rate)) # sigmoid
       self._update_weights_map(training_result[0])

    def _update_weights_map(self, data: tuple[Tensor, Tensor]) -> None:
        classes_mask: Tensor = dataset.get_class_tensor()
        preds, ids = data
        for pred, _id in zip(preds, ids):
          model_pred: int = th.argmax(pred).item()
          weight_flag: bool = model_pred == classes_mask[_id].value
          self._weights_map[_id.to(th.int32)] = weight_flag

    def predict(self, samples: Tensor) -> Tensor:
        return self._simple_learner.predict(samples)

    def get_error_rate(self) -> float:
        return self._error_rate

    def get_beta(self) -> float:
        return self._beta

    def get_weights(self) -> Tensor:
        return self._weights.detach()

    def get_weights_map(self) -> Tensor:
        return self._weights_map




# %% [markdown] id="WWIBpU8_s15G"
# ## StrongLearner

# %% id="Trm_Gj6ys5gX"
class StrongLearner(object):
    def __init__(self, weak_learners: Tensor, k_classes: int = 2):
        self._weak_learners: Tensor = weak_learners
        self._k_classes: int = k_classes
        self._alphas: Tensor = th.ones(self._weak_learners.shape[0])
        self._initialize_alphas()

    def predict_image(self, img: Tensor) -> Tensor:
        final_dist: Tensor = th.zeros(self._k_classes)
        for weak_learner, alpha in zip(self._weak_learners, self._alphas):
          wk_pred: Tensor = th.squeeze(weak_learner.predict(), 1)
          final_dist += alpha * wk_pred

        return th.argmax(final_dist / euclidean.norm(final_dist, 1))

    def predict_images(self, images: Tensor) -> Tensor:
       preds: Tensor = th.tensor([], dtype=th.int32)

       for img in images:
        pred: Tensor = th.unsqueeze(self._predict_image(img), dim=0)
        preds = th.cat((preds, pred))

       return preds

    def _initialize_alphas(self) -> None:
        for idx, weak_learner in enumerate(self._weakLearners):
            self._alphas[idx] = math.log(1 / weak_learner.getBeta())

    def get_weak_learners(self) -> Tensor:
        return self._weak_learners




# %% [markdown] id="LTy251cnxTig"
# #### Tests

# %% colab={"base_uri": "https://localhost:8080/"} id="QEm_B8F8sDTw" outputId="6e668bab-87a2-4bd0-c336-196e7101ef8f"
def fun(i: int):
  alphas = th.tensor([0.4, 0.5, 0.6])
  dist = th.zeros(2)

  for idx in range(3):
    pred = th.squeeze(model.predict(images[i]), 1)
    dist += alphas[idx] * pred

  return th.argmax(dist / euclidean.norm(dist, 1))

preds = th.tensor([])

for i in range(5):
  preds = th.cat((preds, th.unsqueeze(fun(i), 0)))

preds

# %% [markdown] id="mKD4YakusOsn"
# ## AdaBoost training

# %% id="IhLMpbJxse1w"
from typing import Callable
from numpy import ndarray, array, append
from torch import Tensor, ones, sum

"""
    Class wrapping all the functionalities needed to make a training algorithm based
    on an ensemble approach
"""


class AdaBoost:
    def __init__(
            self, n_eras: int, dataset: Dataset,
            n_classes: int, weak_learner_epochs: int = 10,
    ):
        self._n_eras: int = n_eras
        self._dataset: Dataset = dataset
        self._n_classes: int = n_classes
        self._weak_learner_epochs: int = weak_learner_epochs
        self._weak_learners: Tensor = th.tensor([])
        self._weights: Tensor = AdaBoost.initialize_weights()

    @staticmethod
    def initialize_weights() -> Tensor:
        classes_mask: Tensor = dataset.get_class_tensor()
        weights: Tensor = th.zeros(classes_mask.shape[0])

        for enum in list(Labels):
            card: int = dataset.get_cardinality(Labels[enum.name])
            weights[classes_mask == Labels[enum.name]] = 1 / (2 * card)

        return weights

    @staticmethod
    def normalize_weights(weights: Tensor) -> None:
        weights /= sum(weights)

    @staticmethod
    def update_weights(weights: Tensor, weak_learner_beta: float,
                       weak_learner_weights_map: Tensor) -> None:

        weights[weak_learner_weights_map] *= weak_learner_beta

    def start_generator(self, update_weights: bool = True, verbose: int = 0) -> Callable[[Tensor], StrongLearner]:

        def detached_start(weights: Tensor) -> StrongLearner:

            for era in range(self._n_eras):
                AdaBoost.normalize_weights(weights)

                weak_learner: WeakLearner = WeakLearner(
                    self._dataset, self._weights,
                    self._weak_learner_epochs, verbose
                )

                if update_weights:
                    AdaBoost.update_weights(
                        weights, weak_learner.get_beta(),
                        weak_learner.get_weights_map()
                    )

                self._weak_learners = th.cat((
                    self._weak_learners,
                    th.tensor([weak_learner])
                ))

                if verbose > 1:
                    print(f"\033[31mEras left: {self._n_eras - (era + 1)}\033[0m")

            return StrongLearner(self._weak_learners)

        return detached_start

    def start(self, verbose: int = 0) -> StrongLearner:
        start: Callable[[Tensor], StrongLearner] = self.start_generator(True, verbose)
        return start(self._weights)

    def get_weights(self) -> Tensor:
        return self._weights

