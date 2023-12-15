import torch.nn as nn
from torch import Tensor, no_grad
from torch.optim import Adam

from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader

from datasets.custom_coco_dataset import ItemType, BatchType
from layers.fire_layer import FireLayer
from loss_functions.base_weighted_loss import WeightedBaseLoss
from loss_functions.weighted_cross_entropy import WeightedCrossEntropy


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

    def fit(self, dataset: Dataset[ItemType], adaboost_wgt: Tensor, batch_size: int = 32,
            epochs: int = 10, loss: WeightedBaseLoss = WeightedCrossEntropy,
            verbose: int = 0) -> tuple[tuple[Tensor, Tensor], float]:

        adam_opt: Adam = Adam(self.parameters())
        wgtd_loss: WeightedBaseLoss = loss()
        cum_loss: float = .0

        self.train()
        for epoch in range(epochs):
            cum_loss = .0
            for batch in DataLoader(dataset, batch_size, shuffle=True):
                batch: BatchType
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
        # TODO(pierluigi): domanda per biagio
        #   sta cosa qui sotto funziona perché nn.Module implementa __call__ che chiama
        #   automaticamente tutti i campi della classe che sono di tipo nn.Module.
        #   Ma in che ordine li chiama poi? E' tutto gestito comunque?
        #   Altra domanda e che cos'e samples, se e un singolo dato oppure anche un batch di dati

        with no_grad():
            self.eval()  # Set the model to evaluation mode (if applicable)

            return self.__call__(samples)

    def get_modules(self) -> nn.Sequential:
        return self._pipe_line

    def set_modules(self, new_modules: nn.Sequential) -> None:
        self._pipe_line = new_modules
