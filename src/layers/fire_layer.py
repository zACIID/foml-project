from typing import Type

from torch import Tensor, cat
import torch.nn as nn


class FireLayer(nn.Module):
    def __init__(
            self,
            in_channels_sqz: int,
            out_channels_sqz: int,
            out_channels_exp_ones: int,
            out_channels_exp_threes: int,
            act_fun: Type[nn.Module],
            n_dims: int = 3,
            device: str = None
    ):
        super().__init__()

        self._squeeze_layer: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels_sqz, out_channels=out_channels_sqz,
            kernel_size=1, device=device
        )
        self._expand_layer_ones: nn.Conv2d = nn.Conv2d(
            in_channels=out_channels_sqz, out_channels=out_channels_exp_ones,
            kernel_size=1, device=device
        )
        self._expand_layer_threes: nn.Conv2d = nn.Conv2d(
            in_channels=out_channels_sqz, out_channels=out_channels_exp_threes,
            padding=1, kernel_size=3, device=device
        )

        self._n_dims: int = n_dims
        self._act_fun: nn.Module = act_fun()

    def forward(self, ft_map: Tensor) -> Tensor:
        # apply squeeze layer and non linearity
        sqz_ft_map: Tensor = self._act_fun(self._squeeze_layer(ft_map))

        # apply expand layer
        exp_ones_ft_map: Tensor = self._expand_layer_ones(sqz_ft_map)
        exp_threes_ft_map: Tensor = self._expand_layer_threes(sqz_ft_map)

        # choose on which dimension to concat
        dim: int = int(self._n_dims != exp_ones_ft_map.dim())

        # concatenate the 1x1 and 3x3 filters features maps
        return cat((exp_ones_ft_map, exp_threes_ft_map), dim=dim)
