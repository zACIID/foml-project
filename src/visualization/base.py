from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_plotting_grid(
        width: int,
        n_cells: int,
        subplot_size: Tuple[float, float],
        style: str = "ticks",
        **subplots_kwargs
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Returns a plot grid based on the provided parameters.

    :param width: width (in plots) of the grid
    :param n_cells: total number of cells (plots) of the grid
    :param subplot_size: dimension of each subplot in the grid
    :param style: seaborn style of the plots
    :param subplots_kwargs: additional kwargs passed to the underlying pyplot.subplots call
    :return: fig, axs
    """
    sns.set_style(style=style)

    # Calculate dimensions of the grid
    height = n_cells // width
    if width * height < n_cells or height == 0:
        height += 1

    fig_width = width * subplot_size[0]
    fig_height = height * subplot_size[1]

    fig, axs = plt.subplots(ncols=width, nrows=height, figsize=(fig_width, fig_height), **subplots_kwargs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)  # Else fig.suptitle overlaps with the subplots

    return fig, axs
