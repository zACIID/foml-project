from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mtr


def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_title: str = "Confusion Matrix",
        figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    plot = mtr.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=np.unique(y_true),
        normalize="true",
        cmap="PuRd",
        colorbar=False
    )

    fig: plt.Figure = plot.figure_
    ax: plt.Axes = plot.ax_

    ax.set_title(label=plot_title)
    ax.grid(False)
    fig.set_size_inches(w=figsize[0], h=figsize[1])

    return fig
