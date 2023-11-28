from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def pca_variances_chart(
        pca: PCA,
        ax: plt.Axes = None,
        figsize: Tuple[float, float] = (12, 12)
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    else:
        fig = ax.figure

    # Bar plot of explained_variance
    sns.barplot(
        ax=ax,
        x=range(1, len(pca.explained_variance_) + 1),
        y=pca.explained_variance_
    )

    ax.xlabel('PCA Feature')
    ax.ylabel('Explained variance')
    fig.title('PCA Explained Variance')
    fig.show()

    return fig, ax
