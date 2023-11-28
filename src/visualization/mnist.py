import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns

from pipeline.dataset_providers.base import Dataset
from pipeline.feature_engineering.pca_engineering import PCAEngineering


def plot_mean_cluster_images(
        dataset: Dataset,
        top_k: int = 25,
        image_height: int = 28,
        image_width: int = 28,
        figsize_pixels: Tuple[int, int] = (1920, 1920)
) -> go.Figure:
    """
    Plots the mean (digits) of the top k cluster samples, sorted by cardinality,
        for the provided dataset and predictions
    :param dataset: samples to plot (X) and respective cluster labels (y)
    :param top_k: top k clusters, sorted by cardinality, to plot, truncate the rest.
        This limit is set because the number of clusters can be unfeasibly big to plot.
        Defaults to 50.
    :param image_height: image height in pixels, used to reshape the dataset, which is a 2d array
    :param image_width: image width in pixels, used to reshape the dataset, which is a 2d array
    :param figsize_pixels: size (width, height) in pixels
    :return:
    """

    unique_labels, counts = np.unique(dataset.y, return_counts=True)

    # Sort labels by counts, desc
    argsort_indices = -counts.argsort()
    unique_labels = unique_labels[argsort_indices]

    label_means = np.array([
        # Reshape mean into $image_width \times image_height$ image matrix
        np.mean(dataset.X[dataset.y == lbl], axis=0).reshape(image_width, image_height)
        for lbl in unique_labels
    ])

    fig: go.Figure = px.imshow(
        img=label_means[:top_k, :],
        color_continuous_scale="gray",
        binary_string=True,  # This is because the image pixels are numbers \in [0, 1]
        width=figsize_pixels[0],
        height=figsize_pixels[1],
        facet_col=0,
        facet_col_wrap=math.floor(math.sqrt(top_k)),
    )

    fig_annotations: List[go.Annotation] = list(fig['layout']['annotations'])
    """
    Example annotation of the figure (as of plotly 5.16.1):
    
    layout.Annotation({
        'font': {},
        'showarrow': False,
        'text': 'facet_col=8',
        'x': 0.1175,
        'xanchor': 'center',
        'xref': 'paper',
        'y': 0.2866666666666666,
        'yanchor': 'bottom',
        'yref': 'paper'
    })
    
    It can be noted that the text contains the index of the subplot, starting from the top-left.
    We can use such an index to sort the annotations and map them to the labels assigned
        to the subplots
    """
    fig_annotations.sort(key=lambda a: (a["text"][-1]))

    for i in range(len(fig_annotations)):
        fig_annotations[i].update(text=f"Label: {unique_labels[i]}")

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig


def plot_image(
        pixels: np.ndarray,
        image_height: int = 28,
        image_width: int = 28,
) -> plt.Figure:
    """
    :param pixels: 1D array of pixels which will be reshaped into a matrix of image_width \times image_height,
        which represents the digit to plot
    :param image_height: image height in pixels, used to reshape the dataset, which is a 2d array
    :param image_width: image width in pixels, used to reshape the dataset, which is a 2d array
    """

    fig, ax = plt.subplots(1)
    fig: plt.Figure
    ax: plt.Axes

    pixels = pixels.reshape(image_width, image_height)
    ax.imshow(pixels, cmap=plt.cm.get_cmap("gray"), interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return fig


def plot_3d(
        data: Dataset,
) -> go.Figure:
    """
    Plots all the digit images in the dataset in a 3-dimensional space

    :param data: dataset to plot
    """

    # principal component analysis producing two components
    pca = PCAEngineering(n_components=3)
    dummy_dataset = Dataset(X=data.X[[0, 1], :], y=data.y[[0, 1]])
    dataset_2d, _ = pca.engineer(train=data, test=dummy_dataset)

    # convert to str so plotly knows that it's discrete value and uses discrete color scale
    cat_labels = [str(lbl) for lbl in np.sort(dataset_2d.y)]
    df = pd.DataFrame(
        data={
            "x": dataset_2d.X[:, 0],
            "y": dataset_2d.X[:, 1],
            "z": dataset_2d.X[:, 2],
            "cluster": cat_labels
        }
    )
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        color_discrete_sequence=px.colors.qualitative.G10,
        title="3D Dataset",
        category_orders={
            "cluster": cat_labels
        }
    )

    return fig


def plot_2d(
        data: Dataset,
        figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    """
    Plots all the digit images in the dataset in a 2-dimensional space

    :param data: dataset to plot
    :param figsize: size (w, h) of the plot in inches
    """

    # principal component analysis producing two components
    pca = PCAEngineering(n_components=2)
    dummy_dataset = Dataset(X=data.X[[0, 1], :], y=data.y[[0, 1]])
    dataset_2d, _ = pca.engineer(train=data, test=dummy_dataset)

    fig, ax = plt.subplots(1, figsize=figsize)
    fig: plt.Figure
    ax: plt.Axes

    palette = sns.color_palette()
    sns.scatterplot(
        ax=ax,
        x=dataset_2d.X[:, 0],
        y=dataset_2d.X[:, 1],
        hue=dataset_2d.y,
        palette=palette,
        alpha=0.7
    )

    fig.suptitle("2D Dataset")

    return fig
