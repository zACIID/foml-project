from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.feat_engineering as fe
from visualization.base import get_plotting_grid


def feature_distributions_plot(
        data: pd.DataFrame,
        subplot_size: Tuple[float, float],
        numerical_mode: str = "boxplot",
        num_barplot_under: int = 10,
        title: str = "",
        title_size: int = 22,
        width: int = 4
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plots the distribution of all the feature of the provided dataframe in a grid like manner.
    Numerical feature distributions are plotted with a boxplot or violin plot (depending on the specified mode),
    while distributions for other types of feature (presumably categorical) are plotted with a histogram.

    :param data: dataframe to plot the feature distributions for
    :param numerical_mode: type of plot to make for numerical features.
        Can be either "boxplot" or "violin".
    :param num_barplot_under: limit of unique values a numerical feature can have to have its
        distribution plotted as a barplot.
    :param subplot_size: dimension of each subplot in the grid
    :param title: title of the whole plot
    :param title_size: size of the title
    :param width: width (in plots) of the grid
    :return: (fig, axs) grid of distribution plots, one for each feature of the provided dataframe.
    """

    # Arg check
    BOXPLOT_MODE = "boxplot"
    VIOLIN_MODE = "violin"
    if numerical_mode != BOXPLOT_MODE and numerical_mode != VIOLIN_MODE:
        raise Exception(f"Mode can be either '{BOXPLOT_MODE}' or '{VIOLIN_MODE}', got {numerical_mode}")

    # Create plotting grid
    n_features = len(data.columns)
    fig, axs = get_plotting_grid(width, n_features, subplot_size, style="whitegrid")

    # Boxplot/Violin plot for numeric features, histogram for the rest
    plot_row = 0
    for i, f_col in enumerate(data.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        feature_to_plot = data[f_col]
        plot_onto = axs[plot_row, plot_col]

        # Check if the feature is numerical
        # integer, unsigned, float, complex
        n_uniques = len(feature_to_plot.unique())
        if feature_to_plot.dtype.kind in 'iufc' and n_uniques > num_barplot_under:
            plot_func = sns.boxplot if numerical_mode == BOXPLOT_MODE else sns.violinplot
            plot_func(ax=plot_onto, x=feature_to_plot, color="#ffa500")
        else:
            # reset_index() because it makes the feature values, which were the index, into a proper column,
            #   which can then be referred in the barplot func
            # normalize=True creates a "proportion" column
            val_counts = feature_to_plot.value_counts(normalize=True).reset_index()

            # Case of numerical feature with few values:
            # truncate values after 3 decimal digits, else they're too long to display
            if feature_to_plot.dtype.kind in 'iufc':
                val_counts[f_col] = val_counts[f_col].round(1)

            plot_ax = sns.barplot(ax=plot_onto, data=val_counts, x=f_col, y="proportion", palette="pastel")
            plot_ax.set(xlabel=f_col, ylabel="proportion")

    if title != "":
        fig.suptitle(title, fontsize=title_size)
        fig.tight_layout()

    return fig, axs


def bivariate_feature_plot(
        data: pd.DataFrame,
        y_var: Tuple[str, pd.Series],
        subplot_size: Tuple[float, float],
        mode: str = "hexbin",
        width: int = 2,
        percentile_range: Tuple[float, float] = (0, 100),
        title: str = "",
        title_size: int = 30,
        show_legend: bool = True,
        hexbin_kwargs: Dict[str, object] = None,
        scatter_kwargs: Dict[str, object] = None
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plots a grid of hexbin plots, each comparing a feature in the provided dataframe with the
    provided target.

    :param data: dataframe containing the features to compare with the provided variable (x-axis)
    :param y_var: variable (name, data) to compare with the provided features (y-axis).
    :param subplot_size: dimension of each subplot in the grid
    :param title: title of the whole plot
    :param title_size: size of the title
    :param width: width (in plots) of the grid
    :param mode: type of plots to draw, can be either "hexbin" or "scatter"
    :param percentile_range: range that determines which values will be displayed in the plots.
        Useful because the presence of outliers makes the chart less clear.
    :param show_legend: True if legend has to be displayed for each subplot, false otherwise
    :param hexbin_kwargs: additional parameter to pass to the underlying pyplot.hexbin,
        used when mode argument is "scatter"
    :param scatter_kwargs:  additional parameter to pass to the underlying seaborn.scatterplot,
        used when mode argument is "scatter"
    :return: (fig, axs) a grid of hexbin or scatter plots
    """

    # Arg check
    HEXBIN_MODE = "hexbin"
    SCATTER_MODE = "scatter"
    if mode != HEXBIN_MODE and mode != SCATTER_MODE:
        raise Exception(f"Mode can be either '{HEXBIN_MODE}' or '{SCATTER_MODE}', got {mode}")

    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Create grid
    n_features = len(data.columns)
    fig, axs = get_plotting_grid(width, n_features, subplot_size)

    # Create a plot for each grid square
    plot_row = 0
    for i, col in enumerate(data.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        feature = data[col]
        y_name, y_data = y_var

        # Get the data withing the specified percentile range
        lower_q = percentile_range[0] / 100
        upper_q = percentile_range[1] / 100
        x_ranged, y_ranged = _get_within_quantile_range(x=feature, y=y_data,
                                                        lower_q=lower_q, upper_q=upper_q)

        # Set x and y labels to feature and y_var names
        plot_onto = axs[plot_row, plot_col]
        plot_onto.set_xlabel(col)
        plot_onto.set_ylabel(y_name)

        if mode == HEXBIN_MODE:
            hexbin = plot_onto.hexbin(x=x_ranged.values, y=y_ranged.values,
                                      **hexbin_kwargs)
            if show_legend:
                cb = fig.colorbar(hexbin, ax=plot_onto)
                cb.set_label('counts')
        else:
            # Select the data from the original dataframe in order to keep the other columns:
            # this  way, seaborn kwargs that refer to such columns (e.g. hue, size) can be passed
            scatter_data = data.copy()
            scatter_data = scatter_data[scatter_data[col].isin(x_ranged.values)]
            scatter_data[y_name] = y_data
            scatter_data = scatter_data[scatter_data[y_name].isin(y_ranged.values)]

            sns.scatterplot(ax=plot_onto, data=scatter_data, x=col, y=y_name,
                            **scatter_kwargs)

            if not show_legend:
                legend = plot_onto.get_legend()

                if legend is not None:
                    legend.remove()

    if title != "":
        plt.suptitle(title, fontsize=title_size)

    return fig, axs


def _get_within_quantile_range(x: pd.Series, y: pd.Series,
                               lower_q: float, upper_q: float) -> (pd.Series, pd.Series):
    """
    Returns the (x, y) pairs where both values fall in the specified quantile range.

    :param x: x values
    :param y: y values
    :param lower_q: lower limit of the range
    :param upper_q: upper limit of the range
    :return:
    """
    quantile_range_mask = (
            (x >= x.quantile(lower_q))
            & (x <= x.quantile(upper_q))
            & (y >= y.quantile(lower_q))
            & (y <= y.quantile(upper_q))
    )

    x_ranged = x[quantile_range_mask]
    y_ranged = y[quantile_range_mask]

    return x_ranged, y_ranged


# TODO(pierluigi): consider using https://github.com/ResidentMario/missingno
def missing_values_plot(
        data: pd.DataFrame,
        subplot_size: Tuple[float, float],
        title: str = "", width: int = 7,
        mode: str = "pct",
        **barplot_kwargs
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plots a grid of barplots, each representing missing value information about each feature in the
    provided dataframe.

    :param data: data to plot missing values information for
    :param subplot_size: dimension of each subplot in the grid
    :param title: title of the (whole) plot
    :param width: width (in plots) of the grid
    :param mode: either "pct" or "count". If pct, plots missing%, else missing count
    :param barplot_kwargs: additional parameter to pass to the underlying barplot function
    :return: (fig, axs) a grid of barplots about the numbers/percentages of missing values
        for each feature in the provided dataframe
    """

    # Arg check
    PCT_MODE = "pct"
    COUNT_MODE = "count"
    if mode != PCT_MODE and mode != COUNT_MODE:
        raise Exception(f"mode parameter should be '{PCT_MODE}' or '{COUNT_MODE}', got {mode}")

    # Get dataframe containing missing values information
    missing_df = fe.get_missing_info(df=data)
    features_col = missing_df["column"]
    pct_col = missing_df["missing %"]
    count_col = missing_df["missing count"]

    # Create plotting grid
    n_features = len(data.columns)
    fig, axs = get_plotting_grid(width, n_features, subplot_size, style="whitegrid")

    # Create a plot for each grid square
    plot_row = 0
    for i, feature in enumerate(data.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        plot_onto = axs[plot_row, plot_col]

        # Mask used to extract values belonging to the current feature
        # from the missing info dataframe
        current_feature_mask = features_col == feature

        # Get the current feature missing information
        if mode == PCT_MODE:
            missing_info = pct_col[current_feature_mask]
            plot_onto.set_ylim([0, 100])

            # Show tick every 10%
            plot_onto.set_yticks(list(range(0, 101, 10)))
        else:
            missing_info = count_col[current_feature_mask]
            plot_onto.set_ylim([0, len(data)])

        sns.barplot(ax=plot_onto, x=[feature], y=missing_info, **barplot_kwargs)

    if title != "":
        plt.suptitle(title, fontsize=16)

    return fig, axs


def label_counts_histogram(labels: np.ndarray) -> plt.Figure:
    """
    Plot clusters frequencies in a histogram
    :param labels: labels associated to each data point
    :return:
    """

    fig, ax = plt.subplots(1)
    fig: plt.Figure
    ax: plt.Axes

    unique_labels, counts = np.unique(labels, return_counts=True)
    ax.bar(unique_labels, counts, edgecolor='black')

    # Title and axes
    ax.set_title('Label Counts')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')

    return fig
