import math
from typing import Tuple, Sequence

import pandas as pd
from sklearn.utils.validation import check_array

import visualization.exploration as vis
from ..datasets.dataset_analyzer import DatasetAnalyzer, Dataset, DatasetAnalysis, AnalysisPlotTypes


class MNISTDigitsAnalyzer(DatasetAnalyzer):
    def __init__(
            self,
            feature_subplot_size: Tuple[float, float] = (2, 2),
            features_idx_to_plot: Sequence[int] = None
    ):
        """

        :param feature_subplot_size: Used to determine the size of a feature subplot in a feature grid
            generated during the analysis
        :param features_idx_to_plot: collection of feature indices to plot when generating charts via analyze().
            Useful because datasets might have a lot of features, causing the plot computation
            to be extremely expensive. Defaults to the first 50 features
            (all of them if data has less than 50 features).
        """

        self.feature_subplot_size = feature_subplot_size
        self.features_idx_to_plot = features_idx_to_plot

    def analyze(self, dataset: Dataset) -> DatasetAnalysis:
        check_array(dataset.X)

        if self.features_idx_to_plot is None:
            self.features_idx_to_plot = list(range(0, min(50, dataset.X.shape[1])))

        df = pd.DataFrame(
            data=dataset.X[:, self.features_idx_to_plot],
            index=range(0, dataset.X.shape[0]),
            columns=(f"f{i}" for i in self.features_idx_to_plot)
        )

        fig1, axs1 = vis.feature_distributions_plot(
            data=df,
            subplot_size=self.feature_subplot_size,
            numerical_mode="boxplot",
            num_barplot_under=3,
            width=math.floor(math.sqrt(len(self.features_idx_to_plot))),
            title=f"{dataset.id} (MNIST Digit Dataset)"
        )

        # TODO seems unstable probably beacause too many features, makes python crash
        # fig2, axs2 = vis.bivariate_feature_plot(
        #     data=df,
        #     y_var=("digits", pd.Series(dataset.y)),
        #     subplot_size=self.feature_subplot_size
        # )

        return DatasetAnalysis(
            dataset=dataset,
            plots={
                AnalysisPlotTypes.FEATURE_DISTRIBUTIONS: fig1,
                # AnalysisPlotTypes.BIVARIATE_PLOTS: fig2
            }
        )
