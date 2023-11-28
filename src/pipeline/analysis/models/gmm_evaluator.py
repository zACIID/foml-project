import math
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import rand_score

import visualization.exploration as vis_expl
import visualization.mnist as vis_mnist
import visualization.unsupervised_learning as vis_ul
from classifiers.gmm import GMM
from pipeline.analysis.models.model_evaluator import ModelEvaluator, T, ModelEvaluation, PlotType, ScoreTypes
from pipeline.dataset_providers.base import Dataset
from pipeline.model_builder import ModelData


class GMMEvaluator(ModelEvaluator):
    def evaluate(self, model_data: ModelData[T]) -> Optional[ModelEvaluation[GMM]]:
        if not isinstance(model_data.model, GMM):
            # Skip evaluation, cannot handle other models
            return None

        model: GMM = model_data.model
        dataset = (model_data.testing_dataset_engineered
                   if model_data.testing_dataset_engineered is not None
                   else model_data.testing_dataset)
        X = dataset.X

        predictions = model.predict(X)

        # Plot digits on the original dataset
        plotting_dataset = (model_data.testing_dataset
                            if model_data.testing_dataset is not None
                            else model_data.testing_dataset_engineered)
        mean_digits_plot = vis_mnist.plot_mean_cluster_images(
            dataset=Dataset(
                X=plotting_dataset.X,
                y=predictions
            ),
        )
        label_counts_plot = vis_expl.label_counts_histogram(labels=predictions)
        label_counts_plot.suptitle("Predictions")
        conf_matrix_plot = vis_ul.confusion_matrix(y_true=dataset.y, y_pred=predictions)

        rand_idx = rand_score(labels_true=dataset.y, labels_pred=predictions)

        return ModelEvaluation(
            model_data=model_data,
            predictions=predictions,
            scores={
                ScoreTypes.RAND_INDEX: rand_idx
            },
            plots={
                PlotType.MEAN_CLASSIFICATIONS_PLOT: mean_digits_plot,
                PlotType.LABEL_COUNTS: conf_matrix_plot,
                PlotType.CONFUSION_MATRIX: conf_matrix_plot
            }
        )


def internal_gmm_representation(
        evaluation: ModelEvaluation[GMM],
        figsize_pixels: Tuple[int, int] = (1280, 720)
) -> go.Figure:
    gmm = evaluation.model_data.model.inner_model

    # array-like of shape (n_components, n_features)
    # The mean of each mixture component
    mixture_means = gmm.means_

    # Make means plottable in a square image grid
    n_feats = gmm.n_features_in_
    image_side_pixels = math.ceil(math.sqrt(n_feats))

    # Calculate remaining "cells" to fill with 0s so that the array can
    #   be rescaled to a square image
    padding = image_side_pixels ** 2 - n_feats
    padded_means = np.array([
        np.concatenate([m, np.zeros(padding)])
        for m in mixture_means
    ])

    return vis_mnist.plot_mean_cluster_images(
        dataset=Dataset(
            X=padded_means,
            y=np.array(range(0, len(padded_means)))  # One label for each mixture mean
        ),
        image_width=image_side_pixels,
        image_height=image_side_pixels,
        top_k=10,
        figsize_pixels=figsize_pixels
    )
