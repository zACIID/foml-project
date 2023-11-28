from typing import Optional

from sklearn.metrics import rand_score

import visualization.exploration as vis_expl
import visualization.mnist as vis_mnist
import visualization.unsupervised_learning as vis_ul
from classifiers.ncut import NCut
from pipeline.analysis.models.model_evaluator import ModelEvaluator, T, ModelEvaluation, PlotType, ScoreTypes
from pipeline.dataset_providers.base import Dataset
from pipeline.model_builder import ModelData


class NCutEvaluator(ModelEvaluator):
    def evaluate(self, model_data: ModelData[T]) -> Optional[ModelEvaluation[NCut]]:
        if not isinstance(model_data.model, NCut):
            # Skip evaluation, cannot handle other models
            return None

        model: NCut = model_data.model
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
