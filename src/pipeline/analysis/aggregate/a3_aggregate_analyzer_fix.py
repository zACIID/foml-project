import enum
from typing import Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from classifiers.base import BaseClassifier
from classifiers.gmm import GMM
from classifiers.mean_shift import MeanShift
from classifiers.ncut import NCut
from pipeline.analysis.aggregate.aggregate_analyzer import AggregateAnalyzer, AggregateAnalysis
from pipeline.analysis.models.model_evaluator import ModelEvaluation, ScoreTypes
from pipeline.dataset_providers.base import Dataset

"""
Keeping this here in remembrance of tough times where I didn't want to break already serialized objects
"""


class A3AggregatePlotTypesFix(enum.Enum):
    GMM_TRAINING_TIMES_BY_DIM = 0
    MEAN_SHIFT_TRAINING_TIMES_BY_DIM = 1
    NCUT_TRAINING_TIMES_BY_DIM = 2
    ALL_MODEL_TRAINING_TIMES_BY_DIM = 3

    AVG_MODEL_RAND_INDEX_BY_DIM = 4

    GMM_RAND_INDEX_BY_DIM = 5
    MEAN_SHIFT_RAND_INDEX_BY_DIM = 6
    NCUT_RAND_INDEX_BY_DIM = 7
    ALL_MODEL_RAND_INDEX_BY_DIM = 8

    GMM_N_CLUSTERS_BY_DIM = 9
    MEAN_SHIFT_N_CLUSTERS_BY_DIM = 10
    NCUT_N_CLUSTERS_BY_DIM = 11
    ALL_N_CLUSTERS_BY_DIM = 12


class A3AggregateAnalyzerFix(AggregateAnalyzer[A3AggregatePlotTypesFix]):
    """
    Aggregate analyzer for FoAI's 3rd Assignment
    """

    def analyze(self, evaluations: Sequence[ModelEvaluation]) -> AggregateAnalysis[A3AggregatePlotTypesFix]:
        gmm_evals: Sequence[ModelEvaluation[GMM]] = [
            e for e in evaluations if isinstance(e.model_data.model, GMM)
        ]
        ms_evals: Sequence[ModelEvaluation[MeanShift]] = [
            e for e in evaluations if isinstance(e.model_data.model, MeanShift)
        ]
        ncut_evals: Sequence[ModelEvaluation[NCut]] = [
            e for e in evaluations if isinstance(e.model_data.model, NCut)
        ]

        gmm_evals_df = _gmm_evals_to_dataframe(gmm_evals)
        ms_evals_df = _mean_shift_evals_to_dataframe(ms_evals)
        ncut_evals_df = _ncut_evals_to_dataframe(ncut_evals)
        combined_df = pd.concat([gmm_evals_df, ms_evals_df, ncut_evals_df], ignore_index=True)

        # Find best models for each type
        best_models = {}
        worst_models = {}
        if len(gmm_evals) > 0:
            best_models[GMM] = _get_best_model(gmm_evals, eval_df=gmm_evals_df)
            worst_models[GMM] = _get_worst_model(gmm_evals, eval_df=gmm_evals_df)
        if len(ms_evals) > 0:
            best_models[MeanShift] = _get_best_model(ms_evals, eval_df=ms_evals_df)
            worst_models[MeanShift] = _get_worst_model(ms_evals, eval_df=ms_evals_df)
        if len(ncut_evals) > 0:
            best_models[NCut] = _get_best_model(ncut_evals, eval_df=ncut_evals_df)
            worst_models[NCut] = _get_worst_model(ncut_evals, eval_df=ncut_evals_df)

        return AggregateAnalysis(
            plots={
                # Training times
                A3AggregatePlotTypesFix.GMM_TRAINING_TIMES_BY_DIM: _training_times_plot(gmm_evals_df),
                A3AggregatePlotTypesFix.MEAN_SHIFT_TRAINING_TIMES_BY_DIM: _training_times_plot(ms_evals_df),
                A3AggregatePlotTypesFix.NCUT_TRAINING_TIMES_BY_DIM: _training_times_plot(ncut_evals_df),
                A3AggregatePlotTypesFix.ALL_MODEL_TRAINING_TIMES_BY_DIM: _training_times_plot(combined_df),

                # Rand Index
                A3AggregatePlotTypesFix.GMM_RAND_INDEX_BY_DIM: _rand_idx_plot(gmm_evals_df),
                A3AggregatePlotTypesFix.MEAN_SHIFT_RAND_INDEX_BY_DIM: _rand_idx_plot(ms_evals_df),
                A3AggregatePlotTypesFix.NCUT_RAND_INDEX_BY_DIM: _rand_idx_plot(ncut_evals_df),
                A3AggregatePlotTypesFix.ALL_MODEL_RAND_INDEX_BY_DIM: _rand_idx_plot(combined_df),

                # N clusters
                A3AggregatePlotTypesFix.GMM_N_CLUSTERS_BY_DIM: _n_clusters_plot(gmm_evals_df),
                A3AggregatePlotTypesFix.NCUT_N_CLUSTERS_BY_DIM: _n_clusters_plot(ncut_evals_df),
                A3AggregatePlotTypesFix.MEAN_SHIFT_N_CLUSTERS_BY_DIM: _n_clusters_plot(ms_evals_df),
                A3AggregatePlotTypesFix.ALL_N_CLUSTERS_BY_DIM: _n_clusters_plot(combined_df),
            },
            best_models=best_models,
            worst_models=worst_models
        )


_TRAINING_TIME_COL = "Traning Time [s]"
_MODEL_ID_COL = "Model"
_DIMENSIONS_COL = "#Dimensions"
_N_CLUSTERS_COL = "#Clusters"
_RAND_INDEX_COL = "Rand Index"
_HYPERPARAM_COL = "Hyperparameter"


def _gmm_evals_to_dataframe(evaluations: Sequence[ModelEvaluation[GMM]]) -> pd.DataFrame:
    if len(evaluations) == 0:
        return _get_empty_eval_df()

    records = [
        {
            _MODEL_ID_COL: "GMM",
            _TRAINING_TIME_COL: e.model_data.training_time,
            _N_CLUSTERS_COL: e.model_data.model.inner_model.n_components,
            _RAND_INDEX_COL: e.scores[ScoreTypes.RAND_INDEX],
            _HYPERPARAM_COL: e.model_data.model.inner_model.n_components,
            _DIMENSIONS_COL: _extract_test_dataset(e).X.shape[1]  # columns of 2D array
        } for e in evaluations
    ]

    return pd.DataFrame.from_records(data=records)


def _mean_shift_evals_to_dataframe(evaluations: Sequence[ModelEvaluation[MeanShift]]) -> pd.DataFrame:
    if len(evaluations) == 0:
        return _get_empty_eval_df()

    records = [
        {
            _MODEL_ID_COL: "MeanShift",
            _TRAINING_TIME_COL: e.model_data.training_time,

            # cluster_centers_ -> np.ndarray of shape (n_clusters, n_features)
            _N_CLUSTERS_COL: e.model_data.model.inner_model.cluster_centers_.shape[0],

            _RAND_INDEX_COL: e.scores[ScoreTypes.RAND_INDEX],
            _HYPERPARAM_COL: e.model_data.model.inner_model.bandwidth,
            _DIMENSIONS_COL: _extract_test_dataset(e).X.shape[1]  # columns of 2D array
        } for e in evaluations
    ]

    return pd.DataFrame.from_records(data=records)


def _ncut_evals_to_dataframe(evaluations: Sequence[ModelEvaluation[NCut]]) -> pd.DataFrame:
    if len(evaluations) == 0:
        return _get_empty_eval_df()

    records = [
        {
            _MODEL_ID_COL: "NCut",
            _TRAINING_TIME_COL: e.model_data.training_time,
            _N_CLUSTERS_COL: e.model_data.model.inner_model.n_clusters,
            _RAND_INDEX_COL: e.scores[ScoreTypes.RAND_INDEX],
            _HYPERPARAM_COL: e.model_data.model.inner_model.n_clusters,
            _DIMENSIONS_COL: _extract_test_dataset(e).X.shape[1]  # columns of 2D array
        } for e in evaluations
    ]

    return pd.DataFrame.from_records(data=records)


def _extract_test_dataset(e: ModelEvaluation) -> Dataset:
    return e.model_data.testing_dataset_engineered if e.model_data.testing_dataset_engineered is not None else e.model_data.testing_dataset


def _get_empty_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            _MODEL_ID_COL: [],
            _TRAINING_TIME_COL: [],
            _N_CLUSTERS_COL: [],
            _RAND_INDEX_COL: [],
            _HYPERPARAM_COL: [],
            _DIMENSIONS_COL: [],
        }
    )


def _training_times_plot(df: pd.DataFrame, figsize: Tuple[float, float] = (16, 10)) -> plt.Figure:
    return _lineplot(
        data=df,
        y_col=_TRAINING_TIME_COL,
        x_col=_DIMENSIONS_COL,
        hue_col=_HYPERPARAM_COL,
        style_col=_MODEL_ID_COL,
        figsize=figsize
    )


def _rand_idx_plot(df: pd.DataFrame, figsize: Tuple[float, float] = (16, 10)) -> plt.Figure:
    return _lineplot(
        data=df,
        y_col=_RAND_INDEX_COL,
        x_col=_DIMENSIONS_COL,
        hue_col=_HYPERPARAM_COL,
        style_col=_MODEL_ID_COL,
        figsize=figsize
    )


def _n_clusters_plot(df: pd.DataFrame, figsize: Tuple[float, float] = (16, 10)) -> plt.Figure:
    return _lineplot(
        data=df,
        y_col=_N_CLUSTERS_COL,
        x_col=_DIMENSIONS_COL,
        hue_col=_HYPERPARAM_COL,
        style_col=_MODEL_ID_COL,
        figsize=figsize
    )


def _lineplot(
        data: pd.DataFrame,
        y_col: str,
        x_col: str,
        hue_col: str = None,
        style_col: str = None,
        figsize: Tuple[float, float] = (16, 10)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fig: plt.Figure
    ax: plt.Axes

    _ = sns.lineplot(
        ax=ax,
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        palette="pastel",
        markers=True
        # facet_kws={'legend_out': True}
    )

    ax.tick_params(axis="x", rotation=90)
    plt.ticklabel_format(style="plain", axis="y", useOffset=False)  # repress scientific notation and offset usage
    # plt.close()

    return ax.figure


_B = TypeVar("_B", bound=BaseClassifier)


def _get_best_model(evaluations: Sequence[ModelEvaluation[_B]], eval_df: pd.DataFrame) -> ModelEvaluation[_B]:
    """
    :param evaluations: evaluations to extract the best model from
    :param eval_df: model evaluation dataframe
    :return: best model evaluation of provided type, based on rand index
    """

    max_rand_idx = eval_df[_RAND_INDEX_COL].max()
    best_model = [
        e
        for e in evaluations
        if e.scores[ScoreTypes.RAND_INDEX] == max_rand_idx
    ][0]  # Take the first occurrence

    return best_model


def _get_worst_model(evaluations: Sequence[ModelEvaluation[_B]], eval_df: pd.DataFrame) -> ModelEvaluation[_B]:
    """
    :param evaluations: evaluations to extract the worst model from
    :param eval_df: model evaluation dataframe
    :return: worst model evaluation of provided type, based on rand index
    """

    min_rand_idx = eval_df[_RAND_INDEX_COL].min()
    worst_model = [
        e
        for e in evaluations
        if e.scores[ScoreTypes.RAND_INDEX] == min_rand_idx
    ][0]  # Take the first occurrence

    return worst_model
