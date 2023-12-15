import pandas as pd

import src.utils.time_utils as ut

# Used for HalvingGridSearchCV, still experimental in sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html
from sklearn.experimental import enable_halving_search_cv
import sklearn.model_selection as modsel


@ut.print_time_perf
def grid_search_cv_tuning(
        model,
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        hyper_params: dict[str, object],
        k_folds: int,
        n_jobs: int = None,
        verbosity: int = 0,
        **gridsearchcv_kwargs
) -> modsel.HalvingGridSearchCV:
    """
    Tunes the model by performing cross (stratified) k-fold cross validation
    on the provided data and hyperparameters.

    Mostly a wrapper of sklearn.model_selection.HalvingGridSearchCV
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html)

    :param model: model to tune
    :param train_data: training data
    :param train_target: target associated with the training data
    :param hyper_params: dictionary containing possible values of the hyperparameters
        used in the validation process. Keyed by param name.
    :param k_folds: number of folds in the cross validation process
    :param n_jobs: number of jobs to run the grid search cv on.
        If None, just 1; if -1, then all processors are used.
    :param verbosity: level of verbosity
    :param gridsearchcv_kwargs: additional keyword arguments passed to
        the underlying grid search instance
    :return: grid search cv instance
    """

    tuned_model = modsel.HalvingGridSearchCV(
        estimator=model,
        param_grid=hyper_params,
        cv=k_folds,
        verbose=verbosity,
        n_jobs=n_jobs,
        **gridsearchcv_kwargs,
    )
    tuned_model.fit(train_data, train_target)

    return tuned_model
