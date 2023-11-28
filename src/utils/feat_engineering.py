from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, RFE

import src.utils.time_utils as utils
import src.utils.training as tr


def fill(df: pd.DataFrame, with_: dict[str, object]):
    """
    Replace (fill) missing values based on the provided dictionary of columns/fillers.

    :param df: dataframe to perform the operations in
    :param with_: dictionary that pairs column names with the value used to fill them
    """

    return df.fillna(with_)


def with_missing_mask(df: pd.DataFrame, for_cols: list[str] = None) -> pd.DataFrame:
    """
    Generate a dataframe with additional columns, one for each existing one,
    that represent a "missing value mask" made of 1 and 0.
    1 means the value in the original column is missing, 0 vice versa.

    :param df: dataframe to generate the missing value mask for
    :param for_cols: columns to generate the missing value mask for.
        If None, the mask is generated for all columns.
    :return: provided dataframe with the additional columns
    """

    # Generate a dataframe whose column contain info (either 1 or 0)
    # on whether the value was missing (nan)
    if for_cols is None:
        missing_mask_cols = df.isna().astype(int)
    else:
        missing_mask_cols = df[for_cols].isna().astype(int)

    # Rename for better readability
    rename_dict = {}
    to_rename = df.columns if for_cols is None else for_cols
    for col in to_rename:
        rename_dict[col] = f"[{col}] Missing mask"

    missing_mask_cols.rename(rename_dict, axis="columns")
    return pd.concat([df, missing_mask_cols], axis="columns")


def categorical_features_from_threshold(df: pd.DataFrame, is_categorical_threshold: int = 10,
                                        exclude: list[str] = None) -> pd.DataFrame:
    """
    Applies a heuristic that divides data based on the number of unique values they have.
    If a column (feature) in a dataframe has <= the specified threshold of unique values,
    it is considered categorical, as opposed numerical.

    :param df: dataframe to extract the categorical features from
    :param is_categorical_threshold: number of unique values over which a
        column is not considered categorical
    :param exclude: names of feature that should not be treated as categorical
    :return: dataframe containing the categorical features
    """
    # True if categorical, false otherwise
    is_categorical_mask = np.array([len(df[col].unique()) <= is_categorical_threshold for col in df
                                    if col not in exclude])

    return df[is_categorical_mask]


def get_features_from_name(df: pd.DataFrame, identifiers: list[str],
                           exclude: list[str] = None) -> pd.DataFrame:
    """
    Selects features based on their names.
    If they contain at least one of the strings provided, then the feature
    is selected, exception being if its name is contained in the exclusion list.

    :param df: dataframe to extract the categorical features from
    :param identifiers: strings contained in the name of a feature.
        If a feature name contains at least one of these, then the feature is selected.
    :param exclude: names of feature that should not be selected
    :return: dataframe containing the features
    """

    if exclude is None:
        exclude = []

    features = pd.DataFrame()
    for col in df:
        if any(id_ in col for id_ in identifiers) and col not in exclude:
            features[col] = df[col]

    return features


def simplify_categorical_features(df: pd.DataFrame,
                                  categorical_features: list[str] = None,
                                  simplification_threshold: float = 90,
                                  minimum_to_keep: int = 0,
                                  maximum_to_keep: int = None,
                                  simplify_with: object = "other") -> pd.DataFrame:
    """
    Simplifies the least frequent categories of all the categorical features in the provided dataframe
    into a single category with the specified value.

    :param df: dataframe to perform the simplification on.
    :param categorical_features: features to simplify.
        If None, all features in the dataframe will be simplified.
    :param simplification_threshold: percentage of explained values by the first n most frequent categories
        after which the m remaining categories are simplified into a single summary category.
        For example: if the threshold is 90 and there are 10 categories, and 7 categories account for
        90% of the values, the remaining 3 are substituted by a single category, lowering the
        total category number to 8.
    :param minimum_to_keep: minimum number of categories that must be kept before starting to simplify.
        This parameter makes sure that there are at least x original categories after the simplification
        has taken place, which leads to a minimum number of categories of x + 1.
    :param maximum_to_keep: maximum number of categories that can be kept before starting to simplify.
        If set to None, then there is no limit.
        For example, if this parameter is set to 10, and 10 unique categories are kept before reaching
        the simplification threshold, the remaining ones are replaced by a single category,
        which leads to a final number of categories of 11 (10 + 1).
        This is useful to force a low number of maximum categories.
    :param simplify_with: value to substitute the simplified categories with
    :return:
    """

    # Avoid side effects
    df = df.copy()

    # Argument check
    if maximum_to_keep is not None and maximum_to_keep < 0:
        raise Exception("The provided maximum number of categories are not valid!")

    if minimum_to_keep is None or minimum_to_keep < 0:
        raise Exception("The provided minimum number of categories are not valid!")

    if maximum_to_keep is not None and minimum_to_keep is not None:
        if maximum_to_keep <= minimum_to_keep:
            raise Exception("The provided minimum/maximum range of categories are not valid!")

    # Extract a dataframe containing the categorical columns
    if categorical_features is not None:
        categorical_features_df = df[categorical_features]
    else:
        categorical_features_df = df

    for feat in categorical_features_df:
        # Extract the unique categories from each categorical feature
        feat_to_simplify = categorical_features_df[feat]

        # produce a list of tuples that pairs each category with its frequency in percentage,
        # then sort it and extract the most frequent categories until the threshold is met
        cat_freq_pairs = get_value_frequency_pairs(feat_to_simplify)

        # sort the categories by frequency in desc order
        most_frequent = sorted(cat_freq_pairs, key=itemgetter(1), reverse=True)

        # if maximum_to_keep is None, there is no limit to how many
        # unique categories can be kept
        max_allowed = maximum_to_keep
        if max_allowed is None:
            max_allowed = len(most_frequent)

        # extract categories to summarize after one of the thresholds is passed
        to_replace = []
        current_pct = 0
        kept = 0
        for cat_freq_pair in most_frequent:
            # Categories can be simplified if the threshold has been met
            # or the number of kept categories exceeds the maximum,
            # only if at leas the minimum number of categories has been kept
            if (current_pct > simplification_threshold or kept == max_allowed) \
                    and kept > minimum_to_keep:
                category = cat_freq_pair[0]
                to_replace.append(category)
            else:
                kept += 1

            freq_pct = cat_freq_pair[1]
            current_pct += freq_pct

        # Replace the categories that need to be simplified
        simplified = feat_to_simplify.replace(to_replace=to_replace, value=simplify_with)
        debug = simplified.unique()

        # Replace the original column in the dataframe with the simplified one
        df[feat] = simplified

    return df


def get_value_frequency_pairs(from_: pd.Series) -> list[(object, float)]:
    """
    Returns a list of tuples that pairs each value with its frequency (expressed in %)
    in the provided pandas Series.

    :param from_: series to extract value frequencies from
    :return: list of value-frequency pairs
    """

    unique_values = from_.unique()
    value_freq_pairs = []
    for val in unique_values:
        freq_pct = (np.array(from_ == val).sum() / len(from_)) * 100
        value_freq_pairs.append((val, freq_pct))

    return value_freq_pairs


def compare_missing_info(col1: str, col2: str,
                         df: pd.DataFrame = None,
                         missing_info_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Returns a dataframe containing the necessary info to compare the
    missing values of the two specified columns.
    :param col1: name of the first column
    :param col2: name of the second column
    :param df: dataframe to compute the missing info from
    :param missing_info_df: dataframe containing info about the missing values of the two columns.
        If not None, this will be used to gather the necessary comparison information.
    :return:
    """

    if missing_info_df is None:
        missing_info_df = get_missing_info(df)

    columns = missing_info_df["column"]
    return missing_info_df[(columns == col1) | (columns == col2)]


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing information about the presence of missing values
    in the columns of the provided dataframe.

    :param df: dataframe to retrieve the info for
    :return: Dataframe with columns "column", "dtype", "missing count", "missing %"
    """

    columns = []
    dtypes = []
    missing_pct = []
    missing_num = []
    for col in df:
        columns.append(col)
        dtypes.append(df[col].dtype)

        n = df[col].isnull().sum()
        missing_num.append(n)

        pct = (n / len(df)) * 100
        missing_pct.append(pct)

    return pd.DataFrame(data={
        "column": columns,
        "dtype": dtypes,
        "missing count": missing_num,
        "missing %": missing_pct
    })


def compare_non_missing_info(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Returns the ratio of equal values between the two specified columns,
    excluding the rows where at least one of the two columns has a missing value.
    Comparison happens between values belonging to the same row.

    :param df: dataframe that contains the column to compare
    :param col1: first column to compare
    :param col2: second column to compare
    :return: equal values / total possible values
    """

    col1_not_na_mask = df[col1].notna()
    col1_not_na_number = np.sum(col1_not_na_mask)

    col2_not_na_mask = df[col2].notna()
    col2_not_na_number = np.sum(col2_not_na_mask)

    no_nans_df = df[col1_not_na_mask & col2_not_na_mask]
    no_nans_number = len(no_nans_df)

    # If there are no rows where both are non-missing, ratio is 0
    if no_nans_number != 0:
        equal_ratio = np.sum(no_nans_df[col1] == no_nans_df[col2]) / no_nans_number
    else:
        equal_ratio = 0

    return pd.DataFrame(data={
        f"{col1} non-missing": col1_not_na_number,
        f"{col2} non-missing": col2_not_na_number,
        "non-missing equal ratio": equal_ratio
    }, index=[0])


def get_unique_info_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing information about the number of unique
    values for each column of the provided dataframe.

    :param df: dataframe to retrieve the info for
    :return: Dataframe with columns "column", "dtype", "uniques count"
    """
    columns = []
    dtypes = []
    uniques = []
    for col in df:
        n_uniques = len(df[col].unique())

        columns.append(col)
        dtypes.append(df[col].dtype)
        uniques.append(n_uniques)

    return pd.DataFrame({
        "column": columns,
        "dtype": dtypes,
        "#uniques": uniques
    })


@utils.print_time_perf
def rfecv(estimator, X: pd.DataFrame, y: np.ndarray, subset_size: float = 1.0, rnd_seed: int = None,
          **rfecv_kwargs) -> RFECV:
    """

    :param estimator: estimator to perform RFECV with
    :param X: samples
    :param y: target
    :param subset_size: dimension of a subset of the provided data to actually utilize for the RFECV process.
        It is a float ranging from 0.0 to 1.0. Useful to speed up the process
    :param rnd_seed: seed used to make reproducible splits
    :param rfecv_kwargs: keyword args passed to the underlying RFECV instance.
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    :return: fitted RFECV instance
    """
    selector = RFECV(estimator, **rfecv_kwargs)

    small_sample = tr.TrainTestSplit.from_full_data(X, y, test_size=(1.0 - subset_size), rnd_seed=rnd_seed)
    selector.fit(small_sample.x_train.values, small_sample.y_train)

    return selector


@utils.print_time_perf
def rfe(estimator, X: pd.DataFrame, y: np.ndarray, subset_size: float = 1.0, rnd_seed: int = None,
        **rfe_kwargs) -> RFE:
    """

    :param estimator: estimator to perform RFE with
    :param X: samples
    :param y: target
    :param subset_size: dimension of a subset of the provided data to actually utilize for the RFECV process.
        It is a float ranging from 0.0 to 1.0. Useful to speed up the process
    :param rnd_seed: seed used to make reproducible splits
    :param rfe_kwargs: keyword args passed to the underlying RFE instance.
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    :return: fitted RFE instance
    """
    selector = RFE(estimator, **rfe_kwargs)

    small_sample = tr.TrainTestSplit.from_full_data(X, y, test_size=(1.0 - subset_size), rnd_seed=rnd_seed)
    selector.fit(small_sample.x_train.values, small_sample.y_train)

    return selector
