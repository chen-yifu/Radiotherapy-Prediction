from typing import Tuple
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.io import my_print_header, my_print
from utils.get_timestamp import get_timestamp

df_compare_path = f"data/output/{get_timestamp()}/Impute_Comparison.csv"
hyperparameter_path = "data/output/"


class ColumnGridSearchResults:
    """
    An object to save the results of grid-search for a column
    Also provides functions to get the best model, and the best hyperparameters

    Metric mapping schema:
    {KNN_metrics:
        {hyperparameter_setting1:
            {error: float,
            accuracy: float,
            F1: float,
            compares: [(ground truth, imputed), (ground truth, imputed), ...]},
        hyperparameter_setting2:
            {error: float,
            accuracy: float,
            F1:float,
            compares: [(ground truth, imputed), (ground truth, imputed), ...]},
        ...
        }
    },
    {RF_metrics: ...}"""
    def __init__(self, column_name):
        self.column_name = column_name
        self.metrics = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
            )

    def get_best_model(self, metric_name):
        "Returns the best model and the best hyperparameters for this model"
        if metric_name == "accuracy":
            # TODO ...
            pass
        elif metric_name == "F1":
            # TODO implement get best model by F1
            pass

    def save_best_model(self, metric_name):
        # TODO
        pass

    def load_best_model(self, metric_name):
        # TODO
        pass


def impute(imputer, df: pd.DataFrame):
    """Perform imputation on a dataframe using the given imputer

    Args:
        imputer (sklearn.Impute): the imputer to use
        df (pd.DataFrame): the dataframe to impute

    Returns:
        pd.DataFrame: the imputed dataframe
    """
    new_df = pd.DataFrame(imputer.fit_transform(df))
    new_df.columns = df.columns
    new_df.index = df.index
    return new_df


def find_best_KNN(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    solid_df: pd.DataFrame,
    very_solid_df: pd.DataFrame,
    result_holder: ColumnGridSearchResults,
    column_name: str,
    standardize: bool = True,
    n_neighbors_range: range = range(1, 100, 2)
):
    """Grid-search to find the optimal hyperparameters, and impute using KNN.
    Args:
        df (pd.DataFrame): The dataframe to impute
        df_metadata (pd.DataFrame): The metadata of the dataframe
        solid_df (pd.DataFrame): The solid (low sparsity) dataframe
        very_solid_df (pd.DataFrame): The very low sparsity dataframe
        result_holder (ColumnGridSearchResults): Object to store the results
        column_name (str): The name of the column to impute
        standardize (bool, optional): Whether standarsize euclidean distance
    Returns:
        KNNImputer: the optimal KNN imputer
    """

    my_print_header(f"KNN: search best n_neighbors over {n_neighbors_range}.")

    # TODO: Standardize the numeric columns of DataFrame
    # TODO: for the categorical ones create separate columns for each category (use one-hot/standard function)

    my_print(f"Imputing {column_name}", plain=True)
    temp_df = very_solid_df.copy(deep=True)
    temp_df = temp_df.drop("PRE_record_id", axis=1)
    if column_name not in temp_df.columns:
        temp_df[column_name] = df[column_name]
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_n = 0
    for n_neighbors in n_neighbors_range:
        compares = {}  # Mapping schema: {column1: {gt: int}, ...}
        total_sq_error = 0
        total_count = 0
        KNN = KNNImputer(n_neighbors=n_neighbors)
        for i in range(0, len(df)):
            try:
                # Temporarily set a cell to nan, and impute
                cur_gt = temp_df.loc[i, column_name]
                if str(cur_gt) == "nan":
                    continue
                temp_df.loc[i, column_name] = np.nan
                imp_df = impute(KNN, temp_df)
                cur_imp = imp_df.loc[i, column_name]
                total_sq_error += (cur_imp - cur_gt)**2
                total_count += 1
                temp_df.loc[i, column_name] = cur_gt  # Restore ground truth
                compares[(i, column_name)] = {"g truth": cur_gt,
                                              "imputed": cur_imp}
            except ValueError:
                # TODO solve try-except block about length mismatch 44/45
                my_print(f"Skipped row {i} due to an error.")
                continue

        rmse = (total_sq_error/total_count) ** 0.5
        rmse = round(rmse, 5)
        if rmse < best_rmse:
            best_n = n_neighbors
            best_rmse = rmse
            compares = compares
        hyperparameter = frozenset({"n_neighbors": n_neighbors})
        result_holder.metrics["K Nearest Neighbors"][hyperparameter] = {
            "rmse": rmse,
            "compares": compares
            }
        my_print(
                f"n_neighbors = {n_neighbors}. RMSE = {rmse} "
                f"| Best n_neighbors = {best_n}. Best RMSE = {best_rmse}",
                plain=True
                )
    my_print(f"KNN: Best n_neighbors = {best_n}")
    return KNNImputer(n_neighbors=best_n)


def find_best_RF(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    solid_df: pd.DataFrame,
    very_solid_df: pd.DataFrame,
    result_holder: ColumnGridSearchResults,
    column_name: str,
    max_depth_range: range = range(1, 20, 2),
    n_estimators_range: range = range(1, 20, 2)
):
    """Search the optimal Random Forest hyperparameters, and impute the column.
    Args:
        df (pd.DataFrame): The dataframe to impute
        df_metadata (pd.DataFrame): The metadata of the dataframe
        solid_df (pd.DataFrame): The solid (low sparsity) dataframe
        very_solid_df (pd.DataFrame): The very solid dataframe
        result_holder (ColumnGridSearchResults): The object with results
        column_name (str): The name of the column to impute
        max_depth_range (range): The range of max_depth to search over
        n_estimators_range (range): The range of n_estimators to search over
    Returns:
        IterativeImputer: the optimal RF imputer (wrapped by IterativeImputer)
    """
    my_print_header(f"RF: search best max_depth over {max_depth_range}"
                    f", and n_estimators {n_estimators_range}.")

    # TODO: for col in tqdm(df.columns):
    temp_df = very_solid_df.copy(deep=True)
    temp_df = temp_df.drop("PRE_record_id", axis=1)
    if column_name not in temp_df.columns:
        temp_df[column_name] = df[column_name]
    # temp_df = temp_df[temp_df[col].notna()] # Drop rows with nan in column
    # # TODO ennsure nans are dropped
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_max_depth = 0
    best_n = 0
    for max_depth in max_depth_range:
        for n_estimators in n_estimators_range:
            kf = KFold(
                n_splits=5,
                shuffle=True)
            compares = {}  # Mapping schema: {column1: {gt: int}, ...}
            total_sq_error = 0
            total_count = 0
            RF = IterativeImputer(estimator=RandomForestRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators))
            for i, (train_index, test_index) in enumerate(kf.split(temp_df)):
                try:
                    # Temporarily set cells to nan, and impute
                    cur_gt = temp_df.loc[test_index, column_name]
                    temp_df.loc[test_index, column_name] = np.nan
                    imp_df = impute(RF, temp_df)
                    cur_imp = imp_df.loc[test_index, column_name]
                    for temp_gt, temp_imp in zip(cur_gt, cur_imp):
                        if str(temp_gt) == "nan":
                            continue
                        total_sq_error += (temp_gt - temp_imp)**2
                        total_count += 1
                    # Restore ground truth of this column's fold
                    temp_df.loc[test_index, column_name] = cur_gt
                    for idx in test_index:
                        compares[(idx, column_name)] = {"g truth": cur_gt,
                                                        "imputed": cur_imp}
                except ValueError as e:
                    raise e

            rmse = round((total_sq_error/total_count) ** 0.5, 5)
            if rmse < best_rmse:
                best_n = n_estimators
                best_max_depth = max_depth
                best_rmse = rmse
                compares = compares
            my_print(
                f"n_estimators = {n_estimators}, max_depth = {max_depth}."
                f"RMSE = {rmse} |"
                f"Best n_estimators = {best_n}, max_depth = {best_max_depth}."
                f"Best RMSE = {best_rmse}",
                plain=True
            )
            hyperparameter = frozenset({
                "n_estimators": n_estimators,
                "max_depth": max_depth
                })
            result_holder.metrics["Random Forest"][hyperparameter] = {
                "rmse": rmse,
                "compares": compares
                }

    my_print(f"Best RF: n_estimators = {best_n}, max_depth = {best_max_depth}")

    best_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                max_depth=best_max_depth,
                n_estimators=best_n))

    return best_imputer


def impute_column(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    solid_df: pd.DataFrame,
    very_solid_df: pd.DataFrame,
    column_name: str,
    debug_mode: bool = False
        ) -> Tuple[pd.DataFrame, ColumnGridSearchResults]:
    """Optimize KNN and RF Imputers, and use the best model to impute the column.

    Args:
        df (pd.DataFrame): original DataFrame containing missing values
        df_metadata (pd.DataFrame): metadata DataFrame
        solid_df (pd.DataFrame): original DataFrame w/ solid columns only
        very_solid_df (pd.DataFrame): original DataFrame w/ very solid columns
        column_name (str): the column to impute
        debug_mode (bool, optional): Whether run in debug mode to save time.

    Returns:
        Tuple[pd.DataFrame, ColumnGridSearchResults]: Imputed DataFrame and
            results of grid search
    """
    my_print_header(f"Optimizing for the best imputer for {column_name}...")

    result_holder = ColumnGridSearchResults(column_name)

    if debug_mode:
        KNN_n_neighbors_range = range(1, 10, 2)
        RF_n_estimators_range = range(1, 5, 2)
        RF_max_depth_range = range(1, 5, 2)
    else:
        KNN_n_neighbors_range = range(1, 100, 5)
        RF_n_estimators_range = range(10, 211, 40)  # range(1, 15, 3)
        RF_max_depth_range = range(1, 15, 3)  # range(1, 15, 3)

    find_best_KNN(
        df,
        df_metadata,
        solid_df,
        very_solid_df,
        result_holder,
        column_name,
        n_neighbors_range=KNN_n_neighbors_range
        )
    find_best_RF(
        df,
        df_metadata,
        solid_df,
        very_solid_df,
        result_holder,
        column_name,
        max_depth_range=RF_max_depth_range,
        n_estimators_range=RF_n_estimators_range
        )

    return df, result_holder
