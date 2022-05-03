from typing import Dict, List, OrderedDict, Tuple, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn import preprocessing
# from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import re
import pickle
from collections import defaultdict
from utils.io import my_print_header, my_print
from utils.get_timestamp import get_timestamp
import utils.config as config

# df_compare_path = f"data/output/{get_timestamp()}/Impute_Comparison.csv"


class ColumnGridSearchResults:
    """
    An object to save the results of grid-search for a column
    Also provides functions to get the best model, and the best hyperparameters

    Metric mapping schema:
    {KNN:
        {hyperparameter_setting1:
            {
                rmse: float,
                accuracy: float,
                F1: float,
                compares: [(g truth, imputed), (g truth, imputed), ...]
                model_pickle_path: str
            },
        hyperparameter_setting2:
            {
                rmse: float,
                accuracy: float,
                F1:float,
                compares: [(g truth, imputed), (g truth, imputed), ...]
                model_pickle_path: str
            },
        ...
        }
    },
    {RF: ...}
    """
    def __init__(self, column_name):
        self.column_name = column_name
        self.metrics = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
            )
        self.model_names = set()

    def add_compare(
        self,
        model_name: str,
        hyperparameter_setting: tuple,
        gtruth: Union[str, float, int],
        imputed: Union[str, float, int]
    ) -> None:
        """Add the (ground truth, imputed) tuple to the compares field

        Args:
            model_name (str): name of model, e.g., RF or KNN
            hyperparameter_setting (tuple): hyperparameter setting
            gtruth (Union[str, float, int]): original value in DataFrame
            imputed (Union[str, float, int]): imputed value by model
        """
        self.model_names.add(model_name)
        if type(imputed) == float:
            imputed = round(imputed, 4)
        tupl = (gtruth, imputed)
        holder = self.metrics[model_name][hyperparameter_setting]
        compares = holder["compares"]
        if not compares or type(compares) == float:
            holder["compares"] = [tupl]
        else:
            holder["compares"].append(tupl)

    def show_compare(
        self,
        model_name: str,
        hyperparameter_setting: tuple
    ) -> List[tuple]:
        """Given a model name and hyperparameter setting, return the compares

        Args:
            model_name (str): name of model, e.g., RF or KNN
            hyperparameter_setting (tuple): hyperparameter setting
        """
        compares = self.metrics[model_name][hyperparameter_setting]["compares"]
        my_print(f"Comparing gt-impute {model_name} {hyperparameter_setting}:")
        print(compares)

    def calc_metrics(
        self,
        model_name: str,
        hyperparameter_setting: tuple,
        metric: str
    ) -> None:
        """Calculate the metrics for a given model and hyperparameter setting

        Args:
            model_name (str): name of model, e.g., RF or KNN
            hyperparameter_setting (tuple): hyperparameter setting
            metric (str): RMSE, F1, ACCURACY
        """
        holder = self.metrics[model_name][hyperparameter_setting]
        compares = holder["compares"]
        if type(compares) == float:
            print("No 'compares' to calculate metrics")
            return -1
        # Calculate RMSE of the imputation
        # TODO implement metric calculation for accuracy, F1, rmse
        if metric == "rmse":
            total_sq_error, total_count = 0, 0
            for gt, imp in compares:
                total_sq_error += (gt - imp) ** 2
                total_count += 1
            rmse = (total_sq_error/total_count) ** 0.5
            rmse = round(rmse, 4)
            holder["rmse"] = rmse
            return rmse
        elif metric == "accuracy":
            # TODO
            pass
        elif metric == "F1":
            # TODO
            pass

    def get_best_model(
        self,
        metric_name: str
    ) -> Tuple[object, str, tuple]:
        """"Returns the best model with the best hyperparameters for metric

        Args:
            metric_name (str): the metric used to select the best model
        """
        all_hypers = [(hyper_dict, name) for name in self.model_names
                      for hyper_dict in self.metrics[name]]

        # Take the minimum of metric among all hyperparameters of all models
        (hyper_dict, name) = min(
            all_hypers,
            key=lambda x: self.metrics[x[1]][x[0]][metric_name]
        )
        # Load best model
        path = self.metrics[name][hyper_dict]["model_pickle_path"]
        with open(path, "rb") as f:
            model = pickle.load(f)
            return model, name, hyper_dict

    def save_model(
        self,
        model_name: str,
        model: object,
        hyperparam: tuple
    ) -> str:
        """Save a model to pickle file, and return its path

        Args:
            model_name (str): type of model, e.g., KNN or RF
            model (object): the model to save
            hyperparam (tuple): the hyperparameters of the model

        Returns:
            str: path to the saved model
        """
        experiment_dir = config.experiment_dir
        # regex demo: https://regex101.com/r/u3WSmc/1
        hyper_str = re.sub(
            r"\(|\)|,| |{|}|'",
            "",
            str(hyperparam).replace("), (", "-"))
        save_dir = os.path.join(experiment_dir, "models", self.column_name)
        save_path = os.path.join(save_dir, f"{model_name}-{hyper_str}.pkl")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        self.metrics[model_name][hyperparam]["model_pickle_path"] = save_path
        return save_path

    def impute_using_best_model(
        self,
        df: pd.DataFrame,
        metric: str
    ) -> pd.DataFrame:
        """Impute the missing values using the best model

        Args:
            df (pd.DataFrame): the dataframe to impute
            metric (str): the metric to select the best model
        """
        model = self.get_best_model(metric)
        imputed_df = impute(model, df)
        return imputed_df

    def save_best_models(self):
        # TODO
        pass

    def load_best_models(self) -> Dict:
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
    imputed_df = imputer.fit_transform(df)
    # print("Shape of original df", df.shape)
    # print("Shape of imputed df", imputed_df.shape)
    new_df = pd.DataFrame(
        imputed_df,
        columns=df.columns
        # index=df.index
        )
    return new_df


def find_best_KNN(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    base_cols_df: pd.DataFrame,
    result_holder: ColumnGridSearchResults,
    column_name: str,
    standardize: bool = True,
    n_neighbors_range: range = range(1, 100, 2)
):
    """Grid-search to find the optimal hyperparameters, and impute using KNN.
    Args:
        df (pd.DataFrame): The dataframe to impute
        df_metadata (pd.DataFrame): The metadata of the dataframe
        base_cols_df (pd.DataFrame): The DataFrame with the columns to use
        result_holder (ColumnGridSearchResults): Object to store the results
        column_name (str): The name of the column to impute
        standardize (bool, optional): Whether standarsize each feature
    Returns:
        KNNImputer: the optimal KNN imputer
    """

    my_print(f"KNN: search best n_neighbors over {n_neighbors_range}.")

    # TODO: for the categorical ones create separate columns for each category?
    #       (use one-hot/standard function)

    my_print(f"Imputing {column_name}", plain=True)
    # Keep only numeric columns (exclude the patient IDs, which is string type)
    base_cols_df = base_cols_df.drop("PRE_record_id", axis=1)
    # Standardize the numeric columns of DataFrame
    if standardize:
        scaler = preprocessing.StandardScaler().fit(base_cols_df)
        base_cols_df = pd.DataFrame(
            scaler.transform(base_cols_df),
            columns=base_cols_df.columns
            )
    temp_df = base_cols_df.copy(deep=True)
    if column_name not in temp_df.columns:
        temp_df[column_name] = df[column_name]
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_n = 0
    for n_neighbors in n_neighbors_range:
        KNN = KNNImputer(n_neighbors=n_neighbors)
        hyperparameter = tuple(
            OrderedDict({"n_neighbors": n_neighbors}).items()
        )
        for i in range(0, len(df)):
            try:
                # Temporarily set a cell to nan, and impute
                cur_gt = temp_df.loc[i, column_name]
                if str(cur_gt) == "nan":
                    continue
                temp_df.loc[i, column_name] = np.nan
                imp_df = impute(KNN, temp_df)
                cur_imp = imp_df.loc[i, column_name]
                temp_df.loc[i, column_name] = cur_gt  # Restore ground truth
                result_holder.add_compare(
                    "KNN",
                    hyperparameter,
                    cur_gt,
                    cur_imp
                    )
            except Exception as e:
                # print("ERROR", e)
                my_print(f"Skipped row {i} due to an error.")
                # continue
                raise e
        rmse = result_holder.calc_metrics("KNN", hyperparameter, "rmse")
        if rmse < best_rmse:
            best_n = n_neighbors
            best_rmse = rmse

        result_holder.save_model("KNN", KNN, hyperparameter)
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
    base_cols_df: pd.DataFrame,
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
        base_cols_df (pd.DataFrame): The DataFrame with the columns to use
        result_holder (ColumnGridSearchResults): The object with results
        column_name (str): The name of the column to impute
        max_depth_range (range): The range of max_depth to search over
        n_estimators_range (range): The range of n_estimators to search over
    Returns:
        IterativeImputer: the optimal RF imputer (wrapped by IterativeImputer)
    """
    my_print(f"RF: search best max_depth over {max_depth_range}"
             f", and n_estimators {n_estimators_range}.")

    # TODO: for col in tqdm(df.columns):
    temp_df = base_cols_df.copy(deep=True)
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
                n_splits=3,
                shuffle=True,
                random_state=0)
            hyperparameter = tuple(sorted({
                "n_estimators": n_estimators,
                "max_depth": max_depth
                }.items(), key=lambda x: x[0]))
            RF = IterativeImputer(estimator=RandomForestRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=0))
            for i, (train_index, test_index) in enumerate(kf.split(temp_df)):
                # print("BEFORE")  #, temp_df.isna().sum())
                # print(temp_df)
                try:
                    # Temporarily set cells in fold to nan, and impute
                    cur_gt = temp_df.loc[test_index, column_name]
                    if str(cur_gt) == "nan":
                        continue
                    temp_df.loc[test_index, column_name] = np.nan
                    # print("DURING")  #, temp_df.isna().sum())
                    # print(temp_df)
                    # If the entire column is nan, skip
                    if temp_df[column_name].isna().sum() == len(temp_df):
                        print(f"Skipping fold {i} due to all nan.")
                        continue
                    imp_df = impute(RF, temp_df)
                    cur_imp = imp_df.loc[test_index, column_name]
                    # Restore ground truth
                    temp_df.loc[test_index, column_name] = cur_gt
                    for imp, gt in zip(cur_imp, cur_gt):
                        if str(gt) == "nan":
                            continue
                        result_holder.add_compare(
                            "RF",
                            hyperparameter,
                            gt,
                            imp
                            )
                        # print(f"Added compare for gt {gt} and imp {imp}")
                    # print("AFTER")  #, imp_df.isna().sum())
                    # print(imp_df)
                except ValueError as e:
                    print(f"Skipped row {i} due to an error: {e}")
                    if config.debug_mode:
                        raise e
                    continue

            rmse = result_holder.calc_metrics("RF", hyperparameter, "rmse")
            if rmse < best_rmse:
                best_n = n_estimators
                best_max_depth = max_depth
                best_rmse = rmse
            result_holder.save_model("RF", RF, hyperparameter)
            my_print(
                f"n_estimators = {n_estimators}, max_depth = {max_depth}."
                f" RMSE = {rmse} |"
                f" Best n_estimators = {best_n}, max_depth = {best_max_depth}."
                f" Best RMSE = {best_rmse}",
                plain=True
            )

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
    target_metric: str
        ) -> Tuple[pd.DataFrame, ColumnGridSearchResults]:
    """Optimize KNN and RF Imputers, and use the best model to impute the column.

    Args:
        df (pd.DataFrame): original DataFrame containing missing values
        df_metadata (pd.DataFrame): metadata DataFrame
        solid_df (pd.DataFrame): original DataFrame w/ solid columns only
        very_solid_df (pd.DataFrame): original DataFrame w/ very solid columns
        column_name (str): the column to impute
        target_metric (str): the metric to optimize (F1, accuracy, rmse)

    Returns:
        Tuple[pd.DataFrame, ColumnGridSearchResults]: Imputed DataFrame and
            results of grid search
    """
    my_print(f"Optimizing for the best imputer for {column_name}...")

    result_holder = ColumnGridSearchResults(column_name)

    if config.debug_mode:
        KNN_n_neighbors_range = range(1, 5, 1)
        RF_n_estimators_range = range(1, 3, 1)
        RF_max_depth_range = range(1, 3, 1)
    else:
        KNN_n_neighbors_range = range(1, 50, 3)
        RF_n_estimators_range = range(10, 351, 40)  # range(1, 15, 3)
        RF_max_depth_range = range(1, 11, 3)  # range(1, 15, 3)

    find_best_KNN(
        df,
        df_metadata,
        very_solid_df,
        result_holder,
        column_name,
        n_neighbors_range=KNN_n_neighbors_range
        )

    find_best_RF(
        df,
        df_metadata,
        very_solid_df,
        result_holder,
        column_name,
        max_depth_range=RF_max_depth_range,
        n_estimators_range=RF_n_estimators_range
        )

    # Use the best model to impute this column
    best_model, name, hyperparam = result_holder.get_best_model(target_metric)
    my_print(f"The model with best {target_metric} is {name} {hyperparam}.")
    result_holder.show_compare(name, hyperparam)
    print(f"Imputing {column_name} using best model...")
    imputed_df = result_holder.impute_using_best_model(df, best_model)

    return imputed_df, result_holder
