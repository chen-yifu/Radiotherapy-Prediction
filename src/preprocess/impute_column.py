from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.IO import *
from utils.get_timestamp import *
from tqdm import tqdm

df_compare_path = f"data/output/{get_timestamp()}/AllTranTrainVal-CompareImpute.csv"
hyperparameter_path = f"data/output/"

class ColumnGridSearchResults:
    # The class to store the results of grid-search for a column
    # Provides functions to get the best model, and the best hyperparameters
    def __init__(self, column_name):
        # Metric mapping schema:
        # {column1: 
        #   {hyperparameter_setting1: 
        #       {error: float,
        #       accuracy: float, 
        #       F1: float,
        #       compares: [(ground truth, imputed), (ground truth, imputed), ...]},
        #   hyperparameter_setting2:
        #       {error: float,
        #       accuracy: float,
        #       F1:float,
        #       compares: [(ground truth, imputed), (ground truth, imputed), ...]},
        #   ...
        #   }    
        # },
        # {column 2: ...}
        self.column_name = column_name
        self.KNN_metrics = defaultdict(lambda: defaultdict(float))
        self.RF_metrics  = defaultdict(lambda: defaultdict(float))

    def get_best_model(self, metric_name):
        # Returns the best model and the best hyperparameters for this model
        if metric_name == "accuracy":
            # TODO ...
            pass
        elif metric_name == "F1":
            # TODO implement get best model by F1
            pass
        

def impute(imputer, df):
    new_df = pd.DataFrame(imputer.fit_transform(df))
    new_df.columns = df.columns
    new_df.index = df.index
    return new_df

def impute_missing_KNN(df, df_metadata, solid_df, very_solid_df, result_holder, column_name):
    # Use the very_solid_df to impute all columns that contain missing cells
    # my_print_header("Imputing missing values using KNN Imputer...")
        # Return the optimal KNN imputer for this dataset
    my_print_header("Performing KNN grid search to find the optimal n_neighbors for KNN Imputer.")
    
    # TODO: Standardize the numeric columns of DataFrame
    # TODO: for the categorical ones create separate columns for each category (use one-hot/standard function)
    
    # for col in tqdm(df.columns):
    my_print(f"Imputing {column_name}", plain=True)
    temp_df = very_solid_df.copy(deep=True)
    temp_df = temp_df.drop("PRE_record_id", axis=1)
    if column_name not in temp_df.columns:
        temp_df[column_name] = df[column_name]
        
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_n = 0
    for n_neighbors in range(1, 100, 2):
        compares = {} # Mapping schema: {column1: {gt: int}, ...}
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
                temp_df.loc[i, column_name] = cur_gt # Restore ground truth of this cell
                compares[(i, column_name)] = [cur_gt, cur_imp]
            except ValueError as e:
                # TODO solve try-except block about length mismatch 44/45
                # my_print(f"Skipped row {i}")
                raise e
                continue
        
        rmse = (total_sq_error/total_count) ** 0.5
        rmse = round(rmse, 5)
        if rmse < best_rmse:
            best_n = n_neighbors
            best_rmse = rmse
            compares  = compares
        hyperparameter = frozenset({"n_neighbors": n_neighbors})
        result_holder.KNN_metrics[hyperparameter] = {"rmse": rmse, "compares": compares}
        my_print(
                f"n_neighbors = {n_neighbors}. RMSE = {rmse} "
                f"| Best n_neighbors = {best_n}. Best RMSE = {best_rmse}",
                 plain=True
                )
        # {column1: 
        #   {hyperparameter_setting1: 
        #       {error: float,
        #       accuracy: float, 
        #       F1: float},
    # return compares

# def RF_grid_search(df, df_metadata, solid_df, very_solid_df, shallow=True):
#     # Return the optimal RF imputer for this dataset
#     # If shallow, then max_depth ≤ 5 and n_estimators ≤ 10
#     my_print_header("Performing Random Forest grid search to find the optimal max_depth and n_estimators for Random Forest Imputer.")
    
#     return


def impute_missing_RF(df, df_metadata, solid_df, very_solid_df, result_holder, column_name):
    # Use the very_solid_df to impute all columns that contain missing cells
        # Return the optimal KNN imputer for this dataset
        
    my_print_header("Performing RF grid search to find the optimal max_depth and n_estimators for RF Imputer.")
    
    # TODO: for col in tqdm(df.columns):
    temp_df = very_solid_df.copy(deep=True)
    temp_df = temp_df.drop("PRE_record_id", axis=1)
    if column_name not in temp_df.columns:
        temp_df[column_name] = df[column_name]
    # temp_df = temp_df[temp_df[col].notna()] # Drop rows with nan in column # TODO ennsure nans are dropped
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_max_depth = 0
    best_n = 0
    for max_depth in range(1, 20, 2):
        for n_estimators in range(1, 20, 2):
            kf = KFold(n_splits=5, shuffle=True, random_state=max_depth*1000+n_estimators)
            compares = {} # Mapping schema: {column1: {gt: int}, ...}
            total_sq_error = 0
            total_count = 0
            RF = IterativeImputer(estimator=RandomForestRegressor(max_depth=5,n_estimators=10))
            for i, (train_index, test_index) in enumerate(kf.split(temp_df)):
                try:
                    # Temporarily set cells to nan, and impute
                    cur_gt = temp_df.loc[test_index, column_name]
                    temp_df.loc[test_index, column_name] = np.nan
                    imp_df = impute(RF, temp_df)
                    cur_imp = imp_df.loc[test_index, column_name]
                    for temp_gt, temp_imp in zip(cur_gt, cur_imp):
                        if str(temp_gt) == "nan": continue
                        total_sq_error += (temp_gt - temp_imp)**2
                        total_count += 1
                        temp_df.loc[test_index, column_name] = cur_gt # Restore ground truth of this cell
                    for idx in test_index:
                        compares[(idx, column_name)] = [cur_gt, cur_imp]
                except ValueError as e:
                    # TODO solve try-except block about length mismatch 44/45
                    my_print(f"Skipped split {i}", plain=True)
                    continue
            
            rmse = round((total_sq_error/total_count) ** 0.5, 5)
            if rmse < best_rmse:
                best_n = n_estimators
                best_max_depth = max_depth
                best_rmse = rmse
                compares  = compares
            my_print(
                f"n_estimators = {n_estimators}, max_depth = {max_depth}. RMSE = {rmse} |"
                f"Best n_estimators = {best_n}, max_depth = {best_max_depth}. Best RMSE = {best_rmse}",
                plain=True
            )
            hyperparameter = frozenset({"n_estimators": n_estimators, "max_depth": max_depth})
            result_holder.RF_metrics[hyperparameter] = {"rmse": rmse, "compares": compares}
    # return compares


def impute_column(df, df_metadata, solid_df, very_solid_df, column_name):
    
    my_print_header(f"Performing RF and KNN grid searches to find the best imputer for {column_name}...")
    
    result_holder = ColumnGridSearchResults(column_name)
    
    # impute_missing_RF(df, df_metadata, solid_df, very_solid_df, result_holder, column_name)
    impute_missing_KNN(df, df_metadata, solid_df, very_solid_df, result_holder, column_name)
    
    return df, result_holder

