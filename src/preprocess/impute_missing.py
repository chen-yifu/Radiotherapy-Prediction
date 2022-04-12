from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from utils.printers import *
from tqdm import tqdm
df_compare_path = "data/output/AllTranTrainVal-CompareImpute.csv"

def impute(imputer, df):
    new_df = pd.DataFrame(imputer.fit_transform(df))
    # print("NEWDF", new_df.columns)
    # print("DF", df.columns)
    # print(set(new_df.columns).difference(set(df.columns)))
    new_df.columns = df.columns
    new_df.index = df.index
    return new_df

def impute_missing_KNN(df, df_metadata, solid_df, very_solid_df):
    # Use the very_solid_df to impute all columns that contain missing cells
    
    my_print_header("Imputing missing values using KNN Imputer...")
        # Return the optimal KNN imputer for this dataset
    print("Performing KNN grid search to find the optimal n_neighbors for KNN Imputer.")
    metrics = {} # Mapping schema: {column1: {accuracy: float, F1: float, ...}}
    
    # TODO: Standardize the numeric columns of DataFrame
    # TODO: for the categorical ones create separate columns for each category (use one-hot/standard function)
    
    for col in tqdm(df.columns):
    # col = "PRE_img_size"
        print(f"Imputing {col}")
        temp_df = very_solid_df.copy(deep=True)
        temp_df = temp_df.drop("PRE_record_id", axis=1)
        if col not in temp_df.columns:
            temp_df[col] = df[col]
            
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
                    cur_gt = temp_df.loc[i, col]
                    if str(cur_gt) == "nan":
                        continue
                    temp_df.loc[i, col] = np.nan
                    imp_df = impute(KNN, temp_df)
                    cur_imp = imp_df.loc[i, col]
                    total_sq_error += (cur_imp - cur_gt)**2
                    total_count += 1
                    temp_df.loc[i, col] = cur_gt # Restore ground truth of this cell
                    compares[(i, col)] = [cur_gt, cur_imp]
                except ValueError as e:
                    # TODO solve try-except block about length mismatch 44/45
                    print(f"Skipped row {i}")
                    continue
            
            rmse = (total_sq_error/total_count) ** 0.5
            rmse = round(rmse, 5)
            if rmse < best_rmse:
                best_n = n_neighbors
                best_rmse = rmse
                compares  = compares
            print(f"n_neighbors = {n_neighbors}. RMSE = {rmse} | Best n_neighbors = {best_n}. Best RMSE = {best_rmse}")
    return compares

# def RF_grid_search(df, df_metadata, solid_df, very_solid_df, shallow=True):
#     # Return the optimal RF imputer for this dataset
#     # If shallow, then max_depth ≤ 5 and n_estimators ≤ 10
#     my_print_header("Performing Random Forest grid search to find the optimal max_depth and n_estimators for Random Forest Imputer.")
    
#     return


def impute_missing_RF(df, df_metadata, solid_df, very_solid_df):
    # Use the very_solid_df to impute all columns that contain missing cells
    
    my_print_header("Imputing missing values using RF Imputer...")
        # Return the optimal KNN imputer for this dataset
    print("Performing RF grid search to find the optimal max_depth and n_estimators for RF Imputer.")
    metrics = {} # Mapping schema: {column1: {accuracy: float, F1: float, ...}}
    
    # TODO: for col in tqdm(df.columns):
    col = "PRE_img_size"
    temp_df = very_solid_df.copy(deep=True)
    temp_df = temp_df.drop("PRE_record_id", axis=1)
    if col not in temp_df.columns:
        temp_df[col] = df[col]
    # temp_df = temp_df[temp_df[col].notna()] # Drop rows with nan in column # TODO ennsure nans are dropped
    # Perform grid-search to find the optimal n_neighbors for KNN Imputer
    best_rmse = float("inf")
    best_max_depth = 0
    best_n = 0
    for max_depth in range(3, 15, 2):
        for n_estimators in range(3, 20, 2):
            kf = KFold(n_splits=5, shuffle=True, random_state=max_depth*1000+n_estimators)
            compares = {} # Mapping schema: {column1: {gt: int}, ...}
            total_sq_error = 0
            total_count = 0
            RF = IterativeImputer(estimator=RandomForestRegressor(max_depth=5,n_estimators=10))
            for i, (train_index, test_index) in enumerate(kf.split(temp_df)):
                try:
                    # Temporarily set cells to nan, and impute
                    cur_gt = temp_df.loc[test_index, col]
                    temp_df.loc[test_index, col] = np.nan
                    imp_df = impute(RF, temp_df)
                    cur_imp = imp_df.loc[test_index, col]
                    for temp_gt, temp_imp in zip(cur_gt, cur_imp):
                        if str(temp_gt) == "nan": continue
                        total_sq_error += (temp_gt - temp_imp)**2
                        total_count += 1
                        temp_df.loc[test_index, col] = cur_gt # Restore ground truth of this cell
                    for idx in test_index:
                        compares[(idx, col)] = [cur_gt, cur_imp]
                except ValueError as e:
                    # TODO solve try-except block about length mismatch 44/45
                    print(f"Skipped split {i}")
                    continue
            
            rmse = round((total_sq_error/total_count) ** 0.5, 5)
            if rmse < best_rmse:
                best_n = n_estimators
                best_max_depth = max_depth
                best_rmse = rmse
                compares  = compares
            print(
                f"max_depth = {max_depth}, n_estimators = {n_estimators}. RMSE = {rmse} |\
 Best n_estimators = {best_n}, max_depth = {best_max_depth}. Best RMSE = {best_rmse}")
    return compares



