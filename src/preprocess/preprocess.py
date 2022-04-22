# Preprocess the dataset through:
# - renaming variables
# - cleaning up noisy values
# - applying expert pre-processing rules
# - filling in missing values through imputation

# TODO clean up imports
from src.preprocess import expert_impute, impute_column, rename_columns, time_to_numeric
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
from src.preprocess.get_solid_df import get_solid_df
from utils.get_timestamp import *
from utils.io import *
from src.preprocess import *
from sklearn.impute import *
from collections import Counter
import pandas as pd
import numpy as np
import json
import os

col_type_path = "data/metadata/col_types.json"
df_path = "data/AllTranTrainVal.csv"
metadata_path = "data/metadata/Metadata.xlsx"


# out_KNN_path = "data/preprocessed/AllTranTrainVal-KNNImputed.csv"
# out_RF_path = "data/preprocessed/AllTranTrainVal-RFImputed.csv"
very_solid_threshold = 0.05 # Max percentage of missing cells for a column to be considered very_solid
solid_threshold = 0.20 # Max percentage of missing cells for a column to be considered solid


def preprocess(debug_mode=False):

    # Read Dataset
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)

    # Column Renaming - add PRE/INT/POS prefix to column names
    rename_columns.rename_columns(df, df_metadata)

    # TODO Feature Engineering - consolidate and engineer new features
    engineer_features.engineer_features(df, df_metadata)

    if debug_mode:
        df = df.iloc[:10, :]
    else:
        # Dataset Cleansing - remove noisy values such as "n/a"
        cleansed_locs = cleanse_dataset.cleanse_dataset(df, df_metadata)
        # Expert Imputation - apply manual imputation rules
        imputed_locs = expert_impute.expert_impute(df, df_metadata)
        
    # TODO Visualize Changed Cells
    
    # Convert Time columns into Numeric columns
    df = time_to_numeric.time_to_numeric(df, df_metadata)
    
    # Get Solid DataFrame - remove columns that are too sparse
    my_print_header("Getting solid-only PRE columns to help impute the sparse columns.")
    solid_df = get_solid_df.get_solid_df(df, df_metadata, sparsity_threshold=solid_threshold)
    very_solid_df = get_solid_df.get_solid_df(df, df_metadata, sparsity_threshold=very_solid_threshold)
    
    # TODO ML Imputation - Apply KNN and Random Forest Imputers
    # df_RF = impute_missing.impute_missing_RF(df, df_metadata, solid_df, very_solid_df)
    # df_KNN = impute_column.impute_missing_KNN(df, df_metadata, solid_df, very_solid_df)
    result_holders = []
    df_preprocessed, result_holder = impute_column.impute_column(
        df,
        df_metadata,
        solid_df,
        very_solid_df,
        "PRE_img_size",
        debug_mode=debug_mode
        )
    result_holders.append(result_holder)
    save_experiment_df(df_preprocessed, "AllTranTrainVal-preprocessed.csv", "preprocessed csv file")
    save_experiment_pickle(result_holders, "GridSearchResults.pkl", "KNN and RF imputation hyperparamter search")