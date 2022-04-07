# Preprocess the dataset through:
# - renaming variables
# - cleaning up noisy values
# - applying expert pre-processing rules
# - filling in missing values through imputation

# TODO clean up imports
from src.preprocess import expert_impute, rename_columns, impute_missing, time_to_numeric
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
from src.preprocess.get_solid_df import get_solid_df
from utils.printers import *
from utils.loaders import *
from src.preprocess import *
import pandas as pd
from sklearn.impute import *
import numpy as np
from collections import Counter
import json, pickle

col_type_path = "data/metadata/col_types.json"
df_path = "data/AllTranTrainVal.csv"
metadata_path = "data/metadata/Metadata.xlsx"
out_path = "data/output/AllTranTrainVal-Preprocessed.csv"
out_path_KNN = "data/output/AllTranTrainVal-KNNImputed.csv"
out_path_RF = "data/output/AllTranTrainVal-RFImputed.csv"
very_solid_threshold = 0.05 # Max percentage of missing cells for a column to be considered very_solid
solid_threshold = 0.20 # Max percentage of missing cells for a column to be considered solid


def preprocess():
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)
    
    # Column Renaming - add PRE/INT/POS prefix to column names
    rename_columns.rename_columns(df, df_metadata)
    
    # TODO Feature Engineering - consolidate and engineer new features
    engineer_features.engineer_features(df, df_metadata)
    
    # Dataset Cleansing - remove noisy values such as "n/a"
    # cleansed_locs = cleanse_dataset.cleanse_dataset(df, df_metadata)
    
    # Expert Imputation - apply manual imputation rules
    # imputed_locs = expert_impute.expert_impute(df, df_metadata)
    
    # TODO Visualize Changed Cells
    
    # Convert Time columns into Numeric columns
    df = time_to_numeric.time_to_numeric(df, df_metadata)
    
    # Get Solid DataFrame - remove columns that are too sparse
    my_print_header("Getting solid-only PRE columns to help impute the sparse columns.")
    solid_df = get_solid_df.get_solid_df(df, df_metadata, sparsity_threshold=solid_threshold)
    very_solid_df = get_solid_df.get_solid_df(df, df_metadata, sparsity_threshold=very_solid_threshold)
    
    # TODO ML Imputation - Apply KNN and Random Forest Imputers
    df_KNN = impute_missing.impute_missing_KNN(df, df_metadata, solid_df, very_solid_df)
    df_RF = impute_missing.impute_missing_RF(df, df_metadata, solid_df, very_solid_df)
    df_KNN
    
    df.to_csv(out_path)