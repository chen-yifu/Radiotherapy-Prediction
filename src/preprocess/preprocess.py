# Preprocess the dataset through:
# - renaming variables
# - cleaning up noisy values
# - applying expert pre-processing rules
# - filling in missing values through imputation

from src.preprocess import expert_impute, rename_columns
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
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
out_path = "data/AllTranTrainVal-Preprocessed.csv"

def preprocess():
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)
    
    # Column Renaming - add PRE/INT/POS prefix to column names
    rename_columns.rename_columns(df, df_metadata)
    
    # TODO Feature Engineering - consolidate and engineer new features
    engineer_features.engineer_features(df, df_metadata)
    
    # TODO Dataset Cleansing - remove noisy values such as "n/a"
    cleanse_dataset.cleanse_dataset(df, df_metadata)
    
    # TODO Expert Imputation - apply manual imputation rules
    expert_impute.expert_impute(df, df_metadata)
    
    # TODO ML Imputation - Apply KNN and Random Forest Imputers