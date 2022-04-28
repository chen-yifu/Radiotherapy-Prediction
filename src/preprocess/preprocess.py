import pickle
from src.preprocess import expert_impute, impute_column, rename_columns, time_to_numeric
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
from src.preprocess.get_solid_df import get_solid_df
from utils.get_timestamp import get_timestamp
from utils.io import my_print_header, save_experiment_df, save_experiment_pickle
from src.preprocess import *
# from sklearn.impute import *
import pandas as pd

col_type_path = "data/metadata/col_types.json"
df_path = "data/AllTranTrainVal.csv"
metadata_path = "data/metadata/Metadata.xlsx"
# out_KNN_path = "data/preprocessed/AllTranTrainVal-KNNImputed.csv"
# out_RF_path = "data/preprocessed/AllTranTrainVal-RFImputed.csv"

# Max percentage of missing cells for a column to be considered very_solid
very_solid_threshold = 0.05
# Max percentage of missing cells for a column to be considered solid
solid_threshold = 0.20


def preprocess(debug_mode: bool, experiment_dir: str) -> None:
    """
    Preprocess the dataset through:
        - renaming variables
        - cleaning up noisy values
        - applying expert pre-processing rules
        - filling in missing values through imputation
    Args:
        debug_mode (bool): Whether run in debug mode to save time.
        experiment_dir (str): Path to experiment folder.
    """

    # Read Dataset
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)

    # Column Renaming - add PRE/INT/POS prefix to column names
    rename_columns.rename_columns(df, df_metadata)

    # TODO Feature Engineering - consolidate and engineer new features
    engineer_features.engineer_features(df, df_metadata)

    # Shuffle the rows in DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    if debug_mode:
        df = df.iloc[:20, :]
    else:
        # Dataset Cleansing - remove noisy values such as "n/a"
        cleansed_locs = cleanse_dataset.cleanse_dataset(df, df_metadata)
        # Expert Imputation - apply manual imputation rules
        imputed_locs = expert_impute.expert_impute(df, df_metadata)

    # TODO Visualize Changed Cells

    # Convert Time columns into Numeric columns
    df = time_to_numeric.time_to_numeric(df, df_metadata)


    # Get Solid DataFrame - remove columns that are too sparse
    my_print_header("Using solid PRE columns to impute the other columns.")
    solid_df = get_solid_df.get_solid_df(
        df,
        df_metadata,
        sparsity_threshold=solid_threshold
        )
    very_solid_df = get_solid_df.get_solid_df(
        df,
        df_metadata,
        sparsity_threshold=very_solid_threshold
        )
    # TODO ML Imputation - Apply KNN and Random Forest Imputers
    result_holders = {}
    # col_iter = [c for c in df.columns[1:] if c not in very_solid_df.columns]
    col_iter = df.columns[1:]
    if debug_mode:
        col_iter = col_iter[:20]

    my_print_header("Imputing columns:", ", ".join(col_iter))
    for column in col_iter:
        missingness = df[column].isna().sum() / len(df)
        my_print_header(f"Imputing column {column} with {missingness} missing")
        # Impute a column if and only if it has at least one missing value
        if missingness == 0:
            print(f"{column} has no missing values.")
            continue
        elif missingness == 1:
            print(f"{column} has no non-missing values.")
            continue
        try:
            df_preprocessed, result_holder = impute_column.impute_column(
                df,
                df_metadata,
                solid_df,
                very_solid_df,
                column,
                "rmse",
                debug_mode=debug_mode
                )
            result_holders[column] = result_holder
        except Exception as e:
            print(f"Skipped {column} due to error: {e}")
            if debug_mode:
                raise e
            else:
    
                continue

    # TODO implement df_preprocessed in impute_column
    # save_experiment_df(
    #     df_preprocessed,
    #     "AllTranTrainVal-preprocessed.csv",
    #     "preprocessed csv file"
    #     )

    save_experiment_pickle(
        result_holders,
        "AllColumnsGridSearchResultHolders.pkl",
        "KNN and RF imputation hyperparamter search"
        )

    return
