from src.preprocess import expert_impute
from src.preprocess import impute_column
from src.preprocess import rename_columns
from src.preprocess import time_to_numeric
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
from src.preprocess.get_solid_df import get_solid_df
from utils.io import my_print
from utils.io import my_print_header
from utils.io import save_experiment_df
from utils.io import save_experiment_pickle
from utils.io import bcolors
import utils.config as config
import pandas as pd

col_type_path = "data/metadata/col_types.json"
df_path = "data/input/AllTranTrainVal.csv"
metadata_path = "data/input/metadata/Metadata.xlsx"
# out_KNN_path = "data/preprocessed/AllTranTrainVal-KNNImputed.csv"
# out_RF_path = "data/preprocessed/AllTranTrainVal-RFImputed.csv"

# Max percentage of missing cells for a column to be considered very_solid
very_solid_threshold = 0.05
# Max percentage of missing cells for a column to be considered solid
solid_threshold = 0.20


def preprocess(experiment_dir: str) -> None:
    """
    Preprocess the dataset through:
        - renaming variables
        - cleaning up noisy values
        - applying expert pre-processing rules
        - filling in missing values through imputation
    Args:
        experiment_dir (str): Path to experiment folder.
    """
    debug_mode = config.debug_mode
    # Read Dataset
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)

    # Column Renaming - add PRE/INT/POS prefix to column names
    rename_columns.rename_columns(df, df_metadata)


    # Shuffle the rows in DataFrame
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    if debug_mode:
        df = df.iloc[:50, :]

    # TODO Feature Engineering - consolidate and engineer new features
    df = engineer_features(df, df_metadata)
    save_experiment_df(df, "Dataset-feature_eng.csv", "engineered features")

    # Dataset Cleansing - remove noisy values such as "n/a"
    cleansed_locs = cleanse_dataset(df, df_metadata)
    save_experiment_df(
        df, "Dataset-cleansed.csv", "cleansed dataset"
    )
    # Expert Imputation - apply manual imputation rules
    imputed_locs = expert_impute.expert_impute(df, df_metadata)
    save_experiment_df(
        df, "Dataset-expert-imputed.csv", "expert-imputed"
    )

    df = df.apply(pd.to_numeric, errors='coerce')
    save_experiment_df(
        df, "Dataset-numeric2.csv", "numeric dataframe"
    )


    # TODO Visualize Changed Cells

    # Convert Time columns into Numeric columns
    df = time_to_numeric.time_to_numeric(df, df_metadata)


    # Get Solid DataFrame - remove columns that are too sparse
    my_print_header("Using solid PRE columns to impute the other columns.")
    solid_df = get_solid_df(
        df,
        df_metadata,
        sparsity_threshold=solid_threshold
        )
    save_experiment_df(
        solid_df, "Dataset-solid.csv", "solid columns"
    )
    very_solid_df = get_solid_df(
        df,
        df_metadata,
        sparsity_threshold=very_solid_threshold
        )
    save_experiment_df(
        very_solid_df, "Dataset-very_solid.csv", "very solid columns"
    )
    # Show the columns that have been filtered out by sparsity theshold
    removed_cols = df.columns[~df.columns.isin(solid_df.columns)]
    my_print(f"{len(removed_cols)} columns have > {solid_threshold} missing:")
    print(f"{removed_cols}")
    # TODO ML Imputation - Apply KNN and Random Forest Imputers
    result_holders = {}
    col_iter = df.columns
    # Sort columns by the increasing percentage of missing values
    col_iter = sorted(
        col_iter,
        key=lambda x: df[x].isnull().sum() / len(df),
        reverse=False
    )

    my_print_header("Imputing columns:", ", ".join(col_iter))

    # Use the very solid columns to impute all columns
    base_cols_df = very_solid_df

    for i, column in enumerate(col_iter):
        is_very_solid = column in very_solid_df.columns
        is_solid = column in solid_df.columns
        missingness = df[column].isna().sum() / len(df)
        my_print_header(
            f"Imputing column {column} with {missingness} missing."
            f" {i+1} of {len(col_iter)}."
        )
        print(f"{column} is ", end="")
        if is_very_solid:
            print("very solid, therefore"
                  " any imputed values will be used for future imputations.")
        elif is_solid:
            print("solid.")
        else:
            print("sparse.")
        # Impute a column if and only if it has at least one missing value
        if missingness == 0:
            print(f"{column} has no missing values.")
            continue
        elif missingness == 1:
            print(f"{column} has no non-missing values.")
            continue
        elif column == "POS_did_the_patient_receive_pm":
            print(f"Skipping {column} because it's the target column.")
            continue
        try:
            # TODO use imputed DF to impute the next column
            imputed_df, result_holder = impute_column.impute_column(
                df,
                df_metadata,
                base_cols_df,
                column,
                "rmse"
                )
            result_holders[column] = result_holder
            save_experiment_df(
                df,
                f"Dataset_Imputed-{column}-{i}_{len(col_iter)}.csv",
                f"the imputed csv file for column {column}"
            )
            df[column] = imputed_df[column]
            my_print(
                f"Overwrote missing {column} with imputed values.",
                color=bcolors.OKBLUE
            )
            if is_very_solid:
                base_cols_df = imputed_df
                my_print(
                    f"Overwrote base DataFrame ({column} is very solid).",
                    color=bcolors.OKBLUE
                )
        except Exception as e:
            my_print(f"Skipped {column} due to error: {e}")
            if debug_mode:
                raise e
            else:
                continue

    # TODO implement df_preprocessed in impute_column
    save_experiment_df(
        df,
        "Dataset-Preprocess-Result.csv",
        "preprocessed csv file"
        )

    save_experiment_pickle(
        result_holders,
        "AllColumnsGridSearchResultHolders.pkl",
        "KNN and RF imputation hyperparamter search"
        )

    return
