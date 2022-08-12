import time
from src.preprocess import expert_impute
from src.preprocess import impute_column
from src.preprocess import rename_columns
from src.preprocess import time_to_numeric
from src.preprocess.cleanse_dataset import cleanse_dataset
from src.preprocess.engineer_features import engineer_features
from src.preprocess.get_solid_df import get_solid_df
from utils.io import load_result_holders, print_and_log
from utils.io import print_and_log_w_header
from utils.io import save_experiment_df
from utils.io import save_experiment_pickle
from utils.io import bcolors
import utils.config as config
import pandas as pd

# col_type_path = "data/metadata/col_types.json"
# df_path = "data/input/AllTranTrainVal.csv"
# metadata_path = "data/input/metadata/Metadata.xlsx"
# out_KNN_path = "data/preprocessed/AllTranTrainVal-KNNImputed.csv"
# out_RF_path = "data/preprocessed/AllTranTrainVal-RFImputed.csv"

# Max percentage of missing cells for a column to be considered very_solid
# very_solid_threshold = 0.05
# Max percentage of missing cells for a column to be considered solid
# solid_threshold = 0.20


def preprocess(
    experiment_dir: str,
    very_solid_threshold: float = 0.05,
    solid_threshold: float = 0.20,
    metadata_path: str = "data/input/metadata/Metadata.xlsx",
    # df_path: str = "data/input/AllTranTrainVal.csv",
    df_path: str = "/Users/yifu/PycharmProjects/Radiotherapy-Prediction/data/input/RadiationAndANN_DATA_2021-11-15_0835.csv",
    use_solid_cols: bool = True,
    impute_only=False
) -> None:
    """
    Preprocess the dataset through:
        - renaming variables
        - cleaning up noisy values
        - engineering new features
        - applying expert pre-processing rules
        - searching for the optimal ML hyperparameters for each column
        - filling in missing values through ML imputation using opimal model
    Note: The imputation uses very_solid columns to predict other columns,
          and if use_solid_cols is set to True, it will use solid columns too.

    Args:
        experiment_dir (str): Path to experiment folder.
        very_solid_threshold (float): Max sparsity for very_solid columns.
        solid_threshold (float): Max sparsity for solid columns.
        metadata_path (str, optional): Path to metadata file.
        df_path (str, optional): Path to dataframe file.

    Raises:
        e:  Raise Error if there was a bug and config.debug_mode is True.
    """
    debug_mode = config.debug_mode
    # Read Dataset
    df_metadata = pd.read_excel(metadata_path, sheet_name="Sheet1")
    df = pd.read_csv(df_path)
    save_experiment_df(
        df, "Dataset-Original.csv", "input DataFrame before preprocessing"
    )
    if not impute_only:
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
        cleanse_dataset(df, df_metadata)
        save_experiment_df(df, "Dataset-cleansed.csv", "cleansed dataset")

    # Expert Imputation - apply manual imputation rules
    expert_impute.expert_impute(df, df_metadata)
    save_experiment_df(df, "Dataset-expert-imputed.csv", "expert-imputed")

    df = df.apply(pd.to_numeric, errors='coerce')
    save_experiment_df(
        df, "Dataset-to_numeric.csv", "numeric dataframe"
    )
    

    if "PRE_sln_met_nomogram_prob" not in df:
        print_and_log_w_header("Please apply nomogram on expert-imputed data for PRE_sln_met_nomogram_prob.")
        return
    
    # Convert Time columns into Numeric columns
    df = time_to_numeric.time_to_numeric(df, df_metadata)

    # Get Solid DataFrame - remove columns that are too sparse
    print_and_log_w_header("Using solid PRE columns to impute the other columns.")
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
    print_and_log(f"{len(removed_cols)} columns have > {solid_threshold} missing:")
    print_and_log(f"{removed_cols}", color=bcolors.NORMAL)
    # TODO ML Imputation - Apply KNN and Random Forest Imputers

    # Load result_holders from experiment_dir if there existes one
    result_holders = load_result_holders(experiment_dir)
    col_iter = df.columns

    # Sort columns by the increasing percentage of missing values
    col_iter = sorted(
        col_iter,
        key=lambda x: df[x].isnull().sum() / len(df),
        reverse=False
    )

    print_and_log_w_header("Imputing columns:", ", ".join(col_iter))

    # Use the very solid columns to impute all columns
    base_cols_df = very_solid_df

    for i, column in enumerate(col_iter):
        is_very_solid = column in very_solid_df.columns
        is_solid = column in solid_df.columns
        missingness = df[column].isna().sum() / len(df)
        print_and_log_w_header(
            f"Imputing column {column} with {missingness} missing."
            f" {i+1} of {len(col_iter)}."
        )
        print_str = f"{column} is "
        if is_very_solid:
            print_str += "very solid, therefore" \
                " any imputed values will be used for future imputations."
        elif is_solid:
            print_str += "solid."
        else:
            print_str += "sparse."
        print_and_log(print_str, color=bcolors.NORMAL)

        # Impute a column if and only if it has at least one missing value
        if missingness == 0:
            print_and_log(f"{column} has no missing values.", color=bcolors.NORMAL)
            continue
        elif missingness == 1:
            print_and_log(f"{column} is 100% missing.", color=bcolors.NORMAL)
            continue
        elif column == "POS_did_the_patient_receive_pm":
            print_and_log(f"Skipping {column} because it's the target column.")
            continue
        try:
            # TODO use imputed DF to impute the next column
            if column in result_holders:
                result_holder = result_holders[column]
            else:
                result_holder = None
            save_experiment_df(
                base_cols_df,
                f"Dataset_BaseCols-{column}-{i}_{len(col_iter)}.csv",
                f"the base_cols_df csv file for imputing column {column}"
            )
            imputed_df, result_holder = impute_column.impute_column(
                df,
                df_metadata,
                base_cols_df,
                column,
                "rmse",
                result_holder
                )
            result_holders[column] = result_holder
            save_experiment_df(
                imputed_df,
                f"Dataset_Imputed-{column}-{i}_{len(col_iter)}.csv",
                f"the imputed_df csv file for imputing column {column}"
            )
            df[column] = imputed_df[column]
            print_and_log(
                f"Overwrote missing {column} with imputed values.",
                color=bcolors.OKBLUE
            )
            # FIXME Add parameter for including is_solid columns
            if is_very_solid or (use_solid_cols and is_solid):
                base_cols_df = imputed_df
                print_and_log(
                    f"Overwrote base DataFrame ({column} is solid).",
                    color=bcolors.OKBLUE
                )
            if config.take_breaks:
                print_and_log("Taking a break...")
                time.sleep(30)
        except Exception as e:
            print_and_log(f"Skipped {column} due to error: {e}")
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
