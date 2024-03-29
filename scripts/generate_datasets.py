import pandas as pd
import os
from datetime import datetime
from argparse import ArgumentParser
global out_dir, input_dir

# Parse Arguments
parser = ArgumentParser()
parser.add_argument(
    "--df_dir_path",
    type=str,
    required=False,
    help="path to the folder of pre-processed DataFrames"
)
args = parser.parse_args()

current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H%M")
# input_dir = "/Users/yifu/PycharmProjects/Radiotherapy-Prediction/data/output/2022-07-11-151005/DataFrames"
input_dir = args.df_dir_path
output_dir = os.path.join(input_dir, "subset_dataframes")
# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_idx = 0

class bcolors:
    # Helper class to print in terminal with colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NORMAL = '\033[0m'

def custom_to_csv(df, out_dir, out_name, index=False):
    global df_idx
    # Reformat the out_name with number of columns
    out_name = out_name.replace(".csv", str(len(df.columns)) + "cols.csv")
    # Add Hex prefix
    out_name = format(df_idx, "x").upper() + "_" + out_name
    df_idx += 1
    df.to_csv(os.path.join(out_dir, out_name), index=index)
    column_sparsities = {}
    for column in df.columns:
        column_sparsities[column] = df[column].isnull().sum() / len(df)
    # Round the sparsities to 3 decimal places
    column_sparsities = {k: round(v, 3) for k, v in column_sparsities.items()}
    print(
        f"{bcolors.OKGREEN}{out_name} has {len(df.columns)} columns."
        f"{bcolors.ENDC}"
        f"The columns and sparsities are:"
    )
    # Print 3 columns on each row
    for i, (column, sparsity) in enumerate(column_sparsities.items()):
        if i % 3 == 2:
            end = "\n"
        else:
            end = " | "
        print(f"{bcolors.BOLD}{column}:{bcolors.ENDC} {sparsity}", end=end)
    print("\n"+"-"*50)

# Generate different subsets of the dataset
if __name__ == "__main__":
    # Read the datasets needed
    # df_cleansed = pd.read_csv(os.path.join(input_dir, "Dataset-cleansed.csv"))
    df_solid = pd.read_csv(os.path.join(input_dir, "Dataset-solid.csv"))
    df_very_solid = pd.read_csv(
        os.path.join(input_dir, "Dataset-very_solid.csv")
    )
    df_expert_imputed = pd.read_csv(
        os.path.join(input_dir, "Dataset-expert-imputed.csv")
    )
    df_preprocessed = pd.read_csv(
        os.path.join(input_dir, "Dataset-Preprocess-Result.csv")
    )
    # Calculate the column sparsity of df_expert_imputed as a dictionary
    col_sparsities = {}
    for col in df_expert_imputed.columns:
        col_sparsities[col] = df_expert_imputed[col].isnull().sum() \
            / len(df_expert_imputed)
    print(f"Column sparsities: {col_sparsities}")
    # Generate Alan's Heuristic DF
    alan_picked_cols = [
        "PRE_susp_LN_prsnt_composite",
        "PRE_prominent_axillary_lymph",
        "PRE_axillary_lymphadenopathy",
        "PRE_internal_mammary_lymphaden",
        "PRE_axillary_lymphadenopathy_p",
        "PRE_int_mammary_lymphade_pet"
    ]
    alan_picked_df = df_preprocessed[alan_picked_cols]
    custom_to_csv(
        alan_picked_df,
        output_dir,
        "PRE-alan-heuristic.csv",
    )
    # Generate expert-picked unimputed columns DF
    expert_picked_cols_enhanced = [
        'PRE_record_id',
        'PRE_age_at_dx',
        'PRE_men_status',
        'PRE_tumor_size_mm',
        'PRE_tumor_max_size_composite',
        'PRE_tumor_grade',
        'PRE_tumor_stge',
        'PRE_tumor_location',
        'PRE_tumor_location_trans',
        'PRE_metastatic_carcinoma_on_ax',
        'PRE_lymphovascular_invasion0',
        'PRE_pr_status',
        'PRE_er_status',
        'PRE_her_status',
        'PRE_systhe___chemo',
        'PRE_systhe___hormonal',
        'PRE_systhe___chemo_and_hormonal',
        'PRE_systhe___no_systhe',
        'PRE_systhe___radiation',
        'PRE_susp_LN_prsnt_composite',
        'PRE_susp_LN_size_composite',
        "PRE_susp_LN_prsnt_composite",
        'PRE_margin_status',
        'PRE_closest_margin',
        'PRE_num_closest_margins_trans',
        'PRE_axillary_lymph_node_palpab',
        'PRE_prominent_axillary_lymph',
        "PRE_axillary_lymphadenopathy",
        "PRE_internal_mammary_lymphaden",
        "PRE_axillary_lymphadenopathy_p",
        'PRE_internal_mammary_lymph_nod',
        "PRE_int_mammary_lymphade_pet",
        'PRE_axillary_lymph_node_max_si',
        'PRE_lymph_node_max_size_mm0',
        'PRE_img_size'
    ]
    expert_picked_cols = [
        'PRE_record_id',
        'PRE_men_status',
        'PRE_tumor_size_mm',
        'PRE_tumor_max_size_composite',
        'PRE_tumor_grade',
        'PRE_tumor_location',
        'PRE_metastatic_carcinoma_on_ax',
        'PRE_lymphovascular_invasion0',
        'PRE_pr_status',
        'PRE_er_status',
        'PRE_her_status',
        'PRE_susp_LN_prsnt_composite',
        'PRE_susp_LN_size_composite',
        "PRE_susp_LN_prsnt_composite",
        'PRE_axillary_lymph_node_palpab',
        'PRE_prominent_axillary_lymph',
        "PRE_axillary_lymphadenopathy",
        "PRE_internal_mammary_lymphaden",
        "PRE_axillary_lymphadenopathy_p",
        'PRE_internal_mammary_lymph_nod',
        "PRE_int_mammary_lymphade_pet",
        'PRE_axillary_lymph_node_max_si',
        'PRE_lymph_node_max_size_mm0',
        'PRE_img_size'
    ]
    # expert_picked_df = df_cleansed[expert_picked_cols]
    # custom_to_csv(
    #     expert_picked_df,
    #     output_dir,
    #     "PRE-expert-unimputed.csv"
    # )
    # Generate expert-picked-imputed columns DF
    expert_picked_imputed_df = df_expert_imputed[expert_picked_cols]
    custom_to_csv(
        expert_picked_imputed_df,
        output_dir,
        "PRE-expert-imputed.csv"
    )
    
    expert_picked_enhanced_imputed_df = df_expert_imputed[expert_picked_cols_enhanced]
    custom_to_csv(
        expert_picked_enhanced_imputed_df,
        output_dir,
        "PRE-expert-picked.csv"
    )
    expert_picked_imputed_df_preprocessed = df_preprocessed[expert_picked_cols]
    custom_to_csv(
        expert_picked_imputed_df,
        output_dir,
        "PRE-expert-imputed-preprocessed.csv"
    )
    
    expert_picked_enhanced_imputed_df_preprocessed = df_preprocessed[expert_picked_cols_enhanced]
    custom_to_csv(
        expert_picked_enhanced_imputed_df,
        output_dir,
        "PRE-expert-imputed-enhanced-preprocessed.csv"
    )

    # Generate the DF containing all the columns picked by experts,
    # And also the columns with sparsities at most x%
    def save_sparsity_df(sparsity_threshold, use_PRE_only=True):
        global output_dir
        # Merge the df_expert_imputed with subset of df_preprocessed
        # Keep columns below sparsity threshold

        # TODO, whether implement the following:
        # if use_PRE_only is False, then use all PRE columns, but not the
        # POS columns that are below the threshold
        cols_with_sparsity = [
            col for col, spar in col_sparsities.items()
            if spar <= sparsity_threshold
            # or (not use_PRE_only and col.startswith("PRE"))
        ]
        print(f"{len(cols_with_sparsity)} Columns: {cols_with_sparsity}")
        if use_PRE_only:
            # Remove all columns that doesn't begin with "PRE_"
            cols_with_sparsity = [
                col for col in cols_with_sparsity
                if col.startswith("PRE_")
            ]
            file_name = f"PRE-{sparsity_threshold}spars-expert-ML-imputed.csv"
        else:
            file_name = f"POS-{sparsity_threshold}spars-expert-ML-imputed.csv"
        df_sparsity = df_preprocessed[cols_with_sparsity]
        # Merge DFs, remove duplicate columns by keeping the first one
        df_sparsity = pd.merge(
            df_sparsity,
            expert_picked_enhanced_imputed_df,
            on="PRE_record_id",
            how="outer",
            suffixes=('', '_duplicate')
        )
        # Drop duplicate columns
        cols_to_drop = [
            col for col in df_sparsity.columns
            if col.endswith("_duplicate")
        ]
        df_sparsity = df_sparsity.drop(cols_to_drop, axis=1)
        custom_to_csv(
            df_sparsity,
            output_dir,
            file_name
        )

    for use_PRE_only in [True, False]:
        for sparsity_threshold in [0.05, 0.20, 0.50, 0.80, 1.00]:
            save_sparsity_df(sparsity_threshold, use_PRE_only)


