import numpy as np
from utils.IO import *
from tqdm import tqdm


def expert_impute(df, metadata_df):
    my_print_header("Expert Imputation...")
    imputed_locs = {}
    global row, val, orig_val
    for col in tqdm(metadata_df["Field"]):
        script = metadata_df.loc[metadata_df["Field"] == col]["Imputation_Script_if_Missing"].values[0]
        for i, row in df.iterrows():
            if str(script) == "nan":
                continue
            if str(row[col]) == "nan":
                continue
            val = orig_val = row[col]
            exec(script, globals())
            if str(orig_val) != "nan" and orig_val != val:
                df.loc[i, col] = val
                imputed_locs[(i, col)] = (orig_val, val)
    my_print(
        f"✅ Expert Imputation - Used expert manual rules to impute missing values in some columns. {len(imputed_locs)} cells were filled.", plain=True)
    return imputed_locs