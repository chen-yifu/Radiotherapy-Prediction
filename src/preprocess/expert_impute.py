from utils.io import my_print_header, my_print
from tqdm import tqdm
import numpy as np


def expert_impute(df, metadata_df):
    my_print_header("Expert Imputation...")
    imputed_locs = {}
    global row, val, orig_val
    for col in tqdm(metadata_df["Field"]):
        script = metadata_df.loc[
            metadata_df["Field"] == col
            ]["Imputation_Script_if_Missing"].values[0]
        for i, row in df.iterrows():
            if str(script) == "nan" or not len(str(script)):
                continue
            elif str(row[col]) != "nan" and len(str(row[col])):
                continue
            val = orig_val = row[col]
            exec(script, globals())
            if str(orig_val) == "nan" and orig_val != val:
                if not val or val == "":
                    df.loc[i, col] = np.nan
                else:
                    df.loc[i, col] = val
                imputed_locs[(i, col)] = (orig_val, val)
    my_print("✅ Expert Imputation - Used expert manual rules to"
             "impute missing values in some columns."
             f"{len(imputed_locs)} cells were filled.",
             plain=True)

    return imputed_locs
