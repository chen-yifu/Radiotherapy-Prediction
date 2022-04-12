import numpy as np
from utils.io import *
from tqdm import tqdm

def cleanse_dataset(df, metadata_df):
    my_print_header("Dataset Cleansing...")
    cleansed_locs = {}
    global row, val, orig_val
    for col in tqdm(metadata_df["Field"]):
        script = metadata_df.loc[metadata_df["Field"] == col]["Cleansing_Script"].values[0]
        for i, row in df.iterrows():
            if str(script) == "nan":
                continue
            if str(row[col]) == "nan":
                continue
            val = orig_val = row[col]
            exec(script, globals())
            if str(orig_val) != "nan" and orig_val != val:
                df.loc[i, col] = val
                cleansed_locs[(i, col)] = (orig_val, val)
    my_print(f"✅ Dataset Cleansing - Used expert manual rules to replace noisy values. {len(cleansed_locs)} cells were changed.", plain=True)
    return cleansed_locs