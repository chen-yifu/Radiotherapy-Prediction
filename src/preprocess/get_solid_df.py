from utils.printers import *
import pandas as pd
from tqdm import tqdm


def get_solid_df(df, df_metadata, sparsity_threshold):
    # If more than sparsity_threshold of cells are missing, then remove the column
    columns = sorted(df.columns, key=lambda x: df[x].isna().sum())
    # Remove sparse columns
    for col in df.columns:
        sparsity = round(df[col].isna().sum() / len(df), 4)
        if sparsity > sparsity_threshold:
            columns.remove(col)
        else:
            col_group = df_metadata.loc[df_metadata["Field"] == col]["Group"]
            if len(col_group) and col_group.item() != "PRE":
                columns.remove(col)
            
    print(f"{len(columns)} out of {len(df.columns)} all columns are PRE and have ≤ {sparsity_threshold} missing cells.")
    
    result_df = pd.DataFrame(df[columns])
    return result_df


