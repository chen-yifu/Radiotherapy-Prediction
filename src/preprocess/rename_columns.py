from utils.printers import *
from tqdm import tqdm

def rename_columns(df, df_metadata):
    rename_dict = {}
    for col in tqdm(df.columns):
        row = df_metadata.loc[df_metadata["Original Field Name"] == col]
        prefix = row["Group"].item()
        rename_dict[col] = prefix+"_"+col
    df.rename(columns=rename_dict, inplace=True)
    
    my_print("✅ Column Renamming - Added PRE/INT/POS column name prefixes.")