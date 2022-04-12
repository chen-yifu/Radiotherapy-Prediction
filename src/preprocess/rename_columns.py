from utils.IO import *
from tqdm import tqdm


def rename_columns(df, df_metadata):
    my_print_header("Column Renaming...")
    
    rename_dict = {}
    for col in tqdm(df.columns):
        row = df_metadata.loc[df_metadata["Original Field Name"] == col]
        prefix = row["Group"].item()
        rename_dict[col] = prefix+"_"+col
    df.rename(columns=rename_dict, inplace=True)
    
    print("✅ Column Renamming - Added PRE/INT/POS column name prefixes.")