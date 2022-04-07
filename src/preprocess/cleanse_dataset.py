import numpy as np

def cleanse_dataset(df, metadata_df):
    print(df)
    cleansed_locs = []
    for i, row in df.iterrows():
        for col in metadata_df["Field"]:
            script = metadata_df.loc[metadata_df["Field"] == col]["Cleansing_Script"].values[0]
            if str(script) == "nan":
                continue
            val = orig_val = row[col]
            exec(script)
            print(script)
            print(orig_val, val)
            print("___")
        