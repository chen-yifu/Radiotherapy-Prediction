import numpy as np

def cleanse_dataset(df, metadata_df):
    print(df)
    for i, row in df.iterrows():
        for col in metadata_df["Field"]:
            script = metadata_df.loc[metadata_df["Field"] == col]["Cleansing_Script"].value()
            if script == np.nan:
                continue
            val = row[col]
            print(script)
        print(row.keys())
        break