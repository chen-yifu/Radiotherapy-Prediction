def cleanse_dataset(df, metadata_df):
    print(metadata_df)
    for i, row in df.iterrows():
        for col in row.keys():
            script = metadata_df.loc[metadata_df["Field"] == col]["Cleansing_Script"]
            print(script)
            val = row[col]