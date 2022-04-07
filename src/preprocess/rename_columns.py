def rename_columns(df, df_metadata):
    rename_dict = {}
    for col in df.columns:
        row = df_metadata.loc[df_metadata["Field"] == col]
        prefix = row["Group"].item()
        rename_dict[col] = prefix+"_"+col
    df.rename(columns=rename_dict, inplace=True)