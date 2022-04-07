import pandas as pd


def time_to_numeric(df, df_metadata):
    # Convert Time columns into Numeric columns
    for col in df.columns:
        if len(df_metadata.loc[df_metadata["Field"] == col]):
            type = df_metadata.loc[df_metadata["Field"] == col]["Type"].values[0]
            if type == "time":
                df[col] = df[col].apply(lambda x: pd.to_datetime(x).value)
    return df