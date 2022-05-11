import pandas as pd
from utils.find_column_type import find_column_type, ColumnType


def time_to_numeric(df, df_metadata):
    # Convert Time columns into Numeric columns
    for col in df.columns:
        if len(df_metadata.loc[df_metadata["Field"] == col]):
            type = find_column_type(df_metadata, col)
            if type == ColumnType.TIME:
                df[col] = df[col].apply(lambda x: pd.to_datetime(x).value)
    return df

