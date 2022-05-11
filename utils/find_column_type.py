import pandas as pd


class ColumnType:
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    REAL = "real"
    INTEGER = "integer"
    TIME = "time"
    STRING = "string"


def find_column_type(df_metadata: pd.DataFrame, col: str):
    """
    Find the type of a column.

    Args:
        df_metadata (pd.DataFrame): Metadata DataFrame.
        col (str): Column name.
    """
    type = df_metadata.loc[
        df_metadata["Field"] == col
        ]["Type"].values[0]
    return type


def is_integral_type(col_type: str):
    return col_type in [
        ColumnType.CATEGORICAL,
        ColumnType.ORDINAL,
        ColumnType.INTEGER
    ]

