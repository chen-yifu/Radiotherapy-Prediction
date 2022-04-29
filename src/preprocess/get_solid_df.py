from utils.io import my_print
import pandas as pd


def get_solid_df(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame,
    sparsity_threshold: float
) -> pd.DataFrame:
    """keep only columns with lower than sparsity_threshold missingness

    Args:
        df (pd.DataFrame): original DataFrame
        df_metadata (pd.DataFrame): DataFrame with metadata
        sparsity_threshold (float): threshold for missingness

    Returns:
        pd.DataFrame: DataFrame with only solid columns
    """
    columns = sorted(df.columns, key=lambda x: df[x].isna().sum())
    for col in df.columns:
        sparsity = round(df[col].isna().sum() / len(df), 4)
        if sparsity > sparsity_threshold:
            # Remove sparse columns
            columns.remove(col)
        else:
            col_group = df_metadata.loc[df_metadata["Field"] == col]["Group"]
            if len(col_group) and col_group.item() != "PRE":
                columns.remove(col)

    my_print(f"{len(columns)} out of {len(df.columns)} all columns are PRE"
             f" and have ≤ {sparsity_threshold} missing cells.")
    print(f"They are {columns}")

    result_df = pd.DataFrame(df[columns])
    return result_df

