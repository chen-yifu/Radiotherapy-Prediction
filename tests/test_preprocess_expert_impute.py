import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.preprocess.impute_column import impute
from src.preprocess.expert_impute import expert_impute

data = {
    "PRE_men_status": [0, 2, np.nan, np.nan],
    "PRE_dob": ["1990-10-10", "1940-11-11", "1980-12-12", "1950-3-3"],
    "PRE_age_at_dx": [30, 80, 40, 70]
}

expected = {
    "PRE_men_status": [0, 2, 0, 2],
    "PRE_dob": ["1990-10-10", "1940-11-11", "1980-12-12", "1950-3-3"],
    "PRE_age_at_dx": [30, 80, 40, 70]
}

def test_impute():
    df = pd.DataFrame.from_dict(data)
    df_expected = pd.DataFrame.from_dict(expected)
    df_metadata = pd.read_excel("data/testing/input/metadata/Metadata.xlsx")
    # imputer = KNNImputer(n_neighbors=1)
    # df_imputed = impute(
    #     imputer,
    #     df,
    #     df_metadata,
    #     column_name="men_status"
    # )
    df_imputed = expert_impute(df, df_metadata)
    assert (df_imputed == df_expected).all().all()
