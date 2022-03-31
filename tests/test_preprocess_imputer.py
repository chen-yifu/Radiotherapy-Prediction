import pandas as pd
import numpy as np
from src.preprocess.impute import *

data = {
    "men_status": [1, 0, np.nan, np.nan],
    "dob": ["1970-10-10", "1940-11-11", "1980-12-12", "1950-3-3"],
}

expected = {
    "men_status": [1, 0, 0, 1],
    "dob": ["1970-10-10", "1940-11-11", "1980-12-12", "1950-3-3"],
}

def test_impute():
    df = pd.DataFrame.from_dict(data)
    df_expected = pd.DataFrame.from_dict(expected)
    df_imputed = impute_df(df)
    assert (df_imputed == df_expected).all().all()