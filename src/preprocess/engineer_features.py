import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from utils.io import my_print_header, save_experiment_df
import re


def engineer_features(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Perform feature engineering on the dataframe

    Args:
        df (pd.DataFrame): original DataFrame
        df_metadata (pd.DataFrame): DataFrame with metadata

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """

    # Construct new columns from existing data
    my_print_header("Feature Engineering...")
    abnormal_ln_cols = [
        'PRE_abnormal_lymph',
        'PRE_prominent_axillary_lymph',
        'PRE_axillary_lymphadenopathy',
        'PRE_internal_mammary_lymphaden',
        'PRE_axillary_lymphadenopathy_p',
        'PRE_int_mammary_lymphade_pet'
        ]
    abnormal_ln_size_cols = [
        'PRE_lymph_node_max_size_mm',
        'PRE_lymph_node_max_size_mm0',
        'PRE_axillary_lymph_node_max_si',
        'PRE_internal_mammary_lymph_nod'
        ]

    age_at_dxs = []
    abnormal_ln_sizes = []
    abnormal_ln_presents = []

    # Remove "ANN" prefix from record_id
    df['PRE_record_id'] = df['PRE_record_id'].apply(
        lambda x: re.sub(r'ANN|L|R', '', x))

    for i, row in tqdm(df.iterrows()):
        # TODO pre_op_biopsy_date_year or surgery_date_year - dob
        # Construct "age_at_dx" as the age at the time of diagnosis
        dob = pd.to_datetime(row["PRE_dob"])
        dx_date = pd.to_datetime(row["PRE_dximg_date"])
        if str(dx_date) == "nan":
            age_at_dxs.append(np.nan)
        else:
            years_elapsed = abs(round((dx_date - dob).days / 365.25, 2))
            age_at_dxs.append(years_elapsed)
        # Construct "abnormal_ln_size" as the maximum LN abnormality size
        max_size = 0
        for col in abnormal_ln_size_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            max_size = max(max_size, value)
        abnormal_ln_sizes.append(max_size)
        # Construct "abnormal_ln_present" to be the presence of abnormal LN
        abnormal_ln_present = 3
        for col in abnormal_ln_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            else:
                if str(value).strip().replace(".0", "") == "1":
                    abnormal_ln_present = 1
        abnormal_ln_presents.append(abnormal_ln_present)

    df.insert(
        list(df.columns).index("PRE_dob")+1,
        "PRE_age_at_dx",
        age_at_dxs
        )
    df.insert(
        list(df.columns).index("PRE_lymph_node_max_size_mm")+1,
        "PRE_abnormal_ln_size",
        abnormal_ln_sizes
        )
    df.insert(
        list(df.columns).index("PRE_abnormal_lymph")+1,
        "PRE_abnormal_ln_present",
        abnormal_ln_presents
        )
    # Converet all string cells to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    print("Converted all string cells to numeric, and used NaN if impossible.")
    print("✅ Feature Engineering - Added new feature 'PRE_age_at_dx',"
          "'PRE_abnormal_ln_size', and 'PRE_abnormal_ln_present'.")

    return df
