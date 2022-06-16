import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from utils.find_column_type import is_integral_type
from utils.io import my_print, my_print_header, save_experiment_df
import re


def engineer_features(
    df: pd.DataFrame,
    df_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Perform feature engineering on the dataframe
    Note: When a new feature is added, remember to add to Metadata CSV file
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
    susp_LN_size_composite_cols = [
        'PRE_lymph_node_max_size_mm',
        'PRE_lymph_node_max_size_mm0',
        'PRE_axillary_lymph_node_max_si',
        'PRE_internal_mammary_lymph_nod'
        ]
 
    pre_tumor_max_size_composite_cols = [
        "PRE_img_size",  # Ultrasound, in MM
        "PRE_tumor_size_mm",  # Mammography, in MM
        "PRE_size_of_the_largest_foci_c",  # PET, in CM
        ]
    age_at_dxs = []
    susp_LN_size_composites = []
    susp_LN_prsnt_composites = []
    tumor_max_size_composites = []

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
            print(f"{row['PRE_record_id']} age_at_dx: {years_elapsed}, dob: {dob}, dx_date: {dx_date}")
        # Construct "susp_LN_size_composite" as the maximum LN abnormality size
        max_size = 0
        for col in susp_LN_size_composite_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            max_size = max(max_size, value)
        susp_LN_size_composites.append(max_size)
        
        # Construct "susp_LN_prsnt_composite" to be the presence of abnormal LN
        susp_LN_prsnt_composite = 3
        for col in abnormal_ln_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            else:
                if str(value).strip().replace(".0", "") == "1":
                    susp_LN_prsnt_composite = 1
        susp_LN_prsnt_composites.append(susp_LN_prsnt_composite)
        
        tumor_max_site_composite = 0
        for col in pre_tumor_max_size_composite_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            else:
                if col == "PRE_size_of_the_largest_foci_c":
                    value = value * 10
                tumor_max_site_composite = max(tumor_max_site_composite, value)
        tumor_max_size_composites.append(tumor_max_site_composite)

    df.insert(
        list(df.columns).index("PRE_dob")+1,
        "PRE_age_at_dx",
        age_at_dxs
        )
    df.insert(
        list(df.columns).index("PRE_lymph_node_max_size_mm")+1,
        "PRE_susp_LN_size_composite",
        susp_LN_size_composites
        )
    df.insert(
        list(df.columns).index("PRE_abnormal_lymph")+1,
        "PRE_susp_LN_prsnt_composite",
        susp_LN_prsnt_composites
        )
    
    df.insert(
        list(df.columns).index("PRE_img_size")+1,
        "PRE_tumor_max_size_composite",
        tumor_max_size_composites
        )

    # Converet all string cells to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # If a column is integral type, make it column int
    for col in df.columns:
        if is_integral_type(col):
            df[col] = df[col].astype("int")

    my_print("Converted all strings to numeric, and used NaN if impossible.")
    my_print(
        "✅ Feature Engineering - Added new feature 'PRE_age_at_dx',"
        "'PRE_susp_LN_size_composite',  'PRE_susp_LN_prsnt_composite', and 'PRE_tumor_max_size_composite'"
    )

    return df
