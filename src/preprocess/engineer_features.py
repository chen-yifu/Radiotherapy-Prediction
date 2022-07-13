import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from utils.find_column_type import is_integral_type
from utils.io import print_and_log, print_and_log_w_header, save_experiment_df
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

    def transform_location(la: str, lo: str) -> int:
        """Transform tumor location to a single number

        Args:
            la (str): laterality, 1 = left, 2 = right
            lo (str): location
        """
        if str(la) == "nan" or str(lo) == "nan":
            location_trans = np.nan
        else:
            clock_orientation = re.search(r"^\d+", lo)
            location_trans = ""
            if clock_orientation:
                loc = clock_orientation.group()
                if len(loc) >= 3:
                    loc_num = int(loc[:2])
                    if loc_num > 12:
                        loc = loc[:1]
                    else:
                        loc = loc[:2]
                if len(loc):
                    if la in ["1", "1.0"]: # RIGHT
                        location_trans = "-" + loc
                    elif la in ["2", "2.0"]:
                        location_trans = "+" + loc
            try:
                location_trans = int(location_trans)
            except:
                location_trans = np.nan
        return location_trans
    
    # Construct new columns from existing data
    print_and_log_w_header("Feature Engineering...")
    
    # First fix a few issues where the date is parsed incorrectly by pandas
    # Convert dob column to datetime
    df["PRE_dob"] = pd.to_datetime(df["PRE_dob"])
    # Fix issues where two-digit years e.g., 58 is parsed as 2058
    df["PRE_dob"] = df["PRE_dob"].apply(lambda x: x.year - 100 if x.year > 2010 else x.year)
    df["PRE_dximg_date"] = pd.to_datetime(df["PRE_dximg_date"])
    df["PRE_dximg_date"] = df["PRE_dximg_date"].apply(lambda x: x.year + 100 if x.year < 1990 else x.year)
    df["PRE_surgery_date"] = pd.to_datetime(df["PRE_surgery_date"])
    df["PRE_surgery_date"] = df["PRE_surgery_date"].apply(lambda x: x.year + 100 if x.year < 1990 else x.year)
    df["PRE_pre_op_biop_date"] = pd.to_datetime(df["PRE_pre_op_biop_date"])
    df["PRE_pre_op_biop_date"] = df["PRE_pre_op_biop_date"].apply(lambda x: x.year + 100 if x.year < 1990 else x.year)
    # For bi_rads_score, keep only the numeric prefix part if it exists
    # Convert 4A, 4B, 4C in bi_rads_score to 4, 4.3, 4.6 respectively 
    df["PRE_bi_rads_score"] = df["PRE_bi_rads_score"].apply(lambda x: re.sub(r"^4A", "4", str(x)))
    df["PRE_bi_rads_score"] = df["PRE_bi_rads_score"].apply(lambda x: re.sub(r"^4B", "4.3", str(x)))
    df["PRE_bi_rads_score"] = df["PRE_bi_rads_score"].apply(lambda x: re.sub(r"^4C", "4.7", str(x)))
    # df["PRE_bi_rads_score"] = df["PRE_bi_rads_score"].apply(lambda x: re.search(r"\d+", str(x)).group() if re.search(r"\d+", str(x)) else np.nan)
    
    abnormal_ln_cols = [
        'PRE_abnormal_lymph',
        'PRE_prominent_axillary_lymph',
        'PRE_axillary_lymphadenopathy',
        'PRE_internal_mammary_lymphaden',
        'PRE_axillary_lymphadenopathy_p',
        'PRE_int_mammary_lymphade_pet',
        'PRE_axillary_lymph_node_palpab'
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
    age_at_surg = []
    bmis = []
    pre_tumor_location_trans = []
    pos_tumor_location_trans = []
    margins_trans = []
    susp_LN_size_composites = []
    susp_LN_prsnt_composites = []
    tumor_max_size_composites = []

    
    # Remove "ANN" prefix from record_id
    df['PRE_record_id'] = df['PRE_record_id'].apply(
        lambda x: re.sub(r'ANN', '', str(x)))
    # If record_id has left laterality, replace it with .2
    df['PRE_record_id'] = df['PRE_record_id'].apply(
        lambda x: re.sub(r'L', '.2', str(x)))
    # If record_id has right laterality, replace it with .1
    df['PRE_record_id'] = df['PRE_record_id'].apply(
        lambda x: re.sub(r'R', '.1', str(x)))

    
    for _, row in tqdm(df.iterrows()):
        # transformation for tumor location
        la = str(row["PRE_tumor_laterality"]).strip().lower()
        pre_lo = str(row["PRE_tumor_location"]).strip().lower()
        pos_lo = str(row["POS_tumor_loc"]).strip().lower()
        pre_location_trans = transform_location(la, pre_lo)
        pos_location_trans = transform_location(la, pos_lo)
        if str(pre_location_trans) != "nan":
            print(f"Location trans of {la} {pre_lo}: {pre_location_trans}, patient: {row['PRE_record_id']}")
        if str(pre_location_trans) != "nan":
            print(f"Location trans of {la} {pos_lo}: {pos_location_trans}, patient: {row['PRE_record_id']}")
        pre_tumor_location_trans.append(pre_location_trans)
        pos_tumor_location_trans.append(pos_location_trans)
        # transformation for margins
        margin = row["PRE_closest_margin"]
        if str(margin) == "nan":
            margin_tran = np.nan
        else:
            margin_tran = len(re.split(",|and", str(margin)))
            print(f"{row['PRE_record_id']} margins_trans: {margin_tran}")
        margins_trans.append(margin_tran)
        
        # Use pre_op_biopsy_date_year or surgery_date_year - dob
        # to construct "age_at_dx" as the age at the time of diagnosis
        # dob = pd.to_datetime(row["PRE_dob"])
        # dx_date = pd.to_datetime(row["PRE_dximg_date"])
        dob = row["PRE_dob"]
        dx_date = row["PRE_dximg_date"]
        if str(dx_date) == "nan":
            age_at_dxs.append(np.nan)
        else:
            years_elapsed = round(dx_date - dob)
            age_at_dxs.append(years_elapsed)
            # print(f"{row['PRE_record_id']} age_at_dx: "
            #       f"{years_elapsed}, dob: {dob}, dx_date: {dx_date}")
        
        # Construct "age_at_surg"
        # surg_date = pd.to_datetime(row["PRE_surgery_date"])
        surg_date = row["PRE_surgery_date"]
        if str(surg_date) == "nan":
            age_at_surg.append(np.nan)
        else:
            years_elapsed = round(surg_date - dob)
            age_at_surg.append(years_elapsed)
            print(f"{row['PRE_record_id']} age_at_surg: "
                  f"{years_elapsed}, dob: {dob}, surg_date: {surg_date}")
        
        # Construct "bmis" as the BMI using PRE_height_cm and PRE_weight_kg
        if "nan" in [str(row["PRE_height_cm"]), str(row["PRE_weight_kg"])]:
            bmis.append(np.nan)
        else:
            bmis.append(
                row["PRE_weight_kg"] / ((row["PRE_height_cm"] / 100) ** 2)
            )

        # Construct "susp_LN_size_composite" as the maximum LN abnormality size
        max_size = 0
        for col in susp_LN_size_composite_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            max_size = max(max_size, value)
        susp_LN_size_composites.append(max_size)

        # Construct "susp_LN_prsnt_composite" to be the presence of abnormal LN
        susp_LN_prsnt_composite = 0
        for col in abnormal_ln_cols:
            value = row[col]
            if str(value) == "nan":
                continue
            else:
                if str(value).strip().replace(".0", "") == "1":
                    susp_LN_prsnt_composite = 1
        susp_LN_prsnt_composites.append(susp_LN_prsnt_composite)

        tumor_max_size_composite = np.nan
        for col in pre_tumor_max_size_composite_cols:
            value = row[col]
            if type(value) == str:
                value = re.sub("<|>", "", value)
                try:
                    value = float(value)
                except ValueError:
                    value = np.nan
            if str(value) == "nan":
                continue
            else:
                if col == "PRE_size_of_the_largest_foci_c":
                    value = value * 10
                if tumor_max_size_composite is np.nan:
                    tumor_max_size_composite = value
                else:
                    tumor_max_size_composite = max(tumor_max_size_composite, value)
        tumor_max_size_composites.append(tumor_max_size_composite)

    df.insert(
        list(df.columns).index("PRE_dob")+1,
        "PRE_age_at_dx",
        age_at_dxs
        )
    df.insert(
        list(df.columns).index("PRE_tumor_location")+1,
        "PRE_tumor_location_trans",
        pre_tumor_location_trans
    )
    df.insert(
        list(df.columns).index("PRE_closest_margin")+1,
        "PRE_num_closest_margins_trans",
        margins_trans
    )
    df.insert(
        list(df.columns).index("PRE_age_at_dx")+1,
        "PRE_age_at_surg",
        age_at_surg
        )
    df.insert(
        list(df.columns).index("PRE_height_cm")+1,
        "PRE_bmi",
        bmis
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
    df.insert(
        list(df.columns).index("POS_tumor_loc")+1,
        "POS_tumor_location_trans",
        pre_tumor_location_trans
    )

    # TODO Construct nomogram feature?
    

    # Converet all string cells to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # If a column is integral type, make it column int
    for col in df.columns:
        if is_integral_type(col):
            df[col] = df[col].astype("int")

    print_and_log("Converted all strings to numeric, and used NaN if impossible.")
    print_and_log(
        "✅ Feature Engineering - Added new feature 'PRE_age_at_dx', 'PRE_bmi'"
        "'PRE_susp_LN_size_composite',  'PRE_susp_LN_prsnt_composite', "
        "'PRE_tumor_max_size_composite'"
    )

    return df
