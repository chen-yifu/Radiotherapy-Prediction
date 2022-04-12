import pandas as pd

imputation_dict = {
#     "bi_rads_score": "",
#     "tumor_stge": "",
    "abnormal_lymph": 2,
    "lymph_node_max_size_mm": 0,
    "extent_of_calcification_ma": 0,
    "prominent_axillary_lymph": 2,
    "backgroun_enhancement": 2,
    "max_enhancement_measurement": 0,
    "axillary_lymphadenopathy": 2,
    "internal_mammary_lymphaden": 2,
#     "high_grade_fdg_foci_presen": "",
#     "size_of_the_largest_foci_c": "",
    "axillary_lymphadenopathy_p": 0,
#     "axillary_lymph_node_max_si":"",
    "internal_mammry_lymph_nod": 0,
    "er_status": 0.5,
    "pr_status": 0.5,
    "her_status": 0.5,
    "axillary_lymph_node_core_b": 0,
}

def __init__(self):
    pass

def impute_cell(df, idx, col):
    if col == "men_status":
        return 1 if df["age"] > 50 else 0

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a column of a dataframe, modifies in place.
    """ 
    for column in df.columns:
        for i, row in enumerate(df[column]):
            if pd.isnull(row):
                df.at[i, column] = 1
            
    print(df)