import config

top10_columns = [
    "PRE_tumor_max_size_composite",
    "PRE_susp_LN_size_composite",
    "PRE_his_subtype___dcis",
    "PRE_metastatic_carcinoma_on_ax",
    "PRE_systhe___chemo",
    "PRE_his_subtype___papillary",
    "PRE_er_status",
    "PRE_pr_status",
    "PRE_surg_indicat_prim___recurrent_cancer",
    "PRE_his_subtype___inv_mucinous"
]

nomogram_cols = [
    "PRE_age_at_dx",
    "PRE_age_at_surg",
    "PRE_tumor_max_size_composite",
    "PRE_his_subtype___inv_mucinous",
    "PRE_tumor_location_trns",
    "PRE_lymphovascular_invasion0",
    "PRE_foci",
    "PRE_his_subtype___idc",
    "PRE_his_subtype___ilc",
    "PRE_his_subtype___dcis",
    "PRE_his_subtype___lcis",
    "PRE_tumor_grade",
    "PRE_er_status",
    "PRE_pr_status"
]

class Experiment:
    def __init__(self, name: str, description: str, metadata_path: str, raw_df_path: str, processed_df_path: str, DPI: int=150):
        all_columns = Data.get_df("processed_PRE").columns

    
# def get_nomogram_columns(target_col):
#     return nomogram_cols + [target_col]

# target_columns = ["POS_metastasis", "POS_insitu_upstaged", "POS_did_the_patient_receive_pm", "POS_tumor_focality"] #, "POS_tu_grade", "POS_mar_status"]
target_columns = ["POS_metastasis", "POS_did_the_patient_receive_pm"]
subset_columns = [nomogram_cols, all_columns, top10_columns, top10_columns+nomogram_cols]
subset_columns_names = ["nomogram", "all", "top10", "top10_union_nomogram"]
# Deduplicate the columns and keep the original order
for i in range(len(subset_columns)):
    temp_columns = list(set(subset_columns[i]))
    temp_columns = sorted(temp_columns, key=lambda x: list(subset_columns[i]).index(x))
    subset_columns[i] = temp_columnsall_columns = Data.get_df("processed_PRE").columns
