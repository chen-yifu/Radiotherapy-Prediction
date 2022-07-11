from utils.io import print_and_log, print_and_log_w_header
from tqdm import tqdm


def rename_columns(df, df_metadata):
    print_and_log_w_header("Column Renaming...")


    rename_dict = {}
    # Append prefix and suffix to columns
    for col in tqdm(df.columns):
        print(col)
        row = df_metadata.loc[df_metadata["Original Field Name"] == col]
        prefix = row["Group"].item()
        rename_dict[col] = prefix+"_"+col
    df.rename(columns=rename_dict, inplace=True)

    # Perform renaming for specific columns
    rename_dict = {
        "PRE_gensus___1" : "PRE_gensus___brca1",
        "PRE_gensus___2" : "PRE_gensus___brca2",        
        "PRE_gensus___3" : "PRE_gensus___p53",
        "PRE_gensus___4" : "PRE_gensus___pten",
        "PRE_gensus___5" : "PRE_gensus___other",
        "PRE_gensus___6" : "PRE_gensus___unknown",
        "PRE_gensus___7" : "PRE_gensus___no_gensus",
        "PRE_systhe___1": "PRE_systhe___chemo",
        "PRE_systhe___2": "PRE_systhe___hormonal",
        "PRE_systhe___3": "PRE_systhe___chemo_and_hormonal",
        "PRE_systhe___4": "PRE_systhe___no_systhe",
        "PRE_systhe___5": "PRE_systhe___radiation",
        "PRE_dximg___1" : "PRE_dximg___mammography",
        "PRE_dximg___2" : "PRE_dximg___ultrasound",
        "PRE_dximg___3" : "PRE_dximg___mri",
        "PRE_his_subtype___1": "PRE_his_subtype___idc",
        "PRE_his_subtype___2": "PRE_his_subtype___ilc",
        "PRE_his_subtype___3": "PRE_his_subtype___dcis",
        "PRE_his_subtype___4": "PRE_his_subtype___lcis",
        "PRE_his_subtype___5": "PRE_his_subtype___inv_mucinous",
        "PRE_his_subtype___6": "PRE_his_subtype___papillary",
        "PRE_surgical_indication1_primary_treatment___1": "PRE_surg_indicat_prim___primary_tx",
        "PRE_surgical_indication1_primary_treatment___2": "PRE_surg_indicat_prim___reexcis_marg+_bcs",
        "PRE_surgical_indication1_primary_treatment___3": "PRE_surg_indicat_prim___compl_mast_marg+_bcs",
        "PRE_surgical_indication1_primary_treatment___4": "PRE_surg_indicat_prim___recurrent_cancer",
        "PRE_surgical_indication1_primary_treatment___5": "PRE_surg_indicat_prim___second_primary",
        "PRE_axillary_surgery___1": "PRE_ax_surg___no_ax_surg",
        "PRE_axillary_surgery___2": "PRE_ax_surg___sln_biopsy",
        "PRE_axillary_surgery___3": "PRE_ax_surg___ax_ln_dissect",
        "POS_his_type___1": "POS_his_type___idc",
        "POS_his_type___2": "POS_his_type___ilc",
        "POS_his_type___3": "POS_his_type___dcis",
        "POS_his_type___4": "POS_his_type___lcis",
        "POS_his_type___5": "POS_his_type___other",
        "POS_his_type___6": "POS_his_type___no_residual",
        "POS_his_type___7": "POS_his_type___inv_mucinous",
        "POS_his_type___8": "POS_his_type___inv_papillary",
        "POS_in_situ_component_type___1": "POS_in_situ_component_type___dcis",
        "POS_in_situ_component_type___2": "POS_in_situ_component_type___lcis",
        "POS_clos_margin___1": "POS_clos_margin___posterior",
        "POS_clos_margin___2": "POS_clos_margin___anterior",
        "POS_clos_margin___3": "POS_clos_margin___medial",
        "POS_clos_margin___4": "POS_clos_margin___lateral",
        "POS_clos_margin___5": "POS_clos_margin___inferior",
        "POS_clos_margin___6": "POS_clos_margin___superior",
        "POS_clos_margin___7": "POS_clos_margin___other",
        
    }

    # assert every col is present in df
    for col in rename_dict.keys():
        assert col in df.columns, f"Column {col} is not in df"

    df.rename(columns=rename_dict, inplace=True)
    

    print_and_log("✅ Column Renamming - Added PRE/INT/POS column name prefixes.")
