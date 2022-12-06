from collections import defaultdict
import pandas as pd
import config
import random
import numpy as np
class InclusionCriteria():

    def __init__(
        self, 
        # exclude_neoadjuvant: bool, 
        # require_sln_biopsy: bool, 
        # require_invasive: bool,
        # require_noninvasive: bool,
        # require_nomogram_prob: bool,
        meets_nomogram: bool,
        dropout_rate: float,
        invasiveness: str,
        exclude_pre_ln_positive: bool, 
        exclude_neo_rt: bool,
        # take_complement: bool,
        # key: str=None
    ):
        """Initialize InclusionCriteria object.

        Args:
            exclude_neoadjuvant (bool): Whether to exclude neoadjuvant cases.
            exclude_pre_ln_positive (bool): Whether to exclude pre-LN positive cases.
            require_sln_biopsy (bool): Whether to require SLN biopsy.
            require_invasive (bool): Whether to require invasive histology.
            require_noninvasive (bool): Whether to require noninvasive histology.
            require_nomogram_prob (bool): Whether to require nomogram probability.
            dropout_rate (float): Dropout rate of randomly removing cases who meet inclusion criteria.
            key (str, optional): Key to save eligibility criteria to. Defaults to None.
            take_complement (bool, optional): Whether to take complement of the Nomogram Inclusion criteria. Defaults to False.
        """
        # def take_complement(x):
        #     if x is None:
        #         return False
        #     else:
        #         return not x
            
        # self.exclude_neoadjuvant = exclude_neoadjuvant if not take_complement else not exclude_neoadjuvant
        self.exclude_pre_ln_positive = exclude_pre_ln_positive # if not take_complement else not exclude_pre_ln_positive
        self.exclude_neo_rt = exclude_neo_rt
        # self.require_sln_biopsy = require_sln_biopsy if not take_complement else not require_sln_biopsy
        # self.require_invasive = require_invasive if not take_complement else not require_invasive
        # self.require_noninvasive = require_noninvasive
        # self.require_nomogram_prob = require_nomogram_prob
        self.dropout_rate = dropout_rate
        self.meets_nomogram = meets_nomogram
        self.invasiveness = invasiveness
        # self.key = key
        # self.take_complement = take_complement
        # Schema: {experiment_name: {case_id_1: bool}, ...}
        self.eligibility_dict = defaultdict(lambda: defaultdict(lambda: True))
        # self.criteria_str = f"meets_nomogram={self.meets_nomogram}, invasiveness={self.invasiveness}{'dropout_rate=' + str(self.dropout_rate) if self.dropout_rate else ''}, exclude_pre_ln_positive={self.exclude_pre_ln_positive}"
        self.criteria_str = ""
        attrs = [self.meets_nomogram, self.invasiveness, self.exclude_pre_ln_positive, self.exclude_neo_rt, self.dropout_rate]
        attr_names = ["meets_nomogram", "invasiveness", "exclude_pre_ln_positive", "exclude_neo_rt", "dropout_rate"]
        self.criteria_str = ','.join([f"{attr_names[i]}={attr}" for i, attr in enumerate(attrs) if attr])
        # self.true_criteria = [attr for attr in self.__dict__]
        # self.true_criteria_str = ", ".join(self.true_criteria)
        
    
    # def __init__(
    #     self, 
    #     neoadjuvant: str, 
    #     pre_ln_positive: str, 
    #     sln_biopsy: str, 
    #     hist_invasive: bool,
    #     nomogram_prob: bool,
    #     dropout_rate: float,
    #     key: str=None,
    #     take_complement: bool=False
    # ):
    #     """Initialize InclusionCriteria object.

    #     Args:
    #         dropout_rate (float): Dropout rate of randomly removing cases who meet inclusion criteria.
    #         key (str, optional): Key to save eligibility criteria to. Defaults to None.
    #         take_complement (bool, optional): Whether to take complement of the above criteria. Defaults to False.
    #     """
        
            
    #     neoadjuvant = neoadjuvant
    #     self.dropout_rate = dropout_rate
    #     self.key = key
    #     self.take_complement = take_complement
    #     # Schema: {experiment_name: {case_id_1: bool}, ...}
    #     self.eligibility_dict = defaultdict(lambda: defaultdict(bool))
    #     self.true_criteria = [attr for attr in self.__dict__ if self.__dict__[attr]]
    #     self.true_criteria_str = ", ".join(self.true_criteria)
    
        
    def __str__(self) -> str:
        return f"({self.criteria_str})"
    # def filter(self, df: pd.DataFrame, verbose=1) -> pd.DataFrame:
    #     """Given a dataframe, return a filtered dataframe based on the inclusion criteria.

    #     Args:
    #         df ( pd.DataFrame ): DataFrame to filter.
    #         verbose ( int, optional): Verbosity level. Defaults to 1.

    #     Returns:
    #         pd.DataFrame: Filtered DataFrame.
        
    #     Note: 
    #     If DataFrame was standardized, hence values are not necessarily 0 or 1.
    #     Since original values are 0 or 1, they will be negative and positive, respectively.
    #     """
    #     orig_shape = df.shape
    #     if self.exclude_neoadjuvant:
    #         df = df[df["PRE_systhe___no_systhe"] > 0]
    #         print(f"Excluding cases with neoadjuvant systemic therapy. {df.shape[0]} cases remain.") if verbose > 0 else None
    #     if self.exclude_pre_ln_positive:
    #         df = df[df["PRE_susp_LN_prsnt_composite"] <= 0]
    #         print(f"Excluding pre-LN positive cases. {df.shape[0]} cases remain.") if verbose > 0 else None
    #     if self.require_sln_biopsy:
    #         df = df[df["POS_ax_surg___sln_biopsy"] > 0]
    #         print(f"Excluding cases without SLN biopsy. {df.shape[0]} cases remain.") if verbose > 0 else None
    #     if self.require_invasive:
    #         df = df[df["PRE_his_subtype_is_invasive_composite"] > 0]
    #         print(f"Excluding cases without invasive histology. {df.shape[0]} cases remain.") if verbose > 0 else None
            
    #     if verbose > 0:
    #         print(f"Filtered DataFrame from shape {orig_shape} to {df.shape}")
            
    #     return df
    
    
    # def add_filter_col(self, df: pd.DataFrame, verbose=1) -> pd.DataFrame:
    #     """Given a dataframe, add a column for whether each case meets the inclusion criteria.

    #     Args:
    #         df ( pd.DataFrame ): DataFrame to filter.
    #         verbose ( int, optional): Verbosity level. Defaults to 1.

    #     Returns:
    #         pd.DataFrame: Filtered DataFrame.
        
    #     Note: 
    #     If DataFrame was standardized, hence values are not necessarily 0 or 1.
    #     Since original values are 0 or 1, they will be negative and positive, respectively.
    #     """
    #     orig_shape = df.shape
    #     meets_criteria = pd.Series([True] * len(df))
    #     for i, row in df.iterrows():
    #         if self.exclude_neoadjuvant and row["PRE_systhe___no_systhe"] <= 0:
    #             meets_criteria[i] = False
    #         if self.exclude_pre_ln_positive and row["PRE_susp_LN_prsnt_composite"] > 0:
    #             meets_criteria[i] = False
    #         if self.require_sln_biopsy and row["POS_ax_surg___sln_biopsy"] <= 0:
    #             meets_criteria[i] = False
    #         if self.require_invasive and row["PRE_his_subtype_is_invasive_composite"] <= 0:
    #             meets_criteria[i] = False
    #     df["MEETS_CRITERIA"] = meets_criteria
    #     if verbose > 0:
    #         print(f"Added MEETS_CRITERIA column to DataFrame. {sum(meets_criteria)}/{len(meets_criteria)} cases meet criteria.")
            
    #     return df
    
    
    def check_row_meets_inclusion(self, row):
        """Given a row of a DataFrame, return whether it meets the inclusion criteria.

        Args:
            row ( pd.Series ): Row of a DataFrame.

        Returns:
            bool: Whether the row meets the inclusion criteria.
        """
        needs_meet_nomogram = self.meets_nomogram
        needs_invasiveness = self.invasiveness

        if (needs_invasiveness == "invasive" and row["PRE_his_subtype_is_invasive_composite"] <= 0) or \
            (needs_invasiveness == "non-invasive" and row["PRE_his_subtype_is_invasive_composite"] > 0):
                return False
            
        record_meets_nomogram = True
        if row["PRE_systhe___no_systhe"] <= 0 or row["PRE_susp_LN_prsnt_composite"] > 0 \
            or row["POS_ax_surg___sln_biopsy"] <= 0 or row["PRE_his_subtype_is_invasive_composite"] <= 0:
                record_meets_nomogram = False
        if (needs_meet_nomogram == True and not record_meets_nomogram) or \
            (needs_meet_nomogram == False and record_meets_nomogram):
                return False

        # return True
        
        # if self.exclude_neoadjuvant and row["PRE_systhe___no_systhe"] <= 0:
        #     return #False #if not take_complement else True
        if self.exclude_pre_ln_positive and row["PRE_susp_LN_prsnt_composite"] > 0:
            return False #if not take_complement else True
        if self.exclude_neo_rt and row["PRE_systhe___radiation"] > 0:
            return False
        # if self.require_sln_biopsy and row["POS_ax_surg___sln_biopsy"] <= 0:
        #     return False #if not take_complement else True
        # if self.require_invasive and row["PRE_his_subtype_is_invasive_composite"] <= 0:
        #     return False #if not take_complement else True
        # if self.require_noninvasive and row["PRE_his_subtype_is_invasive_composite"] > 0:
        #     return False #if not take_complement else True
        # if self.require_nomogram_prob and str(row["PRE_sln_met_nomogram_prob"]) == "nan":
        #     return False #if not take_complement else True
        return True #if not take_complement else False
        
    
     
    def save_records_eligibility(self, df: pd.DataFrame, key: str, verbose=1) -> pd.DataFrame:
        """Given a dataframe, filter the dataframe based on the inclusion criteria, and save whether each case meets the criteria.

        Args:
            df ( pd.DataFrame ): DataFrame to filter.
            verbose ( int, optional): Verbosity level. Defaults to 1.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        
        Note: 
        If DataFrame was standardized, hence values are not necessarily 0 or 1.
        Since original values are 0 or 1, they will be negative and positive, respectively.
        """
        orig_shape = df.shape
        eligibility_dict = self.eligibility_dict
        if verbose:
            print(f"Saving record eligibility based on inclusion criteria key {key}")
        
        if key in eligibility_dict:
            print(f"Key {key} already exists in eligibility_dict, so overwriting.")
            
        for i, row in df.iterrows():
            if self.dropout_rate > 0:
                to_drop = np.random.choice([False, True], p=[1-self.dropout_rate, self.dropout_rate])
            else:
                to_drop = False
            eligibility_dict[key][row["PRE_record_id"]] = self.check_row_meets_inclusion(row) and not to_drop
        
        if verbose:
            print(f"Saved {sum(eligibility_dict[key].values())} cases that meet the inclusion criteria.")

        return eligibility_dict[key].copy()
    
    def filter_records_by_eligibility(self, df: pd.DataFrame, key: str, verbose=1) -> pd.DataFrame:
        """Given a dataframe, filter the dataframe based on the inclusion criteria, and save whether each case meets the criteria.

        Args:
            df ( pd.DataFrame ): DataFrame to filter.
            verbose ( int, optional): Verbosity level. Defaults to 1.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        
        Note: 
        If DataFrame was standardized, hence values are not necessarily 0 or 1.
        Since original values are 0 or 1, they will be negative and positive, respectively.
        """
        orig_df = df.copy()
        orig_shape = orig_df.shape
        eligibility_dict = self.eligibility_dict
        
        if verbose:
            print(f"Filtering records based on inclusion criteria name {key}")
            
        if key not in eligibility_dict:
            raise ValueError(f"Key name {key} not found in eligibility_dict. Please run save_records_eligibility first.")
        
        df = df[df["PRE_record_id"].map(eligibility_dict[key])]
        # if verbose > 0:
        #     print(orig_df.head())
        #     for i, row in orig_df.iterrows():
        #         if row["PRE_record_id"] not in eligibility_dict[key]:
        #             print(f"Record {row['PRE_record_id']} not in eligibility_dict.")
        #         if i > 10:
        #             break
        #     print(f"Filtered DataFrame from shape {orig_shape} to {df.shape}")
            
        return df
    
    def get_eligibility_dict(self, standardized):
        """Return a dictionary of standardized eligibility criteria.

        Returns:
            dict: Dictionary of standardized eligibility criteria.
        """
        if standardized:
            return self.eligibility_dict["standardized"]
        else:
            return self.eligibility_dict["original"]
    
    def print_criteria(self):
        # For each attribute in self, print the attribute name and value.
        for attr in self.__dict__:
            print(f"{attr}: {self.__dict__[attr]}")
        