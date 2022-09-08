import pandas as pd


class InclusionCriteria():

    def __init__(
        self, 
        exclude_neoadjuvant: bool, 
        exclude_pre_ln_positive: bool, 
        require_sln_biopsy: bool, 
        require_invasive: bool
    ):
        self.exclude_neoadjuvant = exclude_neoadjuvant
        self.exclude_pre_ln_positive = exclude_pre_ln_positive
        self.require_sln_biopsy = require_sln_biopsy
        self.require_invasive = require_invasive
        
    def filter(self, df: pd.DataFrame, verbose=1) -> pd.DataFrame:
        """Given a dataframe, return a filtered dataframe based on the inclusion criteria.

        Args:
            df ( pd.DataFrame ): DataFrame to filter.
            verbose ( int, optional): Verbosity level. Defaults to 1.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        orig_shape = df.shape
        if self.exclude_neoadjuvant:
            df = df[df["PRE_systhe___no_systhe"] == 1]
            print(f"Excluding cases with neoadjuvant systemic therapy. {df.shape[0]} cases remain.") if verbose > 0 else None
        if self.exclude_pre_ln_positive:
            df = df[df["PRE_susp_LN_prsnt_composite"] == 0]
            print(f"Excluding pre-LN positive cases. {df.shape[0]} cases remain.") if verbose > 0 else None
        if self.require_sln_biopsy and "POS_ax_surg___sln_biopsy" in df.columns:
            df = df[df["POS_ax_surg___sln_biopsy"] == 1]
            print(f"Excluding cases without SLN biopsy. {df.shape[0]} cases remain.") if verbose > 0 else None
        if self.require_invasive:
            df = df[df["PRE_his_subtype_is_invasive_composite"] == 1]
            print(f"Excluding cases without invasive histology. {df.shape[0]} cases remain.") if verbose > 0 else None
            
        if verbose > 0:
            print(f"Filtered DataFrame from shape {orig_shape} to {df.shape}")
            
        return df