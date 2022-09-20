# Impute missing values using sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import config
from IPython.display import display, HTML


class DataProcessor:

    def __init__(self) -> None:
        config.DataProcessor = self
        pass
    
    @ignore_warnings(category=ConvergenceWarning)
    def impute(
        self,
        df_unimputed: pd.DataFrame,
        target_column: str,
        max_iter: int=10,
        verbose: bool=False,
        seed=42,
        ) -> pd.DataFrame:
        """Given a DataFrame with missing values, impute them using IterativeImputer.
        Leave the target_column as is.

        Args:
            df_unimputed (DataFrame): DataFrame with missing values.
            target_column (str): Name of the target column.
            max_iter (int): Maximum number of iterations.
            verbose (bool): Verbosity.
        """
        # If there is a column with all-NaNs, drop the column
        if df_unimputed.isna().all().any():
            cols_to_drop = df_unimputed.columns[df_unimputed.isna().all()]
            df_unimputed = df_unimputed.drop(cols_to_drop, axis=1)
            print(f"Dropping all-NaN column: {list(cols_to_drop)}")
        if verbose:
            num_na = df_unimputed.isna().sum().sum()
            print(f"Started with {num_na} missing values in dataframe.")
        
        assert df_unimputed.columns.str.startswith("POS_").any() <= 1
        
        imp = IterativeImputer(max_iter=max_iter, random_state=seed, verbose=verbose)
        if target_column != None:
            X = df_unimputed.drop(target_column, axis=1)
        else:
            X = df_unimputed

        X = X.dropna(axis=1, how="all")
        X_imputed = imp.fit_transform(X)
        df_imputed = pd.DataFrame(X_imputed, columns=X.columns)
        
        if target_column != None:
            df_imputed[target_column] = df_unimputed[target_column]
        
        if verbose:
            print(f"Standardized {len(list(X.columns))} columns", end=" ")
            if target_column != None:
                print(f" except {target_column} target column was untouched.")        

        return df_imputed

    def standardize(
        self,
        df_unstandardized: pd.DataFrame,
        target_column: str,
        verbose: bool=False
        ) -> pd.DataFrame:
        """
        Given a DataFrame with unstanardized values, standardize them using StandardScaler.
        Leave the target_column as is.
        
        Args:
            df_unstandardized (DataFrame): DataFrame with unstanardized values.
            target_column (str): Name of the target column.
        """
        # If there is a column with all-NaNs, drop the column
        if df_unstandardized.isna().all().any():
            cols_to_drop = df_unstandardized.columns[df_unstandardized.isna().all()]
            df_unstandardized = df_unstandardized.drop(cols_to_drop, axis=1)
            print(f"Dropping all-NaN column: {list(cols_to_drop)}")

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

        if target_column != None:
            X = df_unstandardized.drop(target_column, axis=1)
        else:
            X = df_unstandardized

        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        if target_column != None:
            y = df_unstandardized[target_column]
            df_scaled[target_column] = y
        if verbose:
            print(f"Standardized {len(list(X.columns))} columns", end="")
            if target_column != None:
                print(f" except {target_column} target column was untouched.")
                    
        return df_scaled
    
    
    def generate_ready_df(
        self, 
        target_column, 
        experiment_name, 
        seed, 
        use_PRE_only, 
        filter,
        do_impute=True, 
        impute_max_iter=10, 
        verbose=False, 
        subset_cols=None, 
        cols_to_exclude=[],
    ) -> pd.DataFrame:
        """Generate a DataFrame ready for training.

        Args:
            target_column (str): Name of the target column.
            experiment_name (str): Name of the experiment.
            seed (int): Random seed.
            use_PRE_only (bool): Whether to use PRE only.
            filter (function): Filter to apply to the data.
            do_impute (bool, optional): Whether to impute missing values. Defaults to True.
            impute_max_iter (int, optional): Maximum number of iterations for imputation. Defaults to 10.
            verbose (bool, optional): Verbosity. Defaults to False.
            subset_cols (_type_, optional): Subset of columns to use. Defaults to None.
            cols_to_exclude (list, optional): Columns to exclude. Defaults to [].

        Returns:
            pd.DataFrame: DataFrame ready for training.
        """
        
        Data = config.Data
        df_name = experiment_name
        
        df = Data.get_df("processed").copy()
        if filter is not None:
            df = filter(df)
        if cols_to_exclude:
            df = df.drop([col for col in cols_to_exclude if col in df.columns], axis=1)
        if use_PRE_only:
            df = df[[col for col in df.columns if col.startswith("PRE_")] + [target_column]]
        if subset_cols is not None:
            if "PRE_record_id" in subset_cols:
                df = df[[col for col in subset_cols if col in df.columns]+[target_column]]
            else:
                df = df[[col for col in subset_cols if col in df.columns]+[target_column, "PRE_record_id"]]
        
        df = df.reset_index(drop=True)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        # display(df)
        # print("df shape:", df.shape)
        # print("X.shape, y.shape", X.shape, y.shape)
        X = self.standardize(X, target_column=None, verbose=verbose)
        # print("X.shape after standardization", X.shape)
        X = self.impute(X, target_column=None, max_iter=impute_max_iter, verbose=True, seed=seed)
        # print("X.shape after imputation", X.shape)
        X_and_y = pd.concat([X, y], axis=1)
        # print("X_and_y shape:", X_and_y.shape)
        # X_and_y = X_and_y.dropna(axis=0, how="any")
        # if verbose and X_and_y.isna().sum().sum() > 0:
        #     print(f"Warning: {X_and_y.isna().sum().sum()} missing values in dataframe.")
        # assert X_and_y.isna().sum().sum() == 0
        for col in cols_to_exclude:
            assert col not in X_and_y.columns
        
        df_name = Data.add_df(X_and_y, df_name, is_PRE_only=use_PRE_only, is_PRE_and_POS=not use_PRE_only, is_ready=True)
        print(f"There are {len(X_and_y.columns)} columns in the ready DataFrame.")
        
        return Data.get_df(df_name)
    
    
    # def remove_correlated_features(
    #     self,
    #     df_orig: pd.DataFrame,
    #     ...
    # )