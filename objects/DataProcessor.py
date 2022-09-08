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
        Data = config.Data
        df_name = experiment_name
        
        if use_PRE_only:
            X = Data.get_df("processed_PRE")
        else:
            X = Data.get_df("processed")
        if target_column in X.columns:
            X.drop(target_column, axis=1, inplace=True)
        if filter:
            orig_shape = X.shape
            X = filter(X)
            print(f"Filtered {orig_shape[0] - X.shape[0]} rows, resulting in shape: {X.shape}")
        
        temp_y = Data.get_df("processed")
        y = temp_y[temp_y["PRE_record_id"].isin(X["PRE_record_id"])][target_column]
        X = self.standardize(X, target_column=None, verbose=verbose)
        Data.add_df(X, experiment_name, is_standardized=True)
        X = self.impute(X, target_column=None, max_iter=impute_max_iter, verbose=True, seed=seed)    
        X_and_y = pd.concat([X, y], axis=1)
        # display(X_and_y)
        X_and_y = X_and_y.dropna(axis=0, subset=[target_column])
        # Drop rows with all NaNs
        X_and_y = X_and_y.dropna(axis=0, how="all")
        if subset_cols is not None:
            X_and_y = X_and_y[[col for col in subset_cols if col in X_and_y.columns]+[target_column]]
        if cols_to_exclude:
            X_and_y = X_and_y.drop([col for col in cols_to_exclude if col in X_and_y.columns], axis=1)
        # Drop rows with NaNs from X_and_y
        X_and_y = X_and_y.dropna(axis=0, how="any")
        
        # display(X_and_y)
        if verbose and X_and_y.isna().sum().sum() > 0:
            print(f"Warning: {X_and_y.isna().sum().sum()} missing values in dataframe.")
            # display(X_and_y)
        
        assert X_and_y.isna().sum().sum() == 0
        
        for col in cols_to_exclude:
            assert col not in X_and_y.columns
        
        df_name = Data.add_df(X_and_y, df_name, is_PRE_only=use_PRE_only, is_PRE_and_POS=not use_PRE_only, is_ready=True)
        print(f"The shape of the {df_name} dataframe is {X_and_y.shape}")
        
        return Data.get_df(df_name)
    
    
    # def remove_correlated_features(
    #     self,
    #     df_orig: pd.DataFrame,
    #     ...
    # )