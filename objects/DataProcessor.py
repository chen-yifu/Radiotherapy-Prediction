# Impute missing values using sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd


class DataProcessor:

    def __init__(self) -> None:
        pass
    
    @ignore_warnings(category=ConvergenceWarning)
    def impute(
        self,
        df_unimputed: pd.DataFrame,
        target_column: str,
        max_iter: int=10,
        verbose: bool=False
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
            
        imp = IterativeImputer(max_iter=max_iter, random_state=0, verbose=verbose)
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
            print(f"Standardized {len(list(X.columns))} columns", end="")
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
    