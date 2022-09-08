import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

class FeatureSelector:
    
    def __init__(self):
        config.FeatureSelector = self
        pass
    
    def recursive_feature_elim(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: object,
        model_name: str,
        n_features_to_select: int,
        CV_fold: int,
        step: float,
        importance_getter: str,
        verbose: bool):

        df = df.copy()
        # Drop rows in df if target_column is NaN
        df = df.dropna(axis=0, subset=[target_column])
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        print(y.isna().sum())
        assert len(X) == len(y)
        assert len(X) > 0
        assert len(y) > 0 
        assert y.isna().sum() == 0
        assert X.isna().sum().sum() == 0
        column_names = list(X.columns)

        if verbose:
            print(f"Using {model_name} to select top-{n_features_to_select} {f'with {CV_fold} fold cross-validation.' if CV_fold else ''}")
            print(f"At each step, we remove the {step if step >= 1 else str(step*100) + '%'} least significant feature(s).")
        # RFE = RFE(model, min_features_to_select=n_features_to_select, step=step, cv=CV_fold)
        rfe = RFE(model, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)

        if verbose:
            print(f"Feature selection completed.")
            assert len(rfe.support_) == len(column_names)
            selected_features = [x[1] for x in zip(rfe.support_, column_names)]
            print(f"Selected features: {selected_features}")

        result = pd.DataFrame(columns=['feature', 'rank', 'support'])
        result['feature'] = column_names
        result['rank'] = rfe.ranking_
        result['support'] = rfe.support_
        result.index = range(len(result))
        result = result.sort_values(by='rank', ascending=True)
        return result