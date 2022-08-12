from collections import defaultdict
from collections import OrderedDict
from re import S
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import config

class Predictor:
    def __init__(self):
        
        pass
    
    def predict(
        self,
        df_ready: pd.DataFrame,
        target_column: str,
        k_fold: int,
        verbose: bool=False
        ) -> None:
        """
        Given a DataFrame ready to be trained, train ML models.
        Use stratified k-fold cross-validation.
        Args:
            df_ready (DataFrame): DataFrame ready to be trained.
            target_column (str): Name of the target column.
            verbose (bool): Verbosity.
        """
        VarReader = config.VarReader
        # Drop rows in df_ready if target_column is NaN
        X = df_ready.dropna(axis=0, subset=[target_column])
        y = X[target_column]
        X = X.drop(target_column, axis=1)
        if verbose:
            print(f"Dropped {len(df_ready) - len(X)} rows with NaN in {target_column}; There are {len(X)} rows left.")
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        assert len(X) == len(y)
        assert len(X) > 0
        assert len(y) > 0
        assert y.isna().sum() == 0
        assert X.isna().sum().sum() == 0
        predictions = pd.DataFrame(index=df_ready.index)
        feature_scores = defaultdict(list)
        prob_cols = set()
        class_cols = set()
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            if verbose:
                print(f"Fold {i+1}...", end=" ")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            models = self.initialize_models()
            for name, (model, feature_score__key) in models.items():
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_class = model.predict(X_test)
                predictions_prob_col = f"PRE_{target_column}_{name}_prob"
                predictions_class_col = f"PRE_{target_column}_{name}_class"
                prob_cols.add(predictions_prob_col)
                class_cols.add(predictions_class_col)
                predictions.loc[X.index[test_index], predictions_prob_col] = y_prob
                predictions.loc[X.index[test_index], predictions_class_col] = y_class
                # Save the feature scores
                if feature_score__key == 'coef_':
                    for feature, score in zip(X_test.columns, model.coef_[0]):
                        feature_scores[name].append((feature, score))
                elif feature_score__key == 'feature_importances_':
                    for feature, score in zip(X_test.columns, model.feature_importances_):
                        feature_scores[name].append((feature, score))
                else:
                    raise ValueError(f"Unknown feature score key: {feature_score__key}")
        if verbose:
            print()

        result_df = pd.concat([predictions, df_ready], axis=1)
        # Sort the scores using OrderedDict by absolute value
        for name, scores in feature_scores.items():
            feature_scores[name] = OrderedDict(sorted(scores, key=lambda x: abs(x[1]), reverse=True))
        temp = pd.DataFrame({"coeff": [], "absolute": [], "corr w/ targ": [], "encoding": []})
        for name, scores in feature_scores.items():
            for feature, score in scores.items():
                encoding = VarReader.read_var_attrib(target_column, VarReader.has_missing(df_ready, target_column))["options_str"]
                # Concatenate together
                temp_row = {"coeff": score, "absolute": abs(score), "corr w/ targ": df_ready[feature].corr(df_ready[target_column]), "encoding": encoding}
                temp = pd.concat([temp, pd.DataFrame(temp_row, index=[feature])], axis=1)

        return { 
            "result_df": result_df,
            "feature_scores": feature_scores,
            "prob_columns": sorted(list(prob_cols)),
            "class_columns": sorted(list(class_cols)),
        }
        
                
    def initialize_models(self) -> None:
        """Initialize the models."""
        logreg = LogisticRegression(max_iter=2000, random_state=42)
        en_classifier = LogisticRegression(C=0.5, penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=2000, random_state=42)
        rf = RandomForestClassifier(n_estimators=3000, random_state=42)
        models = {
            "log_reg": (logreg, "coef_"),
            "els_net": (en_classifier, "coef_"),
            "rnd_fst": (rf, "feature_importances_")
        }
        return models

    def get_model_names(self) -> list:
        """Get the names of the models."""
        return list(self.initialize_models().keys())

