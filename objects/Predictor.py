from collections import defaultdict
from collections import OrderedDict
from re import S
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import config

class Predictor:
    def __init__(self):
        self.models = {}
        config.Predictor = self
        pass
    
    @ignore_warnings(category=ConvergenceWarning)
    def predict(
        self,
        df_ready: pd.DataFrame,
        target_column: str,
        k_fold: int,
        verbose: bool=True,
        use_full_model=True,
        classification_task=True,
        seed=42,
        ) -> None:
        """
        Given a DataFrame ready to be trained, train ML models.
        Use stratified k-fold cross-validation.
        Args:
            df_ready (DataFrame): DataFrame ready to be trained.
            target_column (str): Name of the target column.
            k_fold (int): Number of folds.
            verbose (bool): Verbosity.
            subset_columns (list): List of columns to use, if None, use all columns.
            use_full_model (bool): If True, allow higher amount of computation to find optimal models.
        """
        VarReader = config.VarReader
        # Drop rows in df_ready if target_column is NaN
        X = df_ready.dropna(axis=0, subset=[target_column])
        y = X[target_column]
        X = X.drop(target_column, axis=1)
        # record_ids = X["PRE_record_id"]
        # X = X.drop("PRE_record_id", axis=1)
        if verbose:
            print(f"Predicting target column {target_column} with {k_fold}-fold cross-validation.")
            print(f"Dropped {len(df_ready) - len(X)} rows with NaN in {target_column}; There are {len(X)} rows left.")
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
        assert len(X) == len(y)
        assert len(X) > 0
        assert len(y) > 0
        assert y.isna().sum() == 0
        assert X.isna().sum().sum() == 0
        predictions = pd.DataFrame(index=df_ready.index)
        predictions["fold"] = None
        feature_scores = defaultdict(list)
        prob_cols = set()
        class_cols = set()
        model_names = set()
        print("The shape of training data is", X.shape)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            # if verbose:
            print(f"Fold {i+1}...", end=" ")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train, X_test = X_train.drop("PRE_record_id", axis=1), X_test.drop("PRE_record_id", axis=1)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Standardize X_train and X_test
            models = self.initialize_models(use_full_model, verbose, classification_task=classification_task, seed=seed)
            for name, (model, feature_score__key) in models.items():
                print(f"Training {name}...", end=" ")
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_class = model.predict(X_test)
                predictions_prob_col = f"PRE_{target_column}_{name}_prob"
                predictions_class_col = f"PRE_{target_column}_{name}_class"
                prob_cols.add(predictions_prob_col)
                class_cols.add(predictions_class_col)
                model_names.add(name)
                predictions.loc[X.index[test_index], predictions_prob_col] = y_prob
                predictions.loc[X.index[test_index], predictions_class_col] = y_class
                predictions.loc[X.index[test_index], "fold"] = i
                # Save the feature scores
                if feature_score__key == 'coef_':
                    for feature, score in zip(X_test.columns, model.coef_[0]):
                        feature_scores[name].append((feature, score))
                elif feature_score__key == 'feature_importances_':
                    for feature, score in zip(X_test.columns, model.feature_importances_):
                        feature_scores[name].append((feature, score))
                elif feature_score__key == 'coefs_':
                    # Assumes only 1 hidden layer
                    for feature, score in zip(X_test.columns, model.coefs_[0]):
                        feature_scores[name].append((feature, score[0]))
                elif callable(feature_score__key):
                    for feature, score in zip(X_test.columns, feature_score__key(model)):
                        feature_scores[name].append((feature, score))
                else:
                    raise ValueError(f"Unknown feature score key: {feature_score__key}")
        if verbose:
            print("\n")

        assert len(predictions) == len(df_ready), f"len(predictions)={len(predictions)}, len(df_ready)={len(df_ready)}"
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
            "model_names": sorted(list(model_names))
        }
        
                
    def initialize_models(self, use_full_model: bool, verbose: bool, classification_task: bool, seed: int = 42) -> None:
        """Initialize the models.
        Args:
            use_full_model (bool): If True, allow higher amount of computation to find optimal models.
        """
        if use_full_model:
            N = 1000
        else:
            N = 100
        if classification_task:
            models = {
                "logistic_reg": (LogisticRegressionCV(cv=5, max_iter=N, random_state=seed+1), "coef_"),
                "random_forest": (RandomForestClassifier(n_estimators=N, random_state=seed+2), "feature_importances_"),
                "elastic_net": (LogisticRegressionCV(cv=5, l1_ratios=[round(0.2 * i, 2) for i in range(0,6)],penalty='elasticnet', solver='saga', max_iter=N, random_state=seed+3), "coef_"),
                # "elastic_net_calibrated": (
                #     CalibratedClassifierCV(
                #         LogisticRegressionCV(cv=5, l1_ratios=[round(0.2 * i, 2) for i in range(0,6)],penalty='elasticnet', solver='saga', max_iter=N, random_state=seed+3*2),
                #         ensemble=True
                #     ), lambda x: x.calibrated_classifiers_[0].base_estimator.coef_[0]),
                # "elastic_net": (SGDClassifier(loss="log_loss", penalty="elasticnet"), "coef_"),
                "lasso": (SGDClassifier(loss="log_loss", penalty="l1"), "coef_"),
                "svm": (SVC(kernel='linear', probability=True), "coef_"),
                "gradient_boost": (GradientBoostingClassifier(n_estimators=N, random_state=seed+4), "feature_importances_"),
                "xgboost": (XGBClassifier(n_estimators=N, random_state=seed+5), "feature_importances_"),
                "neural_net": (MLPClassifier(max_iter=N, random_state=seed+6), "coefs_"),
                # "els_net_old": (LogisticRegression(C=0, penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=N, random_state=seed), "coef_"),
                # "ridge": (SGDClassifier(loss="log_loss", penalty="l2"), "coef_"),
            }
        else:
            raise NotImplementedError("Regression not implemented yet.")
        if verbose:
            print(f"Initialized {len(models)} models with {N} iterations/trees each.")
        return models

    def get_model_names(self, classification_task: str) -> list:
        """Get the names of the models."""
        return sorted(list(self.initialize_models(use_full_model=False, verbose=False, classification_task=classification_task).keys()))

