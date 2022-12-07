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
from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import config
from utils.printers import print_with_color, bcolors

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
        init_models=None,
        fit_models=True,
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
            print_with_color(f"Predicting target column {target_column} with {k_fold}-fold cross-validation.", color=bcolors.OKBLUE)
            print_with_color(f"Dropped {len(df_ready) - len(X)} rows with NaN in {target_column}; There are {len(X)} rows left.", color=bcolors.OKBLUE)
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
        if fit_models:
            print_with_color(f"The shape of TRAINING data is {X.shape}", color=bcolors.OKBLUE)
        else:
            print_with_color(f"The shape of TESTING data is {X.shape}", color=bcolors.OKBLUE)
        train_shape = X.shape
        if k_fold != -1:
            kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
            for i, (train_index, test_index) in enumerate(kf.split(X, y)):
                # if verbose:
                print_with_color(f"\n\nFold {i+1}...", bcolors.OKGREEN)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                X_train, X_test = X_train.drop("PRE_record_id", axis=1), X_test.drop("PRE_record_id", axis=1)
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # Standardize X_train and X_test
                if init_models is None:
                    models = self.initialize_models(use_full_model, verbose, classification_task=classification_task, seed=seed)
                else:
                    models = init_models
                    if i == 0:
                        print_with_color("Using pre-trained models", color=bcolors.OKGREEN)
                for name, (model, feature_score__key) in models.items():
                    if i == 0:
                        print_with_color(f"Using {name}...", color=bcolors.OKGREEN)
                    if fit_models:
                        model.fit(X_train, y_train)
                        if i == 0:
                            print_with_color(f"Trained {name} on 'other folds' to predict fold {i+1}.{bcolors.ENDC}", color=bcolors.OKGREEN)
                    else:
                        if i == 0:
                            print_with_color(f"Inferencing pre-trained {name} on 'fold' {i+1}.{bcolors.ENDC}", color=bcolors.OKGREEN)
                    
                    if fit_models:
                        if i == 0:
                            print(f"Train index: {train_index[:70]}...")
                    if i == 0:
                        print(f"Validation index: {test_index[:10]}...")
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
                        for feature, score in zip(X_train.columns, model.coef_[0]):
                            feature_scores[name].append((feature, score))
                    elif feature_score__key == 'feature_importances_':
                        for feature, score in zip(X_train.columns, model.feature_importances_):
                            feature_scores[name].append((feature, score))
                    elif feature_score__key == 'coefs_':
                        # Assumes only 1 hidden layer
                        for feature, score in zip(X_train.columns, model.coefs_[0]):
                            feature_scores[name].append((feature, score[0]))
                    elif feature_score__key == "best_estimator_feature_importances":
                        for feature, score in zip(X_train.columns, model.best_estimator_.feature_importances_):
                            feature_scores[name].append((feature, score))
                    elif feature_score__key == "best_estimator_coef_":
                        for feature, score in zip(X_train.columns, model.best_estimator_.coef_[0]):
                            feature_scores[name].append((feature, score))
                    elif callable(feature_score__key):
                        for feature, score in zip(X_train.columns, feature_score__key(model)):
                            feature_scores[name].append((feature, score))
        else:
            i = 0
            X = X.drop("PRE_record_id", axis=1)
            if init_models is None:
                models = self.initialize_models(use_full_model, verbose, classification_task=classification_task, seed=seed)
            else:
                models = init_models
                if i == 0:
                    print_with_color("Using pre-trained models", color=bcolors.OKGREEN)
            for name, (model, feature_score__key) in models.items():
                if i == 0:
                    print_with_color(f"Using {name}...", color=bcolors.OKGREEN)
                if fit_models:
                    model.fit(X, y)
                    if i == 0:
                        print_with_color(f"Trained {name} on all data.{bcolors.ENDC}", color=bcolors.OKGREEN)
                else:
                    if i == 0:
                        print_with_color(f"Inferencing pre-trained {name} on all data.{bcolors.ENDC}", color=bcolors.OKGREEN)
                if not fit_models:
                    # Model is already trained, so predict
                    y_prob = model.predict_proba(X)[:, 1]
                    y_class = model.predict(X)
                    predictions_prob_col = f"PRE_{target_column}_{name}_prob"
                    predictions_class_col = f"PRE_{target_column}_{name}_class"
                    prob_cols.add(predictions_prob_col)
                    class_cols.add(predictions_class_col)
                    model_names.add(name)
                    predictions[X.index, predictions_prob_col] = y_prob
                    predictions[X.index, predictions_class_col] = y_class
                    print_with_color(f"Predicted {name} on all data.{bcolors.ENDC}", color=bcolors.OKGREEN)
                else:
                    # Model is not trained, and we just trained it on all data, so no predictions to make
                    print(f"Trained {name} on all data, so no predictions to make.{bcolors.ENDC}")
                    pass
                # Save the feature scores
                if feature_score__key == 'coef_':
                    for feature, score in zip(X.columns, model.coef_[0]):
                        feature_scores[name].append((feature, score))
                elif feature_score__key == 'feature_importances_':
                    for feature, score in zip(X.columns, model.feature_importances_):
                        feature_scores[name].append((feature, score))
                elif feature_score__key == 'coefs_':
                    # Assumes only 1 hidden layer
                    for feature, score in zip(X.columns, model.coefs_[0]):
                        feature_scores[name].append((feature, score[0]))
                elif callable(feature_score__key):
                    for feature, score in zip(X.columns, feature_score__key(model)):
                        feature_scores[name].append((feature, score))
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
        }, train_shape
        
                
    def initialize_models(self, use_full_model: bool, verbose: bool, classification_task: bool, seed: int = 42) -> None:
        """Initialize the models.
        Args:
            use_full_model (bool): If True, allow higher amount of computation to find optimal models.
        """
        if use_full_model:
            N = config.predictor_N
            if verbose:
                print("Using full model.")
        else:
            N = 100
            if verbose:
                print("Using small model.")
        if classification_task:
            refit = True
            rf_algo = RandomForestClassifier(n_estimators=N, n_jobs=-1, random_state=seed+2)
            self.models = models = {
                "Logistic Regression": (LogisticRegressionCV(cv=10, max_iter=N, n_jobs=-1, random_state=seed+1, Cs=[float(1e4)], refit=refit), "coef_"),
                "Logistic Lasso": (LogisticRegressionCV(cv=10, max_iter=N, n_jobs=-1, penalty="l1", solver="saga", Cs=[0.05], random_state=seed+7, refit=refit), "coef_"),
                "Random Forest": (GridSearchCV(rf_algo, {'max_depth': [3, 5, 10], 'n_estimators': [N]}, cv=3, n_jobs=-1, refit=refit), "best_estimator_feature_importances"),
                "Elastic Net": (LogisticRegressionCV(cv=10, l1_ratios=[round(0.25 * i, 2) for i in range(1,4)], penalty='elasticnet', solver='saga', n_jobs=-1, Cs=[0.05], max_iter=N, random_state=seed+3, refit=refit), "coef_")
                # "Support Vector Machine": (SVC(probability=True, kernel="linear"), "coef_"),
                # "Gradient Boosting": (GradientBoostingClassifier(n_estimators=N, random_state=seed+4), "feature_importances_"),
                # "Etreme Gradient Boosting": (XGBClassifier(n_estimators=N, n_jobs=-1, random_state=seed+5), "feature_importances_"),
                # "Simple Neural Net": (MLPClassifier(max_iter=N, random_state=seed+6), "coefs_"),
                # "Ridge Regression": (SGDClassifier(loss="log_loss", penalty="l2"), "coef_")
            }
        else:
            raise NotImplementedError("Regression not implemented yet.")
        if verbose:
            print(f"Initialized {len(models)} models with {N} iterations/trees/estimators each.")
        return models

    def get_model_names(self, classification_task: str) -> list:
        """Get the names of the models."""
        return sorted(list(self.initialize_models(use_full_model=False, verbose=False, classification_task=classification_task).keys()))

