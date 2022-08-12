import pandas as pd
import config
import re
from collections import OrderedDict
# Import modules for calculating the metrics of prediction
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
# Import libraries for plotting heatmaps and box plots  
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

class Evaluator:
    
    def evaluate_predictions(self, df: pd.DataFrame, pred_col: str, target_column: str, threshold: float, title: str, show_results=True):
        # Drop rows where the target_col is NaN
        df = df.copy()
        df = df.dropna(axis=0, subset=[target_column])
        # Convert pred_col to float type
        df[pred_col] = df[pred_col].astype(float)
        pred_col_binary = pred_col + "_binary"
        # Convert pred_col to binary 1 or 0 based on threshold, save the binary pred_col_binary, using .loc to suppress warnings
        df[pred_col_binary] = df[pred_col] > threshold
        # Filter the df by having non-nan values for the nomogram probability
        df_prediction = df[~df[pred_col].isna()]
        is_multiclass = len(df_prediction[target_column].unique()) > 2
        # Assert there's no nan values in the nomogram probability or the metastasis columns
        assert df_prediction[pred_col].isna().sum() == 0
        assert df_prediction[target_column].isna().sum() == 0
        # Calculate the accuracy, sensitivity, specificity, F1, and AUC metrics
        if is_multiclass:
            ytest = label_binarize(df_prediction[target_column], classes=df_prediction[target_column].unique())
            ypred = label_binarize(df_prediction[pred_col_binary], classes=df_prediction[target_column].unique())
            auc_score = roc_auc_score(ytest, ypred, average="weighted", multi_class="ovr")
            precision, recall, thresholds = 0, 0, 0
        else:
            ytest = df_prediction[target_column]
            ypred = df_prediction[pred_col_binary]
            auc_score = roc_auc_score(ytest, ypred, average="weighted")
            precision, recall, thresholds = precision_recall_curve(ytest, ypred, pos_label=1)
        accuracy = accuracy_score(ytest, ypred)
        f1 = f1_score(ytest, ypred, average="weighted")
        precision = precision_score(ytest, ypred, average="weighted")
        recall = recall_score(ytest, ypred, average="weighted")
        specificity = recall_score(ytest, ypred, average="weighted")
        cls_report = classification_report(df_prediction[target_column], df_prediction[pred_col_binary], zero_division=0)
            
        result = {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "auc": round(auc_score, 4),
            "classification_report": cls_report,
            "df_prediction": df_prediction,
            "pred_col_binary": pred_col_binary,
            "pred_col": pred_col,
            "truth_col": target_column
        }
        if show_results:
            print(f"{'-'*20} {title} {'-'*20}")
            print(f"Accuracy: {result['accuracy']}, ", f"F1: {result['f1']}, ", f"AUC: {result['auc']} ")
            print(f"{result['classification_report']}")

        return result

    def find_threshold(self, df, pred_col, truth_col, metric="accuracy"):
        best_metric = 0
        best_threshold = 0
        for i in range(0, 100, 1):
            threshold = i / 100
            result = self.evaluate_predictions(df, pred_col, truth_col, threshold, "", show_results=False)
            accuracy, f1 = result["accuracy"], result["f1"]
            if metric == "accuracy":
                if accuracy > best_metric:
                    best_metric = accuracy
                    best_threshold = threshold
            elif metric == "f1":
                if f1 > best_metric:
                    best_metric = f1
                    best_threshold = threshold
            else:
                raise ValueError("metric must be either 'accuracy' or 'f1'")
        print(f"Threshold for best {metric}: {best_threshold}")
        return best_threshold
