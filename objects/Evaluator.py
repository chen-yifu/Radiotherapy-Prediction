import config
import re
import time
import os
from collections import OrderedDict
from objects.InclusionCriteria import InclusionCriteria
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import HuberRegressor
from IPython.display import display, HTML
import scipy.stats as stats
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
        
pd.options.mode.chained_assignment = None

class Evaluator:

    def __init__(self) -> None:
        self.VarReader = config.VarReader
        config.Evaluator = self
        pass
    
    def plot_cross_tab(self, df, col1, col2, ax, plot_missing=False, threshold=None):
        VarReader = self.VarReader
        # for col1 in cross_tab_cols_set1:
        var1_dict = VarReader.read_var_attrib(col1, has_missing=VarReader.has_missing(df, col1))
        section1, dtype1, label1, options1, options_str1 = var1_dict["section"], var1_dict["dtype"], var1_dict["label"], var1_dict["options"], var1_dict["options_str"]
        # for col2 in cross_tab_cols_set2:
        var2_dict = VarReader.read_var_attrib(col2, has_missing=VarReader.has_missing(df, col2))
        section2, dtype2, label2, options2, options_str2 = var2_dict["section"], var2_dict["dtype"], var2_dict["label"], var2_dict["options"], var2_dict["options_str"]

        # Create cross-tab table
        if VarReader.is_dtype_categorical(dtype1) and VarReader.is_dtype_categorical(dtype2):
            ax.set_title(f"{col1} ({section1})\nvs\n{col2} ({section2})", fontsize=24)
            # continue
            # If both are categorical, create a contingency table with percentages and margins
            df_cross_tab = pd.crosstab(df[col1], df[col2], margins=True)
            # Rename "All" with "Total" in the df_cross_tab columns and indices
            df_cross_tab.columns = df_cross_tab.columns.tolist()[:-1] + ["Total"]
            df_cross_tab.index = df_cross_tab.index.tolist()[:-1] + ["Total"]
            # Create percentages from cross-tab table
            values = df_cross_tab.values
            # Calculate percentages excluding the last row and last column
            percentages = values[:-1, :-1] / (values[:-1, :-1].sum()) * 100
            # Create column/row-wise margins of percentages 2D array
            row_margins = np.sum(percentages, axis=1)
            # Append 100 to row_margins
            row_margins = np.append(row_margins, 100)
            col_margins = np.sum(percentages, axis=0) 
            # Append col_margins to the end of percentages
            percentages = np.append(percentages, col_margins.reshape(1, percentages.shape[1]), axis=0)
            # Append the row_margins to the end of percentages
            percentages = np.append(percentages, row_margins.reshape(percentages.shape[0], 1), axis=1)
            annotations = np.array([f"{x}\n({round(y)}%)" for x, y in zip(values.flatten(), percentages.flatten())]).reshape(df_cross_tab.shape)
            # Rename the x-ticks and y-ticks with the options
            y_tick_labels = list([f"{k}, {v}" for k, v in options1.items()]) + ["Total"]
            x_tick_labels = list([f"{k}, {v}" for k, v in options2.items()]) + ["Total"]
            # Rotate the x-ticks and y-ticks
            # Plot the cross-tab table, which is a heatmap with no colors and black grid
            sns.heatmap(
                df_cross_tab, annot=annotations, fmt="", cmap="Blues", annot_kws={"size": 30}, linewidths=1,
                cbar=True, square=True, cbar_kws={"shrink": 0.5}, xticklabels=x_tick_labels, yticklabels=y_tick_labels, ax=ax)
            # # Add the options as x-axis labels and y-axis labels
            x_label = f"{label2}"
            y_label = f"{label1}"
            # Align the ticks to the center of the cell
            plot_type = "cross_tab"
        # Create box plot
        elif VarReader.is_dtype_categorical(dtype1) or VarReader.is_dtype_categorical(dtype2):  # One is categorical, the other is numerical
            # continue
            # If one is categorical, create a histogram with overlapping groups/categories
            if VarReader.is_dtype_categorical(dtype1):
                cat_col, num_col = col1, col2
                cat_options, num_options = options1, options2
                cat_options_str, num_options_str = options_str1, options_str2
                cat_dtype, num_dtype = dtype1, dtype2
                cat_label, num_label = label1, label2
            else:
                cat_col, num_col = col2, col1
                cat_options, num_options = options2, options1
                cat_options_str, num_options_str = options_str2, options_str1
                cat_dtype, num_dtype = dtype2, dtype1
                cat_label, num_label = label2, label1
            # The color of the boxplot is the category
            ax.set_title(f"{col1} ({section1})\nvs\n{col2} ({section2})", fontsize=24, loc="center")
            data = {}
            for cat in cat_options.keys():
                num_data = df[df[cat_col] == cat][num_col].values
                num_missing_count = np.count_nonzero(num_data == -1)
                data[cat] = [x for x in num_data if x != -1 and not np.isnan(x)]
            # print("cat_options", cat_options, data.keys())
            ax.boxplot(data.values(), labels=[f"{k}, {v} (N={len(data[k])})" for k, v in cat_options.items()])
            for i, (cat, data_vals) in enumerate(data.items()):
                x_pos = np.random.normal(i+1, 0.02, len(data_vals))
                ax.scatter(x_pos, data_vals, alpha=0.2)
            # Add gridlines
            ax.grid(which="major", axis="x", linestyle="-", linewidth=0.3, color="grey")
            ax.grid(which="major", axis="y", linestyle="-", linewidth=0.3, color="grey")
            if threshold is not None:
                # Make a horizontal line at the threshold
                ax.axhline(y=threshold, color="red", linestyle="--", linewidth=0.5)
            x_label = f"{cat_label}"
            y_label = f"{num_label}"
            plot_type = "boxplot"
        # Create scatter plot
        else:  # Both are numerical
            ax.set_title(f"{col1} ({section1})\nvs\n{col2} ({section2})", fontsize=24)
            # If both are numerical, create a scatter plot with the two columns
            # The x-axis is the first column, the y-axis is the second column
            data = zip(df[col1], df[col2])
            # Remove all instances of -1 and nan from the data
            data = [x for x in data if x[0] != -1 and not np.isnan(x[0]) and x[1] != -1 and not np.isnan(x[1])]
            x_data = [x[0] for x in data]
            y_data = [x[1] for x in data]
            # Plot the scatter plot
            ax.scatter(x_data, y_data, alpha=0.2)
            # Fit a linear regression line to the data and plot it
            slope, intercept = np.polyfit(x_data, y_data, 1)
            ax.plot(x_data, [slope * x + intercept for x in x_data], color="orange", linewidth=1, alpha=0.5)
            # Write the regression equation as opaque text on the top-left corner of the plot
            slope, intercept = round(slope, 2), round(intercept, 2)
            text = f"Linear Regression: y = {slope} * x + {intercept}"
            ax.text(0.05, 0.98, text, transform=ax.gca().transAxes, fontsize=20, va="top", alpha=0.7, color="orange")
            # Add another regression line that is robust to outliers using HuberRegressor
            sklearn_x_data = np.array(x_data).reshape(-1, 1)
            sklearn_y_data = np.array(y_data)
            huber_epsilon = 1.5
            model = HuberRegressor(epsilon=huber_epsilon)
            model.fit(sklearn_x_data, sklearn_y_data)
            # Plot the fitted line
            ax.plot(x_data, model.predict(sklearn_x_data), color="red", linewidth=1, alpha = 0.5, fontsize=20)
            # Write the regression equation as opaque text on the top-left corner of the plot
            text = f"Huber Regression (robust to outliers, ε = {huber_epsilon}): y = {round(model.coef_[0], 2)} * x + {round(model.intercept_, 2)}"
            ax.text(0.05, 0.93, text, transform=ax.gca().transAxes, fontsize=20, va="top", alpha=0.7, color="red")
            # Make grid background with gridlines
            ax.grid(which="major", axis="x", linestyle="-", linewidth=0.1, color="grey")
            ax.grid(which="major", axis="y", linestyle="-", linewidth=0.1, color="grey")
            # Add a caption for number of data points to the top-right of the plot
            ax.text(0.98, 0.98, f"N={len(data)}", horizontalalignment="right", verticalalignment="top", transform=ax.gca().transAxes)
            x_label = f"{label1}"
            y_label = f"{label2}"
            plot_type = "scatter"
        # Add labels to the axes
        x_label = x_label[:80] + "..." if len(x_label) > 80 else x_label
        y_label = y_label[:80] + "..." if len(y_label) > 80 else y_label
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=15)
        
        
    def plot_auc_curve(self, df, pred_col, truth_col, ax):
        ytest = df[truth_col]
        ypred_prob = df[pred_col]
        fpr, tpr, thresholds = roc_curve(ytest, ypred_prob)
        # Find the optimal point on the curve (defined as closest to top-left corner)
        optimal_idx = np.argmax(tpr - fpr)
        # Plot the ROC curve
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
        ax.grid(True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(f'Receiver Operating Characteristic Curve\n(AUC = {round(roc_auc, 4)})', fontsize=24)
        return ax
    
    def find_optimal_auc_threshold(self, df, pred_col, truth_col):
        ytest = df[truth_col]
        ypred_prob = df[pred_col]
        fpr, tpr, thresholds = roc_curve(ytest, ypred_prob)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
        
    def plot_precision_recall_curve(self, df, pred_col, truth_col, ax):
        ytest = df[truth_col]
        ypred_prob = df[pred_col]
        precision, recall, thresholds = precision_recall_curve(ytest, ypred_prob)
        average_precision = average_precision_score(ytest, ypred_prob)
        ax.plot(recall, precision, color='darkorange', lw=2)
        ax.set_xlabel('Recall', fontsize=20)
        ax.set_ylabel('Precision', fontsize=20)
        ax.grid(True)
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(f'Precision-Recall Curve\n(Avg Precision = {round(average_precision, 4)})', fontsize=24)
        return ax
    
    def evaluate_predictions(
        self, 
        df: pd.DataFrame, 
        pred_col_prob: str, 
        pred_col_class, 
        target_column: str, 
        title: str, 
        show_results=True, 
        threshold=None
        ):
        # Drop rows where the target_col is NaN
        df = df.copy()
        df[pred_col_prob] = pd.to_numeric(df[pred_col_prob], errors="coerce")
        df[pred_col_prob] = df[pred_col_prob].astype(float)
        df = df.dropna(axis=0, subset=[target_column])
        # Convert pred_col to float type
        # Coerce all values in pred_col_prob as float, replace errors with with NaN
        # Convert pred_col to binary 1 or 0 based on threshold, save the binary pred_col_binary, using .loc to suppress warnings
        # Filter the df by having non-nan values for the nomogram probability
        df_prediction = df[~df[pred_col_prob].isna()]
        is_multiclass = len(df_prediction[target_column].unique()) > 2
        # Assert there's no nan values in the nomogram probability or the metastasis columns
        assert df_prediction[pred_col_prob].isna().sum() == 0
        assert df_prediction[target_column].isna().sum() == 0
        if threshold is None:
            threshold = self.find_optimal_auc_threshold(df_prediction, pred_col_prob, target_column)
        df_prediction[pred_col_class] = df_prediction[pred_col_prob] > threshold
        # Calculate the accuracy, sensitivity, specificity, F1, and AUC metrics
        if is_multiclass:
            ytest = label_binarize(df_prediction[target_column], classes=df_prediction[target_column].unique())
            ypred = label_binarize(df_prediction[pred_col_class], classes=df_prediction[target_column].unique())
            ypred_prob = df_prediction[pred_col_prob]
            auc_score = roc_auc_score(ytest, ypred_prob)
            accuracy = accuracy_score(ytest, ypred)
            f1 = f1_score(ytest, ypred)
            precision = precision_score(ytest, ypred, zero_division=0)
            recall = recall_score(ytest, ypred, zero_division=0)
            specificity = recall_score(ytest, ypred, zero_division=0, pos_label=0)
        else:
            # Create 2 by 2 subplot
            ytest = df_prediction[target_column]
            ypred = df_prediction[pred_col_class]
            ypred_prob = df_prediction[pred_col_prob]
            auc_score = roc_auc_score(ytest, ypred_prob)
            fpr, tpr, thresholds = roc_curve(ytest, ypred_prob)
            accuracy = accuracy_score(ytest, ypred)
            precision, recall, thresholds = precision_recall_curve(ytest, ypred_prob)
            if show_results:
                fig, ax = plt.subplots(2, 2, figsize=(30, 30))
                self.plot_precision_recall_curve(df_prediction, pred_col_prob, target_column, ax=ax[0, 0])
                self.plot_auc_curve(df_prediction, pred_col_prob, target_column, ax=ax[0, 1])
            precision = precision_score(ytest, ypred, zero_division=0)
            f1 = f1_score(ytest, ypred)
            recall = recall_score(ytest, ypred)
            f1 = f1_score(ytest, ypred)
            specificity = recall_score(ytest, ypred)

        cls_report = classification_report(df_prediction[target_column], df_prediction[pred_col_class], zero_division=0)
        result = {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "auc": round(auc_score, 4),
            "fpr": fpr,
            "tpr": tpr,
            "classification_report": cls_report,
            "df_prediction": df_prediction,
            "pred_col_class": pred_col_class,
            "pred_col_prob": pred_col_prob,
            "truth_col": target_column
        }
        if show_results:
            self.plot_cross_tab(df_prediction, pred_col_class, target_column, ax[1, 1], plot_missing=False)
            self.plot_cross_tab(df_prediction, pred_col_prob, target_column, ax[1, 0], plot_missing=False, threshold=threshold)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.2)
            self.show_plot_and_save(title)
            print(f"Accuracy: {result['accuracy']}, ", f"F1: {result['f1']}, ", f"AUC: {result['auc']} ")
            print(f"Precision: {result['precision']}, ", f"Recall: {result['recall']}, ", f"Specificity: {result['specificity']}")
            print(f"{result['classification_report']}")
            

        return result

    def find_threshold(
        self, 
        df, 
        pred_col_prob, 
        pred_col_class, 
        truth_col, 
        metric="accuracy"
        ):
        best_metric = 0
        best_threshold = 0
        for i in range(0, 100, 1):
            threshold = i / 100
            # def evaluate_predictions(self, df: pd.DataFrame, pred_col_prob: str, pred_col_class, target_column: str, threshold: float, title: str, show_results=True):
            result = self.evaluate_predictions(df, pred_col_prob, pred_col_class, truth_col, threshold, title="", show_results=False)
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

    def evaluate_nomogram(self, processed_df, show_results):
        temp_df = processed_df.dropna(subset=["POS_metastasis", "PRE_sln_met_nomogram_prob"], axis=0)
        nomogram_eval = self.evaluate_predictions(
        temp_df,
        "PRE_sln_met_nomogram_prob",
        "PRE_sln_met_nomogram_class",
        target_column="POS_metastasis",
        title="PRE_sln_met_nomogram_prob",
        show_results=show_results
        )
        return nomogram_eval


    def evaluate_experiment(self, VarReader, target_column, experiment_name, results, inclusion_criteria, show_results=False):
        model_eval_results = {}
        result_df = results[experiment_name]["result_df"]
        #     display(result_df)
        # result_df = result_df[~result_df[get_nomogram_columns(target_column)].isna().any(axis=1)]
        # result_df = result_df[result_df["PRE_susp_LN_prsnt_composite"] > 0]
        # InclusionCriteria = config.InclusionCriteria
        eligibility_dict = inclusion_criteria.get_eligibility_dict(standardized=True)
        if eligibility_dict is not None:
            result_df = result_df[result_df["PRE_record_id"].map(eligibility_dict)]
            print(f"Inclusion criteria applied. {len(result_df)} records remain eligible.")
        
        for pred_col_prob, pred_col_class, model_name in zip(results[experiment_name]["prob_columns"], results[experiment_name]["class_columns"], results[experiment_name]["model_names"]):
            show_results = show_results and model_name in config.models_to_show
            if show_results:
                print("-"*50, model_name, "-"*50)
            else:
                print("will not show results for", model_name)
            VarReader.add_var(pred_col_prob, section="ML", dtype="numeric", label=f"Predicted Probability", options=dict(VarReader.read_var_attrib(target_column, has_missing=False)["options"]))
            VarReader.add_var(pred_col_class, section="ML", dtype="categorical", label=f"Predicted Class", options=dict(VarReader.read_var_attrib(target_column, has_missing=False)["options"]))
            eval_result = self.evaluate_predictions(
                result_df,
                pred_col_prob,
                pred_col_class,
                target_column=target_column,
                title=f"{pred_col_prob} {experiment_name} (N={len(result_df)})" ,
                show_results=show_results
            ) 
            model_eval_results[model_name] = eval_result
        

        eval_num_records = len(result_df[result_df[target_column].notna()])
        return model_eval_results, eval_num_records

    
    def plot_calibration_curve(
        self, 
        results, 
        target_df, 
        valid_idx, 
        target_column, 
        pred_column, 
        experiment_name, 
        model_name, 
        show_plot=True, 
        verbose=False
        ):

        target_df = target_df.loc[valid_idx]
        target_true = target_df[target_column]
        # ml_model_pred = results[experiment_name]["result_df"]
        # ml_model_pred = ml_model_pred.loc[valid_idx]["PRE_POS_metastasis_elastic_net_prob"]
        pred_probs = results[experiment_name]["result_df"].loc[valid_idx][pred_column]
        print(f"The shape of the target_df is {target_df.shape}", f"The shape of the prediction is {pred_probs.shape}")

        # for pred_probs, color, label in zip([ml_model_pred, nomogram_pred], ["orange", "blue"], [f"elastic_net {experiment_name}", "Nomogram"]):
        
        # Calculate the RMSE between the predicted probabilities and actual label
        rmse = round(np.sqrt(mean_squared_error(target_true, pred_probs)), 3)
        # Calculate the log-loss
        loss = round(log_loss(target_true, pred_probs), 3)
        target_true_binned, pred_probs_binned = calibration_curve(target_true, pred_probs, n_bins=10)
        plt.plot(pred_probs_binned, target_true_binned, label=f"{model_name} ({rmse} RMSE, {loss} Log-Loss)", alpha=0.5, marker="o", linestyle="None")
        fit_line = np.polyfit(pred_probs_binned, target_true_binned, 1)
        plt.plot(pred_probs_binned, fit_line[0] * pred_probs_binned + fit_line[1], linestyle="-", alpha=0.7)

        pred_auc = round(roc_auc_score(target_true, pred_probs), 4)
        nomogram_accuracy = round(accuracy_score(target_true, pred_probs > 0.5), 4)

        if verbose:
            print(f"The AUC for {model_name} is {pred_auc}")
            print(f"The accuracy for {model_name} is {nomogram_accuracy}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title(f"{experiment_name}")
        plt.legend(loc="lower right", prop={"size": 24})
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        if show_plot:
            # plt.show()
            self.show_plot_and_save(f"{experiment_name}_{model_name}_calibration_curve")
        
        pred_probs.hist(bins=100, label=f"{model_name}", alpha=0.5)
        plt.legend()
        if show_plot:
            # plt.show()
            self.show_plot_and_save(f"{experiment_name}_{model_name}_histogram")
            # plt.show()
        
        # return {
        #     "elastic_net_auc": elastic_net_auc,
             
        # }
    
    
    def get_significance_level(self, p_value):
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
        
        
    
    def get_feature_stat_significance(
        self, 
        df_ready, 
        subset_cols, 
        subset_cols_name, 
        target_column, 
        target_type="categorical"
        ):
        temp_df = df_ready.copy()
        print("-"*20, f"stat. significance between {target_column} and", subset_cols_name, "-"*20)
        col2pval = {}
        for column in subset_cols:
            if column not in temp_df.columns:
                continue
            try:
                if column != target_column:
                    # Calculate the p-value while ignoring NaNs in both columns
                    temp_df_column = temp_df[[column, target_column]].dropna()
                    num_unique = len(temp_df_column[column].unique())
                    if num_unique > 5:
                        col_type = "continuous"
                    elif num_unique <= 1:
                        continue
                    else:  # i.e., 2 <= num_unique <= 5
                        col_type = "categorical"
                    if target_type == "categorical" and col_type == "categorical":
                        # Use the chi-squared test
                        test_name = "Chi^2"
                        chi2, p, dof, expected = chi2_contingency(pd.crosstab(temp_df_column[column], temp_df_column[target_column]))
                    elif target_type == "categorical" and col_type == "continuous":
                        # Use the ANOVA test
                        test_name = "ANOVA/t-test"
                        f_stat, p = f_oneway(*[group[column].dropna().values for name, group in temp_df_column.groupby(target_column)])
                    elif target_type == "continuous" and col_type == "categorical":
                        pass
                    else:
                        pass
                    col2pval[column] = p
                        
            except Exception as e:
                # print(f"Error with column {column}: {e}")    
                raise e
        
        if len(col2pval) > 0:
            # Delete entries with NaN p-values
            col2pval = {k: v for k, v in col2pval.items() if not np.isnan(v)}
            # col2pval = {(str(k[0]), str(k[1]), str(k[2])): v for k, v in col2pval.items() if not np.isnan(v)}
            # Sort and print the p-values in ascending order
            sorted_pvals = sorted(col2pval.items(), key=lambda x: x[1])
            # max_col1 = max([len(x[0][0]) for x in sorted_pvals])
            # max_col2 = max([len(x[0][1]) for x in sorted_pvals])
            # max_test_name = max([len(x[0][2]) for x in sorted_pvals]
            max_len = max([len(x[0]) for x in sorted_pvals])
            for col1, p in sorted_pvals:
                significance_level = self.get_significance_level(p)
                print(f"{col1: <{max_len}}: {p:.8f} {significance_level}")
                        
            return col2pval
        else:
            return None
    
    
    def show_plot_and_save(self, plot_name, legend=None):
        " Show the plot and save it to the output directory "
        results_dir = config.results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # Always set the DPI to max
        # get the subplots that have been made
        # print("!!!"*100)
        # print(plt.get_fignums())
        # if len(plt.get_fignums()) > 1:
        #     for fig_num in plt.get_fignums():
        #         plt.figure(fig_num)
        #         plt.show()
        #         if legend is None:
        #             plt.savefig(os.path.join(results_dir, f"{plot_name}_{fig_num}.png"), dpi=1200)
        #         else:
        #             plt.savefig(os.path.join(results_dir, f"{plot_name}_{fig_num}.png"), dpi=1200, bbox_extra_artists=(legend,), bbox_inches='tight')
        #         print(f"Saved {plot_name}_{fig_num}.png")
        # else:
        if legend is None:
            plt.savefig(os.path.join(results_dir, f"{plot_name}.png"), dpi=500)
        else:
            plt.savefig(os.path.join(results_dir, f"{plot_name}.png"), dpi=500, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.show()
        print(f"Saved {plot_name}.png")
        
        
        # def plot_experiment_groups_aucs(self, group_to_aucs, target_column):
        #     """Given a dictionary with schema: {group_name: [aucs], ...}, plot the mean and SE of AUC for each group

        #     Args:
        #         group_to_mean_auc (_type_): _description_
        #         target_column (_type_): _description_
        #     """
        #     num_groups = len(group_to_aucs[experiment_name].keys())
        #     model2idx = {model_name: i for i, model_name in enumerate(group_to_aucs[experiment_name].keys())}
        #     idx2model = {i: model_name for i, model_name in enumerate(group_to_aucs[experiment_name].keys())}
        #     for col_i, (target_column, experiment_name, subset_cols, _) in enumerate(Experiment):
        #         auc_mean, auc_se = [], []
        #         for model_name, aucs in model_to_aucs[experiment_name].items():
        #             auc_mean.append(np.mean(aucs))
        #             auc_se.append(np.std(aucs) / np.sqrt(num_repeat))
        #             group_to_aucs[experiment_name][model_name] = np.mean(aucs)

        #     x = np.arange(num_groups)  # the label locations
        #     x_to_aucs = [[] for _ in range(num_groups)]
        #     x_to_auc_ses = [[] for _ in range(num_groups)]
        #     for experiment_name in group_to_aucs.keys():
        #         if target_column in experiment_name:
        #             for model_name, auc in group_to_aucs[experiment_name].items():
        #                 x_to_aucs[model2idx[model_name]].append((auc, experiment_name))


        #     fig, ax = plt.subplots(figsize=(num_groups*1.5, num_groups*1.5))
        #     width = 1/(len(x_to_aucs[0])*1.2)
        #     experiment2color = {experiment_name: f"C{i}" for i, experiment_name in enumerate(group_to_aucs.keys()) if target_column in experiment_name}
        #     for i, aucs_and_experiments in enumerate(x_to_aucs):
        #         for j, (auc, experiment_name) in enumerate(aucs_and_experiments):
        #             x_loc = x[i] + width*j
        #             plt.bar(x_loc, auc, width, label=experiment_name, color=experiment2color[experiment_name], alpha=0.8)
        #             plt.text(x_loc, auc, f"{auc:.2f}", color="black", fontweight="bold", fontsize=6, ha="center", va="bottom")

        #     plt.xticks(x+width*(len(aucs_and_experiments)-1)/2, idx2model.values(), rotation=45)
        #     plt.ylabel("AUC (SE)")
        #     plt.title(f"{target_column}")
        #     # Write experiment names to legend
        #     handles, labels = ax.get_legend_handles_labels()
        #     by_label = dict(zip(labels, handles))
        #     plt.legend(by_label.values(), by_label.keys(), loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, fontsize=24)
        #     plt.grid(axis="y", alpha=0.5)
        #     plt.show()