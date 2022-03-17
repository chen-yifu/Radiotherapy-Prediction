import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
os.chdir("/Users/yifuchen/Work/Repositories/Radiotherapy-Prediction/") # Change current directory path to root
df = pd.read_csv("./data/AllTranTrainVal.csv")
df_missing = df.isnull().sum(axis=0)/600
df_missing.to_csv("./data/AllTranTrainVal_sparsity.csv")

col_types = json.load(open("./data/metadata/col_types.json"))

def plot_col(col_name):
    try:
        print("-"*100)
        col_type = col_types[col_name]
        print(f"Plotting column: {col_name} of type {col_type}...\n")
        pos_df = df.loc[df["did_the_patient_receive_pm"] == 1, col_name]
        neg_df = df.loc[df["did_the_patient_receive_pm"] == 0, col_name]
        if col_type == "time":
            pos_df = pos_df.apply(lambda x: pd.to_datetime(x).value)
            neg_df = neg_df.apply(lambda x: pd.to_datetime(x).value)
            
        plt.title(f"{col_name} ({col_type})" )

        if col_type == "enum": # Plot overlapping histograms
            bins = np.linspace(min(min(pos_df), min(neg_df)), max(max(pos_df), max(neg_df)), 20)
            plt.hist([neg_df, pos_df], bins, label=['RTx=0', 'RTx=1'], color=["blue", "orange"])
            plt.legend(loc='best')
            plt.show()
            ax = plt.axes()
            contingency_table = pd.crosstab(df["did_the_patient_receive_pm"], df[col_name], margins=False, dropna=False)
            percent = pd.crosstab(df["did_the_patient_receive_pm"], df[col_name], margins=False, normalize='all').values

            str_percent = []
            for i, row in enumerate(percent):
                temp = []
                for j, v in enumerate(row):
                    v = round(v * 100, 1)
                    temp.append(f"{contingency_table.iloc[i, j]} ({str(v)}%)")

                str_percent.append(temp)
            str_percent = np.asarray(str_percent).reshape(contingency_table.shape)
            sns.heatmap(contingency_table, annot=str_percent, fmt="", ax=ax, cmap='crest')
            plt.show()
        else:
            f1 = plt.figure(1)
            plt.scatter(neg_df, [0] * len(neg_df), label='RTx=0', c="blue", alpha=0.1)
            plt.scatter(pos_df, [1] * len(pos_df), label='RTx=1', c="orange", alpha=0.1)
            plt.legend(loc='best')
            f1.show()
            f2 = plt.figure(2)
            sns.distplot(neg_df,  kde=False, label='RTx=0')
            sns.distplot(pos_df,  kde=False,label='RTx=1')
            f2.show()
    except Exception as e:
        raise e
    plt.show()
    
    
if __name__ == "__main__":
    while True:
        col_name = input("Type a column name to visualize...\n")
        if col_name in df.columns:
            plot_col(col_name)
        else:
            print("Try again.")
    