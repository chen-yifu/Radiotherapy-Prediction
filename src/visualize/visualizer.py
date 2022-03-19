import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
from sklearn.linear_model import LinearRegression

# Update Mar 18: Removed outlier 


os.chdir("/Users/yifuchen/Work/Repositories/Radiotherapy-Prediction/") # Change current directory path to root
df = pd.read_csv("./data/AllTranTrainVal.csv")
df_missing = df.isnull().sum(axis=0)/600
df_missing.to_csv("./data/AllTranTrainVal_sparsity.csv")

col_types = json.load(open("./data/metadata/col_types.json"))

def plot_col(col_name, other_col="did_the_patient_receive_pm"):
    try:
        print("-"*100)
        col_type = col_types[col_name]
        print(f"Plotting column: {col_name} of type {col_type}...\n")
        pos_df = df.loc[df[other_col] == 1, col_name]
        neg_df = df.loc[df[other_col] == 0, col_name]
        if col_type == "time":
            pos_df = pos_df.apply(lambda x: pd.to_datetime(x).value)
            neg_df = neg_df.apply(lambda x: pd.to_datetime(x).value)
            
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
        fig.suptitle("{} ({})".format(col_name, col_type))
        if col_type == "enum": # Plot overlapping histograms
            # bins = np.linspace(min(min(pos_df), min(neg_df)), max(max(pos_df), max(neg_df)), 20)
            h,e = np.histogram([neg_df, pos_df])
            ax1.bar(h, label=['RTx=0', 'RTx=1'], color=["blue", "orange"])
            ax1.legend(loc='best')
            contingency_table = pd.crosstab(df[other_col], df[col_name], margins=False, dropna=False)
            percent = pd.crosstab(df[other_col], df[col_name], margins=False, normalize='all').values

            str_percent = []
            for i, row in enumerate(percent):
                temp = []
                for j, v in enumerate(row):
                    v = round(v * 100, 1)
                    temp.append(f"{contingency_table.iloc[i, j]} ({str(v)}%)")

                str_percent.append(temp)
            str_percent = np.asarray(str_percent).reshape(contingency_table.shape)
            sns.heatmap(contingency_table, annot=str_percent, fmt="", ax=ax2, cmap='crest')
            plt.show()
        else:

            # ax1.scatter(neg_df, [0] * len(neg_df), label='RTx=0', c="blue", alpha=0.1)
            # ax1.scatter(pos_df, [1] * len(pos_df), label='RTx=1', c="orange", alpha=0.1)
            # ax1.legend(loc='best')
            
            # FIXME temporary overwriting the outlier
            df_temp = df.copy()
            df_temp = df_temp[df_temp.apply(lambda x: x["img_size"] <= 400, axis=1)]
            from numpy import cov
            covariance = cov(list(df_temp[col_name]), list(df_temp[other_col]))
            print("covariance", covariance)
            sns.regplot(x=col_name, y=other_col, data=df_temp, ax=ax1, color="blue", scatter_kws={"alpha":0.1})
            sns.distplot(neg_df,  kde=False, label='RTx=0')
            sns.distplot(pos_df,  kde=False,label='RTx=1')
            fig.show()
            df_temp = df_temp[[col_name, other_col]]
            df_temp.dropna(axis = 0, how = 'any', inplace = True)
            reg = LinearRegression().fit(df_temp[col_name].values.reshape(-1,1), df_temp[other_col].values.reshape(-1,1))
            # print("Number of nan", df_temp[col_name].isnull().sum())
            # print("Number of nan 2", df_temp[other_col].isnull().sum())
            r = np.corrcoef(df_temp[col_name].values, df_temp[other_col].values)
            print("Correlation", r)
            coefs = reg.coef_
            print(f"Intercept {reg.intercept_}; Slope {coefs}")
    except Exception as e:
        raise e
    
    
import sys
from itertools import product
if __name__ == "__main__":
    if "-vizall" in sys.argv:
        for col in list(df.columns)[2:]:
            plot_col(col)
    else:
        while True:
            col_name = input("Type column names to visualize...\n")
        #     print("Received: {}".format(col_name))
            if "," in col_name:
                col_names = col_name.split(",")
                print(f"Plotting multiple columns of len {len(col_names)}... {col_names}")
                plot_col(col_names[0], col_names[1])
        #         for col1, col2 in list(product([1,2], repeat=2)):
        #             if col1 == col2: continue
        #             sns.scatterplot(data = df, x = col1, y = col2)
        #     else:
            # if col_name in df.columns:
            #     plot_col(col_name)
            # else:
            #     print("Try again.")
        