# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import config
# from sklearn.linear_model import HuberRegressor

# class Plotter:
    
#     def __init__(self) -> None:
#         self.VarReader = config.VarReader
#         pass
    

#     def make_plots(self, df, cross_tab_cols_set1, cross_tab_cols_set2, show_plots, plot_missing=False):
#         VarReader = self.VarReader
#         for col1 in cross_tab_cols_set1:
#             print("-"*50)
#             var1_dict = VarReader.read_var_attrib(col1, has_missing=VarReader.has_missing(df, col1))
#             section1, dtype1, label1, options1, options_str1 = var1_dict["section"], var1_dict["dtype"], var1_dict["label"], var1_dict["options"], var1_dict["options_str"]
#             for col2 in cross_tab_cols_set2:
#                 var2_dict = VarReader.read_var_attrib(col2, has_missing=VarReader.has_missing(df, col2))
#                 section2, dtype2, label2, options2, options_str2 = var2_dict["section"], var2_dict["dtype"], var2_dict["label"], var2_dict["options"], var2_dict["options_str"]
#                 print(f"{col1} ({dtype1}, {section1}) vs {col2} ({dtype2}, {section2})")
#                 plt.title(f"{col1} ({section1})\nvs\n{col2} ({section2})", fontsize=12, loc="center")
#                 # Create cross-tab table
#                 if VarReader.is_dtype_categorical(dtype1) and VarReader.is_dtype_categorical(dtype2):
#                     # continue
#                     # If both are categorical, create a contingency table with percentages and margins
#                     df_cross_tab = pd.crosstab(df[col1], df[col2], margins=True)
#                     # Rename "All" with "Total" in the df_cross_tab columns and indices
#                     df_cross_tab.columns = df_cross_tab.columns.tolist()[:-1] + ["Total"]
#                     df_cross_tab.index = df_cross_tab.index.tolist()[:-1] + ["Total"]
#                     # Create percentages from cross-tab table
#                     values = df_cross_tab.values
#                     # Calculate percentages excluding the last row and last column
#                     percentages = values[:-1, :-1] / (values[:-1, :-1].sum()) * 100
#                     # Create column/row-wise margins of percentages 2D array
#                     row_margins = np.sum(percentages, axis=1)
#                     # Append 100 to row_margins
#                     row_margins = np.append(row_margins, 100)
#                     col_margins = np.sum(percentages, axis=0) 
#                     # Append col_margins to the end of percentages
#                     percentages = np.append(percentages, col_margins.reshape(1, percentages.shape[1]), axis=0)
#                     # Append the row_margins to the end of percentages
#                     percentages = np.append(percentages, row_margins.reshape(percentages.shape[0], 1), axis=1)
#                     annotations = np.array([f"{x}\n({round(y)}%)" for x, y in zip(values.flatten(), percentages.flatten())]).reshape(df_cross_tab.shape)
#                     # Rename the x-ticks and y-ticks with the options
#                     y_tick_labels = list([f"{k}, {v}" for k, v in options1.items()]) + ["Total"]
#                     x_tick_labels = list([f"{k}, {v}" for k, v in options2.items()]) + ["Total"]
#                     # Rotate the x-ticks and y-ticks
#                     # Plot the cross-tab table, which is a heatmap with no colors and black grid
#                     sns.heatmap(
#                         df_cross_tab, annot=annotations, fmt="", cmap="Blues", annot_kws={"size": 10}, linewidths=1,
#                         cbar=True, square=True, cbar_kws={"shrink": 0.5}, xticklabels=x_tick_labels, yticklabels=y_tick_labels)
#                     # # Add the options as x-axis labels and y-axis labels
#                     x_label = f"{label2}"
#                     y_label = f"{label1}"
#                     plt.xticks(rotation=0)
#                     plt.yticks(rotation=0)
#                     # Align the ticks to the center of the cell
#                     plot_type = "cross_tab"
#                 # Create box plot
#                 elif VarReader.is_dtype_categorical(dtype1) or VarReader.is_dtype_categorical(dtype2):  # One is categorical, the other is numerical
#                     # continue
#                     # If one is categorical, create a histogram with overlapping groups/categories
#                     if VarReader.is_dtype_categorical(dtype1):
#                         cat_col, num_col = col1, col2
#                         cat_options, num_options = options1, options2
#                         cat_options_str, num_options_str = options_str1, options_str2
#                         cat_dtype, num_dtype = dtype1, dtype2
#                         cat_label, num_label = label1, label2
#                     else:
#                         cat_col, num_col = col2, col1
#                         cat_options, num_options = options2, options1
#                         cat_options_str, num_options_str = options_str2, options_str1
#                         cat_dtype, num_dtype = dtype2, dtype1
#                         cat_label, num_label = label2, label1
#                     # cat_col, num_col = (col1, col2) if VarReader.is_dtype_categorical(dtype1) else (col2, col1)
#                     # var_cat_dict = VarReader.read_var_attrib(cat_col, has_missing=VarReader.has_missing(df, cat_col))
#                     # section_cat, cat_dtype, cat_label, cat_options, cat_options_str = var_cat_dict["section"], var_cat_dict["dtype"], var_cat_dict["label"], var_cat_dict["options"], var_cat_dict["options_str"]
#                     # var_num_dict = VarReader.read_var_attrib(num_col, has_missing=VarReader.has_missing(df, num_col))
#                     # section_num, num_dtype, num_label, num_options, num_options_str = var_num_dict["section"], var_num_dict["dtype"], var_num_dict["label"], var_num_dict["options"], var_num_dict["options_str"]
#                     # The color of the boxplot is the category
#                     plt.title(f"{col1} ({section1})\nvs\n{col2} ({section2})", fontsize=12, loc="center")
#                     data = {}
#                     for cat in cat_options.keys():
#                         num_data = df[df[cat_col] == cat][num_col].values
#                         num_missing_count = np.count_nonzero(num_data == -1)
#                         data[cat] = [x for x in num_data if x != -1 and not np.isnan(x)]
#                     # print("cat_options", cat_options, data.keys())
#                     plt.boxplot(data.values(), labels=[f"{k}, {v} (N={len(data[k])})" for k, v in cat_options.items()])
#                     for i, (cat, data_vals) in enumerate(data.items()):
#                         x_pos = np.random.normal(i+1, 0, len(data_vals))
#                         plt.scatter(x_pos, data_vals, alpha=0.1)
#                     # Add gridlines
#                     plt.grid(which="major", axis="x", linestyle="-", linewidth=0.3, color="grey")
#                     plt.grid(which="major", axis="y", linestyle="-", linewidth=0.3, color="grey")
#                     x_label = f"{cat_label}"
#                     y_label = f"{num_label}"
#                     plot_type = "boxplot"
#                 # Create scatter plot
#                 else:  # Both are numerical
#                     # If both are numerical, create a scatter plot with the two columns
#                     # The x-axis is the first column, the y-axis is the second column
#                     data = zip(df[col1], df[col2])
#                     # Remove all instances of -1 and nan from the data
#                     data = [x for x in data if x[0] != -1 and not np.isnan(x[0]) and x[1] != -1 and not np.isnan(x[1])]
#                     x_data = [x[0] for x in data]
#                     y_data = [x[1] for x in data]
#                     # Plot the scatter plot
#                     plt.scatter(x_data, y_data, alpha=0.2)
#                     # Fit a linear regression line to the data and plot it
#                     slope, intercept = np.polyfit(x_data, y_data, 1)
#                     plt.plot(x_data, [slope * x + intercept for x in x_data], color="orange", linewidth=1, alpha=0.5)
#                     # Write the regression equation as opaque text on the top-left corner of the plot
#                     slope, intercept = round(slope, 2), round(intercept, 2)
#                     text = f"Linear Regression: y = {slope} * x + {intercept}"
#                     plt.text(0.05, 0.98, text, transform=plt.gca().transAxes, fontsize=10, va="top", alpha=0.7, color="orange")
#                     # Add another regression line that is robust to outliers using HuberRegressor
#                     sklearn_x_data = np.array(x_data).reshape(-1, 1)
#                     sklearn_y_data = np.array(y_data)
#                     huber_epsilon = 1.5
#                     model = HuberRegressor(epsilon=huber_epsilon)
#                     model.fit(sklearn_x_data, sklearn_y_data)
#                     # Plot the fitted line
#                     plt.plot(x_data, model.predict(sklearn_x_data), color="red", linewidth=1, alpha = 0.5)
#                     # Write the regression equation as opaque text on the top-left corner of the plot
#                     text = f"Huber Regression (robust to outliers, ε = {huber_epsilon}): y = {round(model.coef_[0], 2)} * x + {round(model.intercept_, 2)}"
#                     plt.text(0.05, 0.93, text, transform=plt.gca().transAxes, fontsize=10, va="top", alpha=0.7, color="red")
#                     # Make grid background with gridlines
#                     plt.grid(which="major", axis="x", linestyle="-", linewidth=0.1, color="grey")
#                     plt.grid(which="major", axis="y", linestyle="-", linewidth=0.1, color="grey")
#                     # Add a caption for number of data points to the top-right of the plot
#                     plt.text(0.98, 0.98, f"N={len(data)}", horizontalalignment="right", verticalalignment="top", transform=plt.gca().transAxes)
#                     x_label = f"{label1}"
#                     y_label = f"{label2}"
#                     plot_type = "scatter"
#                 # Add labels to the axes
#                 x_label = x_label[:80] + "..." if len(x_label) > 80 else x_label
#                 y_label = y_label[:80] + "..." if len(y_label) > 80 else y_label
#                 plt.xlabel(x_label)
#                 plt.ylabel(y_label)
#                 if show_plots:
#                     plt.show()
