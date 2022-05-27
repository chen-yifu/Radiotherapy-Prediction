"""
A script to visualize the differences between the original Excel spreadsheet,
and another spreadsheet that contains differences.
"""
# Import libraries for reading data and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xlrd import open_workbook
from styleframe import StyleFrame, utils
import re

# Set printing options with no limits
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

original_path = "data/manual_review/RadiationAndANN_DATA_2021-11-15_0835.csv"
review_path = "data/manual_review/50_Cases_Dana_Reviewed.xlsx"
output_path = "data/manual_review/50_Cases_Dana_Reviewed_Differences.xlsx"


# Helper for checking if the cell background is yellow-highlighted
def cell_has_yellow_bgd(cell):
    hlt = cell.style.bg_color in {utils.colors.yellow, 'FFFFFF00'}
    return hlt


# Check if the cell background is red
def cell_has_red_color(cell):
    hlt = cell.style.font_color in {utils.colors.red, 'FFFF0000'}
    return hlt


if __name__ == "__main__":
    # Read in the data
    original_df = pd.read_csv(original_path)
    reviewed_df = pd.read_excel(review_path)
    # Keep the subsets of rows that have been highlighted in reviewed_df
    sf = StyleFrame.read_excel(
        review_path, read_style=True, use_openpyxl_styles=False
    )
    yellow_df = StyleFrame(
        sf.applymap(cell_has_yellow_bgd).dropna(how="all")
    )
    # Filter sf_df rows by having "True" values in record_id
    hlt_record_ids = []
    for record_id, is_hlt in zip(reviewed_df["record_id"], yellow_df["record_id"]):
        if is_hlt:
            hlt_record_ids.append(record_id)
    # Filter original_df to only include record_ids that have been highlighted
    original_df = original_df[original_df["record_id"].isin(hlt_record_ids)]
    # Keep the subsets of rows that have the same "record_id"
    original_df = original_df.loc[
        original_df["record_id"].isin(reviewed_df["record_id"])
        ]
    reviewed_df = reviewed_df.loc[
        reviewed_df["record_id"].isin(original_df["record_id"])
    ]
    print(f"There are {len(reviewed_df)} rows highlighted in the review.")

    # Keep just the subset of columns that overlap in both dataframes
    original_df = original_df[
        original_df.columns.intersection(reviewed_df.columns)
    ]
    reviewed_df = reviewed_df[
        reviewed_df.columns.intersection(original_df.columns)
    ]

    for name, temp_df in zip(
        ["original_df", "another_df"], [original_df, reviewed_df]
    ):
        # Check if there are duplicate record_ids in this dataframe
        if len(temp_df["record_id"].unique()) != len(temp_df):
            duplicate_ids = temp_df["record_id"].duplicated()
            num_dup = sum(duplicate_ids)
            print(f"Number of duplicate record_ids in {name}: {num_dup}")
            # Remove the duplicates in-place
            temp_df.drop_duplicates(subset="record_id", inplace=True)
        else:
            print(f"There are no duplicate record_ids in {name}")

    # Assert that the two dataframes have the same number of rows
    print(f"Original dataframe has {len(original_df)} rows.")
    print(f"Another dataframe has {len(reviewed_df)} rows.")
    assert original_df.shape[0] == reviewed_df.shape[0]
    # Assert that the two dataframes have the same columns
    print(f"Original dataframe has {len(original_df.columns)} columns.")
    print(f"Another dataframe has {len(reviewed_df.columns)} columns.")
    assert original_df.columns.tolist() == reviewed_df.columns.tolist()

    # For each column, get the difference between original and reviewed data
    # Save the differences to a new dataframe
    diff_df = pd.DataFrame(columns=original_df.columns)
    print("-"*80)
    print(f"Comparing {original_df.shape[0]} rows in the DataFrames")
    tot_num_diff = 0
    tot_num_miss = 0
    diff_dict = {}
    for col in original_df.columns:
        if col == "record_id":
            diff_df[col] = reviewed_df[col]
            continue
        diff_strs = []
        diff_record_ids = []
        # Use a composite string, formatted as
        # "original_value [reviewed_value]" in the position
        # where the values differ
        for i in range(original_df.shape[0]):
            record_id = original_df.iloc[i]["record_id"]
            # Get the rows by record_id
            original_row = original_df.loc[
                original_df["record_id"] == record_id
            ]
            reviewed_row = reviewed_df.loc[
                reviewed_df["record_id"] == record_id
            ]
            # If both values are nan, continue
            if original_row[col].item() == np.nan and \
                    reviewed_row[col].item() == np.nan or \
                    (str(original_row[col].item()) == "nan" and
                        str(reviewed_row[col].item()) == "nan"):
                diff_strs.append("")
                continue
            # If column type is date, compare them as dates
            # If the column type is numeric, get the difference
            elif (original_df[col].dtype == np.dtype("int64") or
                    original_df[col].dtype == np.dtype("float64")) \
                and (reviewed_df[col].dtype == np.dtype("int64") or
                     reviewed_df[col].dtype == np.dtype("float64")):
                diff = original_row[col].values[0] - \
                       reviewed_row[col].values[0]
                if diff != 0:
                    diff_strs.append(
                        f"{original_df[col].iloc[i]} "
                        f"[{reviewed_df[col].iloc[i]}]"
                    )
                    diff_record_ids.append(record_id)
                else:
                    diff_strs.append("")
            else:
                # If the column type is not numeric
                if original_row[col].values[0] != reviewed_row[col].values[0]:
                    diff_strs.append(
                        f"{original_df.iloc[i][col]} "
                        f"[{reviewed_df.iloc[i][col]}]"
                    )
                    diff_record_ids.append(record_id)
                else:
                    diff_strs.append("")
        # If the original value is substring of square bracket,
        # treat it as the same value
        diff_strs_filtered = []
        for diff_str in diff_strs:
            if not diff_str:
                diff_strs_filtered.append("")
                continue
            sq_br_val = re.search(r"\[(.*)\]", diff_str)
            sq_br_val = sq_br_val.group(1)
            # Get the value from beginning until the first square bracket
            orig_val = diff_str.split("[")[0].strip()
            # print(f"orig_val: {orig_val}, sq_br_val: {sq_br_val}, diff_str: {diff_str}")
            if str(sq_br_val) != "nan" and len(sq_br_val) \
                    and str(orig_val) != "nan" and len(orig_val) \
                    and (sq_br_val in diff_str or diff_str in sq_br_val):
                diff_str = ""
            diff_strs_filtered.append(diff_str)
        assert len(diff_strs_filtered) == len(diff_strs)
        diff_df[col] = diff_strs_filtered
        # Print the number of different entries for this column
        num_diff = 0
        for s in diff_strs_filtered:
            if s != "":
                num_diff += 1
                tot_num_diff += 1
            if str(s).startswith("nan"):
                tot_num_miss += 1
        print(f"Number of different entries for {col}: {num_diff}")
        diff_dict[col] = num_diff

    diff_df.to_excel(output_path)
    print("Saved differences to {}".format(output_path))
    print("The total number of entries in DataFrame is:"
          f"{len(diff_df) * (len(diff_df.columns)-1)}")
    print(f"There are {tot_num_diff} different entries in the DataFrames")
    print(f"There are {tot_num_miss} missing entries in the DataFrames")
    # Sort diff_dict by number of missingness, and print out the top 10
    sorted_diff_dict = sorted(
        diff_dict.items(),
        key=lambda x: x[1],
        reverse=True
        )
    print("Top 10 missingness:")
    for i in range(min(10, len(sorted_diff_dict))):
        print(f"{sorted_diff_dict[i][0]}: {sorted_diff_dict[i][1]}")

    # # Plot the differences
    # sns.set(style="whitegrid")
    # sns.pairplot(diff_df, diag_kind="kde")
    # plt.show()
    # # Save the differences to a new CSV file

