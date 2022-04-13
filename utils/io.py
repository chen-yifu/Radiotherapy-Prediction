# IO Helpers for loading, saving, and printing

from utils.get_timestamp import *
from utils.printers import *
import os
# import pickle
import dill as pickle
import pandas as pd

out_dir = f"data/preprocessed"

### LOADER FUNCTIONS ###

import json 

def load_col_type(path):
    with open(path, "r") as f:
        col_type = json.load(f)
    return col_type


### SAVER FUNCTIONS ### 
def initialize_experiment_folder():
    # Make a folder for saving the experiment data
    global cur_timestamp
    cur_timestamp = get_timestamp()
    experiment_path = os.path.join(out_dir, cur_timestamp)
    os.mkdir(experiment_path)
    my_print("Created experiment folder:", experiment_path)
    return experiment_path
    
    
def save_experiment_df(df: pd.DataFrame, file_name: str, description: str):
    # Save the dataframe to experiment folder
    file_path = os.path.join(out_dir, cur_timestamp, file_name)
    df.to_csv(file_path, index=False)
    my_print(f"Saved {description} DataFrame to: {file_path}.")
    return file_path
    
def save_experiment_pickle(object, file_name: str, description: str):
    file_path = os.path.join(out_dir, cur_timestamp, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    my_print(f"Saved {description} Object to: {file_path}.")
    return file_path

def add_to_log(content):
    log_path = os.path.join(out_dir, cur_timestamp, "log.txt")
    if not os.path.exists(log_path):
        content = f"Use the following command in terminal to view this log file:\ncat {log_path}\n" \
            + content
    with open(log_path, 'a') as f:
        f.write(content + '\n')



### PRINTER FUNCTIONS BELOW ### 
# Helper to print in terminal with colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def my_print(*args, add_sep=False, color=bcolors.WARNING, plain=False):
    # print with orange
    text = " ".join(args)
    if add_sep:
        text = "-"*50+"\n"+text+"\n"+"-"*50
    if plain:
        content = text
    else:
        content = color + text + bcolors.ENDC
    print(content)
    add_to_log(content)


def my_print_header(*args):
    my_print("-"*100)
    my_print(*args)