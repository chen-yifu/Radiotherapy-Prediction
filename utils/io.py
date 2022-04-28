# IO Helpers for loading, saving, and printing
from utils.get_timestamp import get_timestamp
import os
import dill as pickle
import json
import pandas as pd
import utils.config as config

out_dir = f"data/experiments/"


# LOADER FUNCTIONS #
def load_col_type(path):
    with open(path, "r") as f:
        col_type = json.load(f)
    return col_type


# SAVER FUNCTIONS #
def initialize_experiment_folder():
    # Make a folder for saving the experiment data
    cur_timestamp = get_timestamp()
    experiment_dir = os.path.join(out_dir, cur_timestamp)
    os.mkdir(experiment_dir)
    config.experiment_dir = experiment_dir
    my_print("Created experiment folder:", experiment_dir)
    return experiment_dir


def save_experiment_df(df: pd.DataFrame, file_name: str, description: str):
    # Save the dataframe to experiment folder
    experiment_dir = config.experiment_dir
    file_path = os.path.join(experiment_dir, file_name)
    df.to_csv(file_path, index=False)
    my_print_header(f"Saved {description} DataFrame to: {file_path}.")
    return file_path


def save_experiment_pickle(object, file_name: str, description: str):
    experiment_dir = config.experiment_dir
    file_path = os.path.join(experiment_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    my_print_header(f"Saved {description} Object to: {file_path}.")
    return file_path


def add_to_log(content):
    experiment_dir = config.experiment_dir
    log_path = os.path.join(experiment_dir, "log.txt")
    if not os.path.exists(log_path):
        content = f"Use this command to view log file:\ncat {log_path}\n" \
            + content
    with open(log_path, 'a') as f:
        f.write(content + '\n')


# PRINTER FUNCTIONS #
class bcolors:
    # Helper class to print in terminal with colors
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
