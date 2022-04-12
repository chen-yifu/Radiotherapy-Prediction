from utils.get_timestamp import *
from utils.printers import *
import os
import pickle
import pandas as pd
out_dir = f"data/preprocessed"


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
    # global cur_timestamp
    file_path = os.path.join(out_dir, cur_timestamp, file_name)
    df.to_csv(file_path, index=False)
    my_print(f"Saved {description} DataFrame to: {file_path}.")
    return file_path
    
def save_experiment_pickle(object, file_name: str, description: str):
    file_path = os.path.join(out_dir, cur_timestamp, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    my_print(f"Pickled {description} Object to: {file_path}.")
    return file_path

