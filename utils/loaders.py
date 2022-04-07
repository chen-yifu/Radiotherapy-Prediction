import json 

def load_col_type(path):
    with open(path, "r") as f:
        col_type = json.load(f)
    return col_type

