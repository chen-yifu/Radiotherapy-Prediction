import pandas as pd
from sklearn.impute import *
import numpy as np
from collections import Counter
from src.preprocess import preprocess
from utils.setup import *
from utils.printers import *

# p = Printer()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable fast debug mode")
    args = parser.parse_args()
    debug_mode = True  if args.debug else False

    my_print(f"Running experiments in {'fast-debug mode' if debug_mode else 'release mode'}...")
    setup()
    
    preprocess.preprocess(debug_mode=debug_mode)
    
    my_print_header(f"Experiment done running in {'fast-debug mode' if debug_mode else 'release mode'}!")