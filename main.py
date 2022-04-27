import argparse
from src.preprocess import preprocess
from utils.setup import setup
from utils.io import my_print, my_print_header, initialize_experiment_folder

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable fast debug mode"
        )
    args = parser.parse_args()
    debug_mode = True if args.debug else False
    mode = "debug mode" if debug_mode else "release mode"
    # Initialize experiment folder and settings
    experiment_dir = initialize_experiment_folder()
    setup()
    # Run preprocessing: feature engineering, cleaning, imputation
    my_print(f"Running in {mode}...")
    preprocess.preprocess(debug_mode=debug_mode)
    my_print_header(f"Done running experiments in {mode}.")
    my_print(f"Experiment results saved to: {experiment_dir}")
