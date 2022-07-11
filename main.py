import argparse
from src.preprocess import preprocess
from utils.setup import setup
from utils.io import print_and_log, print_and_log_w_header, initialize_experiment_folder
import utils.config as config

if __name__ == "__main__":
    setup()
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable fast debug mode"
        )
    parser.add_argument(
        "--impute_only",
        action="store_true",
        help="use imputation only, skip pre-processing"
        )
    parser.add_argument(
        "--take_breaks",
        action="store_true",
        help="whether to sleep CPU for 3 minutes between columns"
        )
    # Argument to select a previous experiment directory
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=False,
        help="path to previous experiment directory with pickled models"
    )
    parser.add_argument(
        "--df_path",
        type=str,
        required=False,
        help="path to the un-preprocessed DataFrame"
    )
    
    # Parse and read arguments
    args = parser.parse_args()
    config.debug_mode = True if args.debug else False
    config.take_breaks = True if args.take_breaks else False
    experiment_dir = args.experiment_dir
    if experiment_dir:
        config.experiment_dir = experiment_dir
    else:
        # Initialize new experiment folder and settings
        experiment_dir = config.experiment_dir = initialize_experiment_folder()
    print_and_log(f"Using experiment folder: {experiment_dir}")

    mode_str = "debug mode" if config.debug_mode else "release mode"
    print_and_log(f'Running in {mode_str}...')

    # Run preprocessing: feature engineering, cleaning, imputation
    preprocess.preprocess(
        experiment_dir=experiment_dir,
        df_path=args.df_path,
        impute_only=args.impute_only
    )
    print_and_log_w_header(f"Done running experiments in {mode_str}.")
    print_and_log(f"Experiment results saved to: {experiment_dir}")
