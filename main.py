import argparse
from src.preprocess import preprocess
from utils.setup import setup
from utils.io import my_print, my_print_header, initialize_experiment_folder
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
    # Argument to select a previous experiment directory
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=False,
        help="path to previous experiment directory with pickled models"
    )
    # Parse and read arguments
    args = parser.parse_args()
    config.debug_mode = True if args.debug else False
    experiment_dir = args.experiment_dir
    if experiment_dir:
        config.experiment_dir = experiment_dir
    else:
        # Initialize new experiment folder and settings
        experiment_dir = config.experiment_dir = initialize_experiment_folder()
    my_print(f"Using experiment folder: {experiment_dir}")

    mode_str = "debug mode" if config.debug_mode else "release mode"
    my_print(f'Running in {mode_str}...')

    # Run preprocessing: feature engineering, cleaning, imputation
    preprocess.preprocess(experiment_dir=experiment_dir)
    my_print_header(f"Done running experiments in {mode_str}.")
    my_print(f"Experiment results saved to: {experiment_dir}")
