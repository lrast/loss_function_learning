# run analysis on base models
import hydra
import argparse
import sys

import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.analysis.base_model_benchmarks import test_accuracy, corrupted_accuracy

analysis_registry = {'accuracy': test_accuracy,
                     'corrupted_accuracy': corrupted_accuracy}


@hydra.main(config_path="../configs/TTA_train", config_name="analysis_base", version_base="1.2")
def analysis_sweep(cfg: DictConfig):
    """ Map analysis over sweep directory as save results to output file
    """
    root_directory = Path(cfg.input_dir)
    output_file = Path(cfg.output_file)
    analyzer = analysis_registry[cfg.analysis.name]

    all_rows = []
    for i, experiment_dir in tqdm(enumerate(root_directory.iterdir())):
        if not experiment_dir.is_dir():
            continue

        run_config = OmegaConf.load(experiment_dir / '.hydra/config.yaml')

        parameter_vals = {key: OmegaConf.select(run_config, key) for key in
                          cfg.tracked_parameters}

        checkpoint_dir = experiment_dir / 'checkpoints' / cfg.stage / 'best'
        results = analyzer(checkpoint_dir, **cfg.analysis.params)

        for entry in results:
            entry.update(parameter_vals)

        all_rows.extend(results)

        pd.DataFrame(all_rows).to_csv(output_file, index=False)


def parse_args_for_hydra():
    """ Parse command line arguments for Hydra.
    This step allows me to mix bare command line arguments with hydra style
    overrides
    """
    parser = argparse.ArgumentParser(description="Analysis script with Hydra config.")
    # Define the positional argument for the directory
    parser.add_argument(
        'input_dir', 
        type=str, 
        help='The directory path to analyze.',
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='The file path to save the analysis results.',
    )
    # Use parse_known_args to ignore the Hydra overrides (like analysis.setting1=10)
    args, hydra_override = parser.parse_known_args()

    dir_override = f'+input_dir="{args.input_dir}"'
    file_override = f'+output_file="{args.output_file}"'
    hydra_override.append(dir_override)
    hydra_override.append(file_override)
    
    return hydra_override


if __name__ == '__main__':
    # Parse command line arguments for Hydra
    hydra_override = parse_args_for_hydra()
    sys.argv = [sys.argv[0]] + hydra_override
    analysis_sweep()
