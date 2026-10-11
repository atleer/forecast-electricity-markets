# %%
from pathlib import Path
from runpy import run_path
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import sys
import json


# %% Check whether google colab kernel is used and clone the repository if it is

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    import subprocess
    import os

    # Check if clone of repository already exists
    if not Path("forecast-electricity-markets").exists():
        # Clone repository
        BRANCH = None
        cmd = ["git", "clone"]
        if BRANCH:
            print(f"Cloning branch {BRANCH}")
            cmd += ["-b", BRANCH]
        cmd.append("https://github.com/atleer/forecast-electricity-markets.git")
        subprocess.run(
            cmd,
            check=True
        )
    root_dir = Path('forecast-electricity-markets')
    os.chdir(root_dir)
    sys.path.insert(0, str(root_dir))


# %% Configure parameters for run

config = {
    "data_resolution": 60,
    "date": datetime.today().date().isoformat(),
    "model_name": 'seq2seqgru',
    "max_epochs": 2,
    "learning_rates": [0.01, 0.001],
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15
}

# %% Extract relevant time series data from raw data
if config['data_resolution'] not in [15, 30, 60]:
    raise ValueError('Only temporal resolutions of dataset available are 15, 30, and 60 minutes.')

if config['data_resolution'] != 60:
    raise NotImplementedError('Only 60 minutes resolution has price ahead data in currently selected regions in this dataset')

raw_data_dir = Path('data/raw')
sel_data_dir = raw_data_dir / 'from_opsd/opsd-time_series-2020-10-06'
filepaths = list(sel_data_dir.glob(f'**/*{config["data_resolution"]}min*.csv'))

for filepath in tqdm(filepaths, 'Extract time-series data'):
    run_path(
        'scripts/validators/validate_opsd_time_series.py',
        init_globals={
            'filepath': filepath,
        }
    )

# %% Selected feature and target columns and clean data

features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
targets_column_names = ['DE_price_ahead']

processed_data_dir = Path('data/processed')
filepaths = list(processed_data_dir.glob(f'**/all_samples/*{config["data_resolution"]}*.parquet'))

for filepath in tqdm(filepaths, 'Extract and clean data from selected feature and target columns'):
    run_path(
        'src/data_pipeline/extract_feature_and_target_data.py',
        init_globals={
            'filepath': filepath,
            'feature_column_names': features_column_names,
            'targets_column_names': targets_column_names,
        }
    )

# %% Split extracted time series data into train, validation, and test subsets

processed_data_dir = Path('data/processed')
filepaths = list(processed_data_dir.glob(f'**/features_and_targets_extracted/*{config["data_resolution"]}*.parquet'))
filepaths

for filepath in tqdm(filepaths, 'Split into train, val, and test subsets'):
    run_path(
        'src/data_pipeline/split_train_test_val.py',
        init_globals={
            'filepath': filepath,
            'frac_train': config['train_split'],
            'frac_val': config['val_split'],
            'frac_test': config['test_split']
        }
    )

# %% Do forecasting with seq2seq model

processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')
filepaths = list(processed_data_dir.glob(f'**/*{config['data_resolution']}*.parquet'))

run_path(
    'analysis/train_model/seq2seq.py',
    init_globals={
        'filepaths': filepaths,
        'max_epochs': config['max_epochs'],
        'learning_rates': config['learning_rates']
    }
);

# %% Visualize forecasting results

run_path(
    'analysis/visualize_forecast.py',
    init_globals={
        'date': config['date'],
        'model_name': config['model_name']
    },
);


# %%
