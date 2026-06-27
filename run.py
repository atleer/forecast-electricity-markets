# %%
from pathlib import Path
from runpy import run_path
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser

# %% Create argument parser
parser = ArgumentParser(description='This program is the workflow manager of the pipeline.')

parser.add_argument('--data_resolution', type=int, help='This argument sets which temporal resolution time series data to use (15, 30, or 60 min, default 60 min)')
parser.add_argument('--date', type=str, help='Format: YYYY-MM-DD. Pick the model checkpoint to calculate metrics and visualize results by providing the date that model training was started.')
parser.add_argument('--model_name', type=str, help='Name of model architecture to use for forecasting')
args = parser.parse_args(args=['--data_resolution', '60', '--date', '2026-05-17', '--model_name', 'Seq2SeqGRU'])

# %% Extract relevant time series data from raw data
if args.data_resolution not in [15, 30, 60]:
    raise ValueError('Only temporal resolutions of dataset available are 15, 30, and 60 minutes.')

if args.data_resolution != 60:
    raise NotImplementedError('Only 60 minutes resolution has price ahead data in currently selected regions in this dataset')

raw_data_dir = Path('data/raw')
sel_data_dir = raw_data_dir / 'from_opsd/opsd-time_series-2020-10-06'
filepaths = list(sel_data_dir.glob(f'**/*{args.data_resolution}min*.csv'))

for filepath in tqdm(filepaths, 'Extract time-series data'):
    run_path(
        'scripts/processors/process_opsd_time_series.py',
        init_globals={
            'filepath': filepath,
        }
    )

# %% Split extracted time series data into train, validation, and test subsets

processed_data_dir = Path('data/processed')
filepaths = list(processed_data_dir.glob(f'**/all_samples/*{args.data_resolution}*.parquet'))

for filepath in tqdm(filepaths, 'Split into train, val, and test subsets'):
    run_path(
        'src/data_pipeline/split_train_test_val.py',
        init_globals={
            'filepath': filepath,
        }
    )


# %% Do forecasting with seq2seq model

processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')
filepaths= list(processed_data_dir.glob(f'**/*{args.data_resolution}*.parquet'))

run_path(
    'analysis/train_model/seq2seq.py',
    init_globals={'filepaths': filepaths}
);


# %% Visualize forecasting results

filepath = Path(f'results/models/{args.model_name}/{args.date}')

run_path(
    'analysis/visualize_forecast.py',
    init_globals={'filepath': filepath},
)


# %%
