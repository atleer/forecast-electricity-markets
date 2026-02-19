# %%
from pathlib import Path
from runpy import run_path
from tqdm import tqdm
import os

# %% Extract relevant time series data from raw data

raw_data_dir = Path('data/raw')
sel_data_dir = raw_data_dir / 'from_opsd/opsd-time_series-2020-10-06'
filepaths = list(sel_data_dir.glob('**/*60min*.csv'))

for filepath in tqdm(filepaths, 'Extract time-series data'):
    run_path(
        'scripts/processors/process_opsd_time_series.py',
        init_globals={
            'filepath': filepath,
        }
    )

# %% Split extracted time series data into train, validation, and test subsets

processed_data_dir = Path('data/processed')
filepaths = list(processed_data_dir.glob('**/all_samples/*60*.parquet'))

for filepath in tqdm(filepaths, 'Split into train, val, and test subsets'):
    run_path(
        'src/data_pipeline/split_train_test_val.py',
        init_globals={
            'filepath': filepath,
        }
    )


# %%
