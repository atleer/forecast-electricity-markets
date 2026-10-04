# %% Import libraries
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from argparse import ArgumentParser

# %% Argument parser

parser = ArgumentParser('This script splits the dataset into train, validation, and test subsets')

parser.add_argument('--train', type=float, default=0.7, help='Set fraction of data to put into train subset')
parser.add_argument('--val', type=float, default=0.15, help='Set fraction of data to put into validation subset')
parser.add_argument('--test', type=float, default=0.15, help='Set fraction of data to put into test subset')

args = parser.parse_args()

if 'frac_train' not in globals():
    frac_train = args.train

if 'frac_val' not in globals():
    frac_val = args.val

if 'frac_test' not in globals():
    frac_test = args.test

# %% Define functions

def split_data(df: pd.DataFrame, frac_train: float = 0.7, frac_val: float = 0.15, frac_test: float = 0.15) -> dict:
    
    n_samples_train = int(frac_train*len(df))
    n_samples_val = int(frac_val*len(df))

    # Do split
    df_train = df[:n_samples_train]
    df_val = df[n_samples_train:(n_samples_train+n_samples_val)]
    df_test = df[(n_samples_train+n_samples_val):]

    subsets = {'train': df_train, 'validation': df_val, 'test': df_test}

    return subsets

def split_and_write_subsets_to_file(df, frac_train: float = 0.7, frac_val: float = 0.15, frac_test: float = 0.15) -> None:

    subsets = split_data(df, frac_train = frac_train, frac_val = frac_val, frac_test = frac_test)

    split_dir_name = f'{int(100*frac_train)}_{int(100*frac_val)}_{int(100*frac_test)}'
    print(f'Train validation test split: {int(100*frac_train)} {int(100*frac_val)} {int(100*frac_test)}')

    # Write to file
    for subset_name, df_subset in subsets.items():
        table = pa.Table.from_pandas(df_subset)

        out_dir = Path(f'data/processed').joinpath(filepath.parts[-3]) / Path(f'splits/{split_dir_name}')
        out_path = (out_dir / subset_name).joinpath(filepath.parts[-1]).with_suffix('.parquet')
        out_path.parent.mkdir(exist_ok=True, parents=True)
        pq.write_table(table, out_path, compression='snappy')

# %% Get data paths
if 'filepath' not in globals():
    processed_data_dir = Path('data/processed')
    filepath = list(processed_data_dir.glob('**/all_samples/*60*.parquet'))[0]
filepath

# %% Read processed data
df = pd.read_parquet(filepath)

# %%

split_and_write_subsets_to_file(df, frac_train, frac_val, frac_test)

