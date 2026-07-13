# %% Import libraries
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def split_data(df: pd.DataFrame, frac_train: float = 0.7, frac_val: float = 0.15, frac_test: float = 0.15) -> dict:
    assert frac_train + frac_test + frac_val == 1.0
    
    n_samples_train = int(frac_train*len(df))
    n_samples_val = int(frac_val*len(df))

    # Do split
    df_train = df[:n_samples_train]
    df_val = df[n_samples_train:(n_samples_train+n_samples_val)]
    df_test = df[(n_samples_train+n_samples_val):]

    subsets = {'train': df_train, 'validation': df_val, 'test': df_test}

    return subsets

def write_subsets_to_file(subsets: dict) -> None: 

    # Write to file
    for subset_name, df_subset in subsets.items():
        table = pa.Table.from_pandas(df_subset)

        out_dir = Path('data/processed').joinpath(filepath.parts[-3])
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
subsets = split_data(df)
write_subsets_to_file(subsets=subsets)

