# %% Import libraries
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# %% Set WD
os.chdir(Path(__file__).parent.parent.parent)
os.getcwd()

# %% Get data paths
if 'filepath' not in globals():
    processed_data_dir = Path('data/processed')
    filepath = list(processed_data_dir.glob('**/all_samples/*60*.parquet'))[0]
filepath

# %% Read processed data
df = pd.read_parquet(filepath)
df

# %% Remove NaNs
valid_mask = df['DE_wind_generation'].notna() & df['DE_price_ahead'].notna()
df_valid = df[valid_mask]
df_valid
# %% Split into train, validation, and test subsets

frac_train = 0.7
frac_val = 0.15
frac_test = 1 - frac_train

n_samples_train = int(frac_train*len(df_valid))
n_samples_val = int(frac_val*len(df_valid))

df_train = df_valid[:n_samples_train]
df_val = df_valid[n_samples_train:(n_samples_train+n_samples_val)]
df_test = df_valid[(n_samples_train+n_samples_val):]

subsets = {'train': df_train, 'validation': df_val, 'test': df_test}

# %%
out_dir = Path('data/processed').joinpath(filepath.parts[-3])
out_path = (out_dir / 'train').joinpath(filepath.parts[-1]).with_suffix('.parquet')
out_path

# %% Write to files

for subset_name, df_subset in subsets.items():
    table = pa.Table.from_pandas(df_subset)

    out_dir = Path('data/processed').joinpath(filepath.parts[-3])
    out_path = (out_dir / subset_name).joinpath(filepath.parts[-1]).with_suffix('.parquet')
    out_path.parent.mkdir(exist_ok=True, parents=True)
    pq.write_table(table, out_path, compression='snappy')


# %%
