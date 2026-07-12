# %% import libraries
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# %% Change working directory
os.chdir(Path(__file__).parent.parent.parent)
os.getcwd()

# %% Load function to clean and extract data

from src.data_pipeline.preprocessing import clean_data

# %% Get data paths
if 'filepath' not in globals():
    processed_data_dir = Path('data/processed')
    filepath = list(processed_data_dir.glob('**/all_samples/*60*.parquet'))[0]
filepath

# %% Set feature and target columns if not provided

if 'features_column_names' not in globals():
    features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']

if 'targets_column_names' not in globals():
    targets_column_names = ['DE_price_ahead']

# %% Read processed data
df = pd.read_parquet(filepath)
df

# %%
df_valid = clean_data(df, features_column_names=features_column_names, targets_column_names=targets_column_names)
df_valid

# %% Write to file

table = pa.Table.from_pandas(df_valid)

out_dir = Path('data/processed').joinpath(filepath.parts[-3]) / Path('features_and_targets_extracted')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir.joinpath(filepath.stem).with_suffix('.parquet')
pq.write_table(table, out_path, compression='snappy')
