# %%
from pathlib import Path
from pydantic import BaseModel, PastDatetime
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

# %% Set working directory

os.chdir(Path(__file__).parent.parent.parent)
os.getcwd()

# %% Get data path(s)

if 'filepath' not in globals():
    raw_data_dir = Path('data/raw')
    selected_data_dir = raw_data_dir / 'from_opsd/opsd-time_series-2020-10-06'
    filepath = list(selected_data_dir.glob('**/*15min*.csv'))[0]
filepath

# %% Load csv data into DataFrame
df = pd.read_csv(filepath)
df.columns

# %% Extract columns of interest
base_cols = ['utc_timestamp', 'cet_cest_timestamp']
patterns = ['DE_wind_generation', 'DE_load_actual']
cols_to_keep = base_cols + [
    col for col in df.columns 
    if any(pattern in col for pattern in patterns)
]
df_sub = df[cols_to_keep]
df_sub.columns

# %% Rename columns

class TimeSeriesTable(BaseModel):
    utc_timestamp: PastDatetime
    cet_cest_timestamp: PastDatetime
    DE_wind_generation: float # from DE_wind_generation_actual
    DE_load_transparency: float # from DE_load_actual_entsoe_transparency
        
conversion_dict = {
    'utc_timestamp': 'utc_timestamp',
    'cet_cest_timestamp': 'cet_cest_timestamp',
    'DE_wind_generation_actual': 'DE_wind_generation',
    'DE_load_actual_entsoe_transparency': 'DE_load_transparency'
}

df_sub_renamed = df_sub.rename(columns=conversion_dict)
df_sub_renamed.columns
# %% Validate fields

rows_out = list()
for row in df_sub_renamed.to_dict(orient='records'):
    rows_out.append(TimeSeriesTable.model_validate(row).model_dump())
df_out = pd.DataFrame(rows_out)


# %% Write to file

table = pa.Table.from_pandas(df_out)

out_dir = Path('data/processed')
out_path = out_dir.joinpath(*filepath.parts[-2:]).with_suffix('.parquet')
out_path.parent.mkdir(exist_ok=True, parents=True)
pq.write_table(table, out_path, compression='snappy')

# %%
