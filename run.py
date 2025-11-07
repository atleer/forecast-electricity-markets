# %%
from pathlib import Path
from runpy import run_path
from tqdm import tqdm
import os

# %%

raw_data_dir = Path('data/raw')
sel_data_dir = raw_data_dir / 'from_opsd/opsd-time_series-2020-10-06'

filepaths = list(sel_data_dir.glob('**/*15min*.csv'))

# %%

for filepath in tqdm(filepaths, 'Extract time-series data'):
    run_path(
        'scripts/processors/process_opsd_time_series.py',
        init_globals={
            'filepath': filepath,
        }
    )

# %%
