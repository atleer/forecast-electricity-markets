# %% Import libraries
import pytest
import os
from pathlib import Path
import numpy as np
import pandas as pd

# %% Change working directory
os.chdir(Path(__file__).parent.parent)

# %% 

from src.data_pipeline.split_train_test_val import split_data, write_subsets_to_file

# %% Generate synthetic data for tests

@pytest.fixture
def generate_sample_data():
    df = pd.DataFrame({
        'utc_timestamp': pd.date_range('2020-01-01', periods = 100, freq = 'h'),
        'DE_wind_generation': np.arange(0,100,1),
        'DE_solar_generation': np.arange(100,200,1),
        'DE_price_ahead': np.arange(200,300,1)
    })
    return df

# %% 

def test_split_data(generate_sample_data, frac_train: float = 0.7, frac_val: float = 0.15, frac_test: float = 0.15):
    df = generate_sample_data

    subsets = split_data(df, frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    df_train, df_val, df_test = list(subsets.values())
    assert len(df_train) + len(df_val) + len(df_test) == len(df)
    assert df_train.index[-1] == df_val.index[0] - 1
    assert df_val.index[-1] == df_test.index[0] - 1
    assert df_test.index[-1] == df.index[-1]
