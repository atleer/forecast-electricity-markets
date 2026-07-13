# %% import libraries
import pandas as pd
import pytest
from pathlib import Path
import os
import numpy as np

# %% Change working directory
os.chdir(Path(__file__).parent.parent)

# %% Import functions to test

from src.data_pipeline.preprocessing import clean_data, scale_features_and_targets

# %% Generate synthetic data for tests

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'utc_timestamp': pd.date_range('2020-01-01', periods = 4, freq = 'h'),
        'DE_wind_generation': [1.0, np.nan, 3.0, 4.0],
        'DE_solar_generation': [5, 6, 7, 8],
        'DE_price_ahead': [9.0, 10.0, np.nan, 12.0]
    })
    return df

# %% Test extracting columns and dropping nans

def test_column_extraction_and_nan_drop(sample_data):
    df = sample_data

    features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
    targets_column_names = ['DE_price_ahead']

    df_clean = clean_data(df, features_column_names, targets_column_names)
    assert df_clean.isna().sum().sum() == 0

    expected_column_names = set(features_column_names + targets_column_names + ['utc_timestamp'])
    assert set(df_clean.columns) == expected_column_names
