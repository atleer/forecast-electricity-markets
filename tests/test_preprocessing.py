# %% import libraries
import pandas as pd
import pytest
from pathlib import Path
import os
import numpy as np

# %% Change working directory
os.chdir(Path(__file__).parent.parent)

# %% Import functions to test

from src.data_pipeline.preprocessing import clean_and_extract_data, scale_features_and_targets

# %% Generate synthetic data for tests

@pytest.fixture
def sample_data():
    df_train = pd.DataFrame({
        'utc_timestamp': pd.date_range('2020-01-01', periods = 4, freq = 'h'),
        'DE_wind_generation': [1.0, np.nan, 3.0, 4.0],
        'DE_solar_generation': [5, 6, 7, 8],
        'DE_price_ahead': [9.0, 10.0, np.nan, 12.0]
    })
    df_val = df_train.copy()
    df_test = df_train.copy()

    return df_train, df_val, df_test

# %% Test extracting columns and dropping nans

def test_column_extraction_and_nan_drop(sample_data):
    df_train, df_val, df_test = sample_data

    features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
    targets_column_names = ['DE_price_ahead']

    df_train_clean, df_val_clean, df_test_clean = clean_and_extract_data(features_column_names, targets_column_names, df_train, df_val, df_test)
    assert df_train_clean.isna().sum().sum() == 0
    assert df_val_clean.isna().sum().sum() == 0
    assert df_test_clean.isna().sum().sum() == 0

    expected_column_names = set(features_column_names + targets_column_names + ['utc_timestamp'])
    assert set(df_train.columns) == expected_column_names
