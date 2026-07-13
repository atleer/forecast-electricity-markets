# import libraries
import pandas as pd
import pytest
from pathlib import Path
import os
import numpy as np

# Change working directory
os.chdir(Path(__file__).parent.parent)

# Import functions to test
from src.data_pipeline.preprocessing import clean_data, scale_features_and_targets
from src.data_pipeline.split_train_test_val import split_data

# Generate synthetic data for tests
@pytest.fixture
def generate_sample_data():
    df = pd.DataFrame({
        'utc_timestamp': pd.date_range('2020-01-01', periods = 100, freq = 'h'),
        'DE_wind_generation': np.arange(0,100,1, dtype=float),
        'DE_solar_generation': np.arange(100,200,1, dtype=float),
        'DE_price_ahead': np.arange(200,300,1, dtype=float)
    })
    return df

@pytest.fixture
def generate_sample_data_w_nan():
    df = pd.DataFrame({
        'utc_timestamp': pd.date_range('2020-01-01', periods = 4, freq = 'h'),
        'DE_wind_generation': [1.0, np.nan, 3.0, 4.0],
        'DE_solar_generation': [5, 6, 7, 8],
        'DE_price_ahead': [9.0, 10.0, np.nan, 12.0]
    })
    return df

# Test extracting columns and dropping nans
def test_column_extraction_and_nan_drop(generate_sample_data_w_nan):
    df = generate_sample_data_w_nan

    features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
    targets_column_names = ['DE_price_ahead']

    df_clean = clean_data(df, features_column_names, targets_column_names)
    assert df_clean.isna().sum().sum() == 0

    expected_column_names = set(features_column_names + targets_column_names + ['utc_timestamp'])
    assert set(df_clean.columns) == expected_column_names

# Test scaling of features and targets
def test_scale_features_and_targets(generate_sample_data):
    df = generate_sample_data

    subsets = split_data(df)
    df_train = subsets['train']

    features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
    targets_column_names = ['DE_price_ahead']

    features_scaled, targets_scaled = scale_features_and_targets(df_train=df_train, df=df, features_column_names=features_column_names, targets_column_names=targets_column_names)

    features_mean = df_train[features_column_names].mean(axis=0).values
    features_std = df_train[features_column_names].std(axis=0).values
    features_unscaled = features_scaled * features_std + features_mean
    assert np.allclose(features_unscaled, df[features_column_names].values, rtol=1e-10)

    targets_mean = df_train[targets_column_names].mean(axis=0).values
    targets_std = df_train[targets_column_names].std(axis=0).values
    targets_unscaled = targets_scaled * targets_std + targets_mean
    assert np.allclose(targets_unscaled, df[targets_column_names].values, rtol=1e-10)




