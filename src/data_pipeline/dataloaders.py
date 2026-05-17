import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from typing import Union
from src.utils import create_sequences
from src.data_pipeline.preprocessing import clean_and_extract_data, scale_features_and_targets

def load_splits(filepaths):
    """Load train, validation, and test datasets"""

    for filepath in filepaths:
        if 'train' in filepath.parts:
            df_train = pd.read_parquet(filepath)
        elif 'validation' in filepath.parts:
            df_val = pd.read_parquet(filepath)
        elif 'test' in filepath.parts:
            df_test = pd.read_parquet(filepath)

    return df_train, df_val, df_test


def build_dataloaders(
        filepaths: list[Union[str, Path]], 
        features_column_names: list[str],
        targets_column_names: list[str],
        input_len: int = 24,
        horizon: int = 48,
        device: str = 'cpu',
        batch_size: int = 256,
    ):
    """Create dataloaders for train, validation, and test datasets
    
    Args:
        filepaths: list of paths to files containing train, validation, and test subdatasets
        input_len: past timesteps to use in forecast (encoder window length)
        horizon: future timesteps to forecast (decoder window length)
        features_column_names: names of columns in data to use in forecast
        targets_column_names: names of columns in data to forecast
        batch_size: batch size in train dataloader
        device: cpu or cuda
    """

    df_train, df_val, df_test = load_splits(filepaths)

    df_train, df_val, df_test = clean_and_extract_data(features_column_names = features_column_names, 
                                                        targets_column_names = targets_column_names, 
                                                        df_train = df_train,
                                                        df_val = df_val,
                                                        df_test = df_test
                                                        )
        

    # Scale data
    features_train, targets_train = scale_features_and_targets(df_train, df_train, features_column_names, targets_column_names)
    features_val, targets_val = scale_features_and_targets(df_train, df_val, features_column_names, targets_column_names)
    features_test, targets_test = scale_features_and_targets(df_train, df_test, features_column_names, targets_column_names)


    # Create sequences
    if targets_train.ndim == 1:
        targets_train = np.expand_dims(targets_train, axis=-1)
        targets_val = np.expand_dims(targets_val, axis=-1)
        targets_test = np.expand_dims(targets_test, axis=-1)

    X_train, y_train = create_sequences(features_train, targets_train, input_len, horizon)
    X_val, y_val = create_sequences(features_val, targets_val, input_len, horizon)
    X_test, y_test = create_sequences(features_test, targets_test, input_len, horizon)

    X_train = torch.from_numpy(X_train).to(dtype = torch.float32, device=device)
    y_train = torch.from_numpy(y_train).to(dtype = torch.float32, device=device)

    X_val = torch.from_numpy(X_val).to(dtype = torch.float32, device=device)
    y_val = torch.from_numpy(y_val).to(dtype = torch.float32, device=device)

    X_test = torch.from_numpy(X_test).to(dtype = torch.float32, device=device)
    y_test = torch.from_numpy(y_test).to(dtype = torch.float32, device=device)

    # %% Create DataLoaders

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
