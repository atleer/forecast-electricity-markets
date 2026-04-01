#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader

# %% Check whether google colab kernel is used and download data if it is

# TODO: Current organization doesn't allow for using google colab

import sys 

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    raise EnvironmentError('Google Colab is not yet supported. Please run this script in a local environment.')
    
    # download data
    import urllib.request

    base_url = "https://raw.githubusercontent.com/atleer/forecast-electricity-markets/main/data/processed/opsd-time_series-2020-10-06"
    splits = ['train', 'validation', 'test']
    filename = 'time_series_60min_singleindex.parquet'

    for split in splits:
        dir_path = f"data/processed/opsd-time_series-2020-10-06/{split}"
        os.makedirs(dir_path, exist_ok=True)
        url = f"{base_url}/{split}/{filename}"
        dest = f"{dir_path}/{filename}"
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {split}")

    root_dir = Path('.')

# %% Set path to root directory

if IN_COLAB:
    root_dir = Path('.')
else:
    filepath = Path(__file__).resolve()
    root_dir = filepath.parent.parent

print(root_dir)

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# %% Set seed and turn of non-deterministic behavior for reproducibility

SEED = 2026

def set_seed(seed: int = 2026) -> None:
    """Set seed and disable non-deterministic behavior for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# %% Set device

def set_device(device: str = 'cpu') -> str:
    """Set device to CUDA if available. Set to cpu if not."""
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

device = set_device()

# %% Load data

if 'filepaths' not in globals():

    processed_data_dir = root_dir / 'data/processed/opsd-time_series-2020-10-06'
    filepaths = list(processed_data_dir.glob('**/*60*.parquet'))
    print(filepaths)

for filepath in filepaths:
    if 'train' in filepath.parts:
        df_train = pd.read_parquet(filepath)
    elif 'validation' in filepath.parts:
        df_val = pd.read_parquet(filepath)
    elif 'test' in filepath.parts:
        df_test = pd.read_parquet(filepath)

# %% Choose columns to use in data

features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
targets_column_names = ['DE_price_ahead']
print(f'Columns selected to be used as features: {features_column_names}')
print(f'Columns selected to be used as targets: {targets_column_names}')

# %% Drop NaNs
df_train = df_train[list(set(features_column_names + targets_column_names))].dropna()
df_val = df_val[list(set(features_column_names + targets_column_names))].dropna()
df_test = df_train[list(set(features_column_names + targets_column_names))].dropna()

# %% Scale data

# OPEN QUESTION: Should I use different scaling than standardization with mean?

features_mean = df_train.mean(axis=0).values
targets_mean = df_train.mean(axis=0).values

features_std = df_train.std(axis=0).values
targets_std = df_train.std(axis=0).values


features_train = ((df_train.values - features_mean)/features_std)
features_val = ((df_val.values - features_mean)/features_std)
features_test = ((df_test.values - features_mean)/features_std)

targets_train = ((df_train.values - targets_mean)/targets_std)
targets_val = ((df_val.values - targets_mean)/targets_std)
targets_test = ((df_test.values - targets_mean)/targets_std)

# %% Create sequences

# TODO: This import won't work on google colab server
from src.utils import create_sequences
from models.architectures import Seq2SeqGRU

input_len = 48
horizon = 24

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
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

# %% Train

from src.utils import train, train_with_early_stopping

learning_rates = [0.01, 0.001]
max_epochs = 1

criterion = nn.MSELoss()

model = Seq2SeqGRU(enc_input_size=len(features_column_names), dec_input_size = len(targets_column_names))
model.eval()
y_pred_val = model(X_val, horizon = horizon)
best_loss_val = criterion(y_pred_val, y_val)

for learning_rate in learning_rates:

    model.train()

    optimizer = torch.optim.Adam(lr = learning_rate, params=model.parameters())

    losses_train, losses_val, stopped_epoch = train_with_early_stopping(model, 
                                                    train_dataloader, 
                                                    val_dataloader,
                                                    horizon = 24, 
                                                    optimizer = optimizer, 
                                                    max_epochs=max_epochs
                                                )

    model.eval()
    y_pred_val = model(X_val, horizon = horizon)

    loss_val = criterion(y_pred_val, y_val)

    if  loss_val < best_loss_val:
        best_loss_val = loss_val
        best_learning_rate = learning_rate
        best_hyper_parameters = {
            'learning_rate': learning_rate
        }

        torch.save({
            "stopped_epoch": stopped_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_validation": best_loss_val,
        }, 'checkpoint.pth')








# %%
