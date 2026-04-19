# %% Autoreload functions for quicker checks of local module modifications

#%load_ext autoreload
#%autoreload 2

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime
import sys

# %% Check whether google colab kernel is used and clone the repository if it is

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    import subprocess

    # Check if clone of repository already exists
    if not Path("forecast-electricity-markets").exists():
        # Clone repository
        BRANCH = None
        cmd = ["git", "clone"]
        if BRANCH:
            print(f"Cloning branch {BRANCH}")
            cmd += ["-b", BRANCH]
        cmd.append("https://github.com/atleer/forecast-electricity-markets.git")
        subprocess.run(
            cmd,
            check=True
        )
    root_dir = Path('forecast-electricity-markets')
else:
    root_dir = Path(__file__).resolve().parent.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.architectures import Seq2SeqGRU

# %% Set seed and turn of non-deterministic behavior for reproducibility

SEED = 2026

def set_seed(seed: int = 2026) -> None:
    """Set seed and disable non-deterministic behavior for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # when set to true, cuda tests multiple convolution algorithm to find fastes for shape, set to true the default is used
    torch.backends.cudnn.benchmark = False
    # only use deterministic algorithms
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

# %%

from src.data_pipeline.preprocessing import clean_and_extract_data, scale_features_and_targets

df_train, df_val, df_test = clean_and_extract_data(features_column_names = features_column_names, 
                                                    targets_column_names = targets_column_names, 
                                                    df_train = df_train,
                                                    df_val = df_val,
                                                    df_test = df_test
                                                    )
    

# %% Scale data
features_train, targets_train = scale_features_and_targets(df_train, df_train, features_column_names, targets_column_names)
features_val, targets_val = scale_features_and_targets(df_train, df_val, features_column_names, targets_column_names)
features_test, targets_test = scale_features_and_targets(df_train, df_test, features_column_names, targets_column_names)

# %% Create sequences

from src.utils import create_sequences

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

# %% Create directories to save results 
# TODO: Technically this ought to be part of run cell, otherwise user have to remember to run this cell before each training to not override results when running interactively, but doesn't work for some reason

date = Path(datetime.today().isoformat().split('T')[0])
try:
    # save to google drive if using colab kernel
    from google.colab import drive
    drive.mount('/content/drive')

    save_dir = Path(f'/content/drive/MyDrive/colab_notebooks/projects/forecast-electricity-markets/models/{model.__class__.__name__}') / date
except ImportError:
    # save locally if not using colab kernel
    save_dir = root_dir / Path(f'results/models/{model.__class__.__name__}') / date
if save_dir.exists():
    num_runs = len(list(save_dir.glob("*/")))
    save_dir = save_dir / Path(f"Run{num_runs}")
else:
    save_dir = save_dir / Path(f"Run0")
save_dir.mkdir(exist_ok=True, parents=True)

# %% Train model

model_name = 'Seq2SeqGRU'

date = Path(datetime.today().isoformat().split('T')[0])
try:
    # save to google drive if using colab kernel
    from google.colab import drive
    drive.mount('/content/drive')

    save_dir = Path(f'/content/drive/MyDrive/colab_notebooks/projects/forecast-electricity-markets/models/{model_name}') / date
except ImportError:
    # save locally if not using colab kernel
    save_dir = root_dir / Path(f'results/models/{model_name}') / date
if save_dir.exists():
    num_runs = len(list(save_dir.glob("*/")))
    save_dir = save_dir / Path(f"Run{num_runs}")
else:
    save_dir = save_dir / Path(f"Run0")
save_dir.mkdir(exist_ok=True, parents=True)

from src.utils import train, train_with_early_stopping

learning_rates = [0.01, 0.001]
max_epochs = 10

criterion = nn.MSELoss()

for learning_rate in learning_rates:
    model = Seq2SeqGRU(enc_input_size=len(features_column_names), 
                   dec_input_size = len(targets_column_names))
    model.to(device)
    model.eval()
    y_pred_val = model(X_val, horizon = horizon)
    best_loss_val = criterion(y_pred_val, y_val)

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

    # save trained model
    if  loss_val < best_loss_val:
        best_loss_val = loss_val
        best_learning_rate = learning_rate
        best_hyper_parameters = {
            'learning_rate': learning_rate
        }

        filename = f'{save_dir}/loss_val={best_loss_val:.3f}.pth'
        torch.save({
            "stopped_epoch": stopped_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_validation": best_loss_val,
            "learning_rate": learning_rate,
        }, filename)


# %%
