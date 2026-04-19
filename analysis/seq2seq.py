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
    filepath = Path(__file__).resolve()
    root_dir = filepath.parent.parent

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


# %% Evaluate Model - Make Plots and Calculate Metrics

# TODO: Change this to the one with the lowest validation loss of the set of hyperparameters tuned
# TODO 2: Move to separate file

# load_path = filename
# loaded_model_results = torch.load(load_path)
# load_path


# %% load model with lowest validation loss among models trained in latest run
load_dir = save_dir
idx_lowest_valloss = min(range(len(list(load_dir.glob('**/*.pth')))), key=lambda i: float(list(load_dir.glob('**/*.pth'))[i].stem.split('=')[1]))
load_path = list(load_dir.glob('**/*.pth'))[idx_lowest_valloss]
loaded_model_results = torch.load(load_path)

# %% load model with lowest validation loss among all models trained today

load_dir = save_dir.parent
idx_lowest_valloss = min(range(len(list(load_dir.glob('**/*.pth')))), key=lambda i: float(list(load_dir.glob('**/*.pth'))[i].stem.split('=')[1]))
load_path = list(load_dir.glob('**/*.pth'))[idx_lowest_valloss]
loaded_model_results = torch.load(load_path)

# %% load model with lowest validation loss among all models
load_dir = save_dir.parent.parent
idx_lowest_valloss = min(range(len(list(load_dir.glob('**/*.pth')))), key=lambda i: float(list(load_dir.glob('**/*.pth'))[i].stem.split('=')[1]))
load_path = list(load_dir.glob('**/*.pth'))[idx_lowest_valloss]
loaded_model_results = torch.load(load_path)


# %%
state_dict = loaded_model_results['model_state_dict']

state_dict.keys()

# %%

# infer model arguments from shape of loaded parameters
# encoder input size is last dim of weights input to hidden matrix in layer 0 of encoder (first dim - ignoring batch size and sequence length - is n_features x n_hidden_states or input_size x hidden_size)
enc_input_size = state_dict['encoder.weight_ih_l0'].shape[-1]
# number of hidden states is last dim of hidden to hidden matrix (first dim is n_features x n_hidden_states)
hidden_size = state_dict['encoder.weight_hh_l0'].shape[-1]
# first dimension of fully connected layer is the number of targets (usually just 1), last dim is number of hidden states, which is what the output is calculated from
dec_input_size = state_dict['fc.weight'].shape[0]

# instantiate model with random weights
model = Seq2SeqGRU(enc_input_size=enc_input_size, 
                   dec_input_size=dec_input_size, 
                   hidden_size=hidden_size, 
                   device=device)

# set model parameters
model.load_state_dict(state_dict)
model.eval()

# calculate metrics for loaded model

y_pred_test = model(X_test, horizon=horizon)

fig, axes = plt.subplots(ncols = 2, figsize = (10, 5))

axes[0].set_title('Whole Test Period')
axes[0].plot(y_test.cpu().numpy().flatten(), label = 'Data')
with torch.no_grad():
    axes[0].plot(y_pred_test.cpu().numpy().flatten(), label = 'Model Forecast')

window_start = 10000
window_end = window_start + 200
axes[1].set_title('Small Time Window')
axes[1].plot(y_test.cpu().numpy().flatten()[window_start:window_end])
with torch.no_grad():
    axes[1].plot(y_pred_test.cpu().numpy().flatten()[window_start:window_end])

fig.legend()


# %%

import matplotlib.dates as mdates

# Take only the first horizon step of each sequence → one prediction per timestamp
y_test_np = y_test[:, 0, 0].cpu().numpy()
with torch.no_grad():
    y_pred_test_np = y_pred_test[:, 0, 0].detach().cpu().numpy()

# Dates aligned to predictions: first prediction starts after input_len timesteps
test_dates = df_test['utc_timestamp'].iloc[input_len: input_len + len(y_test_np)]

fig, axes = plt.subplots(ncols = 2, figsize = (10, 5))

axes[0].set_title('Whole Test Period')
axes[0].plot(test_dates, y_test_np, label = 'Data')
axes[0].plot(test_dates, y_pred_test_np, label = 'Model Forecast')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

window_start = 100
window_end = window_start + 200
axes[1].set_title('Small Time Window')
axes[1].plot(test_dates[window_start:window_end], y_test_np[window_start:window_end])
axes[1].plot(test_dates[window_start:window_end], y_pred_test_np[window_start:window_end])
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

fig.legend()
fig.tight_layout()

# %%
