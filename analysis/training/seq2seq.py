# %% Autoreload functions for quicker checks of local module modifications

%load_ext autoreload
%autoreload 2

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
import os

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
    root_dir = Path(__file__).resolve().parent.parent.parent


# %%
os.chdir(root_dir)

from models.architectures import Seq2SeqGRU
from src.training.device import set_device
from src.training.save_checkpoint import make_checkpoint_dir
from src.training.reproducibility import set_seed
from src.data_pipeline.dataloaders import build_dataloaders

SEED = 2026
set_seed(SEED)
device = set_device()

# %% Choose columns to use in data

features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
targets_column_names = ['DE_price_ahead']
print(f'Columns selected to be used as features: {features_column_names}')
print(f'Columns selected to be used as targets: {targets_column_names}')

input_len = 48
horizon = 24

train_dataloader, val_dataloader, _ = build_dataloaders(
    processed_data_dir=Path('data/processed/opsd-time_series-2020-10-06'),
    input_len=input_len,
    horizon=horizon,
    features_column_names=features_column_names,
    targets_column_names=targets_column_names,
    batch_size=256,
    device=device,
)

X_val, y_val = val_dataloader.dataset.tensors

# %% Create directories to save results 
# TODO: Technically this ought to be part of run cell, otherwise user have to remember to run this cell before each training to not override results when running interactively, but doesn't work for some reason

model_name = 'Seq2SeqGRU'

save_checkpoint_dir = make_checkpoint_dir(model_name)

# %% Train model

from src.training.train_loops import train, train_with_early_stopping

learning_rates = [0.01, 0.001]
max_epochs = 1

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

        filename = f'{save_checkpoint_dir}/loss_val={best_loss_val:.3f}.pth'
        torch.save({
            "stopped_epoch": stopped_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_validation": best_loss_val,
            "learning_rate": learning_rate,
        }, filename)


# %% Sync local and google drive folders with model checkpoints
import subprocess

try:
    from google.colab import drive # if it was not mounted earlier, it was run locally
except ImportError:
    # Copy model checkpoint to google drive if run locally
    # local kernel: upload to google drive via rclone
    date = Path(datetime.today().isoformat().split('T')[0])
    gdrive_dest = f"gdrive:colab_notebooks/projects/forecast-electricity-markets/models/{model_name}/{date}/{save_checkpoint_dir.name}"
    subprocess.run(["rclone", "copy", str(save_checkpoint_dir), gdrive_dest], check=True)
