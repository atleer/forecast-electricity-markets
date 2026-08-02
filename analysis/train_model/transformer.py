# %% Autoreload functions for quicker checks of local module modifications

%load_ext autoreload
%autoreload 2

#%% Import libraries
from pathlib import Path

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

from models.architectures import Transformer
from src.training.device import set_device
from src.training.save_checkpoint import make_checkpoint_dir
from src.training.reproducibility import set_seed
from src.data_pipeline.dataloaders import build_dataloaders

SEED = 2026
set_seed(SEED)
device = set_device()

if 'filepaths' not in globals():
    processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')

    filepaths = list(processed_data_dir.glob('**/*60*.parquet'))
    print(filepaths)

# %% Build dataloaders

features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
targets_column_names = ['DE_price_ahead']

input_len = 48
horizon = 24

train_dataloader, val_dataloader, _ = build_dataloaders(
    filepaths=filepaths,
    input_len=input_len,
    horizon=horizon,
    features_column_names=features_column_names,
    targets_column_names=targets_column_names,
    batch_size=256,
    device=device,
)

X_val, y_val = val_dataloader.dataset.tensors

# %% Create directories to save results 

model_name = 'Transformer'

save_checkpoint_dir = make_checkpoint_dir(model_name)

# %%
Transformer(enc_input_size=len(features_column_names), dim_model=dim_model, num_heads=num_heads, num_layers=1, horizon=horizon,)
# %% Train model

from src.training.train_loops import train, train_with_early_stopping

# TODO: You should use a model config dictionary
if 'model_config' not in globals():
    model_config = dict()

    model_config['max_epochs'] = 1
    model_config['learning_rates'] = [0.01, 0.001]
    model_config['dim_model'] = 32
    model_config['num_heads'] = 8


criterion = nn.MSELoss()

for learning_rate in learning_rates:
    model = Transformer(
        enc_input_size=len(features_column_names), 
        dims_model=dims_model,
        num_heads=num_heads,
        num_layers=1,
        horizon=horizon,
    )
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
# %%
