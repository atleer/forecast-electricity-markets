# %% Import libaries

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from datetime import datetime

# %% Set device

def set_device(device: str = 'cpu') -> str:
    """Set device to CUDA if available. Set to cpu if not."""
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
device = set_device()

# %% Choose model

model_name = 'Seq2SeqGRU'

# %%

try:
    # save to google drive if using colab kernel
    from google.colab import drive
    drive.mount('/content/drive')

    load_dir = Path(f'/content/drive/MyDrive/colab_notebooks/projects/forecast-electricity-markets/models/{model_name}')
except ImportError:
    import os
    root_dir = Path(__file__).parent.parent
    # save locally if not using colab kernel
    load_dir = root_dir / Path(f'results/models/{model_name}')

load_dir
# %%
list(load_dir.glob('**/*.pth'))

# %% Evaluate Model - Make Plots and Calculate Metrics

# %% load model with lowest validation loss among all models trained today
date = Path(datetime.today().isoformat().split('T')[0])
load_dir = load_dir / date
idx_lowest_valloss = min(range(len(list(load_dir.glob('**/*.pth')))), key=lambda i: float(list(load_dir.glob('**/*.pth'))[i].stem.split('=')[1]))
load_path = list(load_dir.glob('**/*.pth'))[idx_lowest_valloss]
#if torch.cuda.is_available():
loaded_model_results = torch.load(load_path, map_location=torch.device(device))


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

import sys
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from models.architectures import Seq2SeqGRU

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
