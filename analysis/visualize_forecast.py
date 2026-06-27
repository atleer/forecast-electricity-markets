# %% Import libaries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from datetime import datetime
import os
import subprocess
import pandas as pd
import matplotlib.dates as mdates

# %% Change working directory to root of repository
root_dir = Path(__file__).parent.parent
os.chdir(root_dir)

# %% Set device
from src.training.device import set_device
from models.architectures import Seq2SeqGRU
from src.data_pipeline.dataloaders import build_dataloaders

device = set_device()

# %% Choose model
if 'model_name' not in globals():
    model_name = 'Seq2SeqGRU'
load_dir = root_dir / Path(f'results/models/{model_name}')

# %% Sync model checkpoints from google drive to local folder
gdrive_path = f"gdrive:colab_notebooks/projects/forecast-electricity-markets/models/{model_name}"
subprocess.run(["rclone", 'copy', gdrive_path, str(load_dir)], check=True)

# %% Evaluate Model - Make Plots and Calculate Metrics

# Load model with lowest validation loss among all models
idx_lowest_valloss = min(range(len(list(load_dir.glob('**/*.pth')))), key = lambda i: float(list(load_dir.glob('**/*.pth'))[i].stem.split('=')[1]))
path_lowest_valloss = list(load_dir.glob('**/*.pth'))[idx_lowest_valloss]
date_benchmark_model = path_lowest_valloss.parts[-3]
model_benchmark = torch.load(path_lowest_valloss, map_location=device)

# %% # Load model with lowest validation loss among models trained on specific day

#TODO: (Maybe) add argument parser to this file where a specific date is set and remove code below
if 'filepath' not in globals():
    date = Path(datetime.today().isoformat().split('T')[0])
    print('Date not provided; using today\'s data.')
    filepath = load_dir / date

if len(list(Path(filepath).glob('**/*.pth'))) == 0:
    raise FileNotFoundError(
        f"No model checkpoint files found in {filepath}"
    )

idx_lowest_valloss = min(range(len(list(filepath.glob('**/*.pth')))), key = lambda i: float(list(filepath.glob('**/*.pth'))[i].stem.split('=')[1]))
path_lowest_valloss = list(filepath.glob('**/*.pth'))[idx_lowest_valloss]
date_model_selected = path_lowest_valloss.parts[-3]
model_selected = torch.load(path_lowest_valloss, map_location=device)

# %% Load test data
# NOTE: Uneccessary to load from train and val folders as well here, but will have to refactor build_dataloaders if I don't load them
processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')
testdata_filepath = list(processed_data_dir.glob('**/*60*.parquet'))

df_test = pd.read_parquet(processed_data_dir / Path('test/time_series_60min_singleindex.parquet'))
# %%

# Choose columns to use in data

features_column_names = ['DE_wind_generation', 'DE_solar_generation', 'DE_price_ahead']
targets_column_names = ['DE_price_ahead']
print(f'Columns selected to be used as features: {features_column_names}')
print(f'Columns selected to be used as targets: {targets_column_names}')

input_len = 48
horizon = 24

_, _, test_dataloader = build_dataloaders(
    filepaths=testdata_filepath,
    input_len=input_len,
    horizon=horizon,
    features_column_names=features_column_names,
    targets_column_names=targets_column_names,
    batch_size=256,
    device=device,
)

X_test, y_test = test_dataloader.dataset.tensors


# %% Make figures

models = {
    date_benchmark_model: model_benchmark,
    date_model_selected: model_selected
}

for date_model, model_to_load in models.items():
    state_dict = model_to_load['model_state_dict']

    # infer model arguments from shape of loaded parameters
    # encoder input size is last dim of weights input to hidden matrix in layer 0 of encoder (first dim - ignoring batch size and sequence length - is n_features x n_hidden_states or input_size x hidden_size)
    enc_input_size = state_dict['encoder.weight_ih_l0'].shape[-1]
    # number of hidden states is last dim of hidden to hidden matrix (first dim is n_features x n_hidden_states)
    hidden_size = state_dict['encoder.weight_hh_l0'].shape[-1]
    # first dimension of fully connected layer is the number of targets (usually just 1), last dim is number of hidden states, which is what the output is calculated from
    dec_input_size = state_dict['fc.weight'].shape[0]

    model = Seq2SeqGRU(enc_input_size=enc_input_size, 
                   dec_input_size=dec_input_size, 
                   hidden_size=hidden_size, 
                   device=device)
    
    model.load_state_dict(state_dict)
    model.eval()

    # Calculate model prediction on test dataset

    y_pred_test = model(X_test, horizon=horizon)

    # Take only the first horizon step of each sequence → one prediction per timestamp
    y_test_np = y_test[:, 0, 0].cpu().numpy()
    with torch.no_grad():
        y_pred_test_np = y_pred_test[:, 0, 0].detach().cpu().numpy()

    # Calculate metrics for model prediction
    metrics = {}
    metrics['MSE'] = np.mean((y_pred_test_np - y_test_np)**2) # mean square error
    metrics['MAE'] = np.mean(np.abs(y_pred_test_np - y_test_np)) # mean absolute error
    metrics['MASE'] = np.mean(np.abs(y_pred_test_np - y_test_np)) / np.mean(np.abs(y_test_np[1:] - y_test_np[:-1])) # mean absolute scaled error
    metrics['RelMSE'] = metrics['MSE'] / np.mean(y_test_np**2) # relative meas square error
    metrics['FB'] = (np.sum(y_pred_test_np) - np.sum(y_test_np)) / np.sum(y_test_np)

    # TODO: Double check the dates here
    # Dates aligned to predictions: first prediction starts after input_len timesteps
    test_dates = df_test['utc_timestamp'].iloc[input_len: input_len + len(y_test_np)]

    fig, axes = plt.subplots(ncols = 2, figsize = (10, 5))

    axes[0].set_title('Whole Test Period')
    axes[0].plot(test_dates, y_test_np, label = 'Data')
    axes[0].plot(test_dates, y_pred_test_np, label = 'Model Forecast')
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].set_ylabel('Z-scored Price')
    axes[0].set_xlabel('Dates')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    window_start = 100
    window_end = window_start + 200
    axes[1].set_title('Small Time Window')
    axes[1].plot(test_dates[window_start:window_end], y_test_np[window_start:window_end])
    axes[1].plot(test_dates[window_start:window_end], y_pred_test_np[window_start:window_end])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].set_xlabel('Dates')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(f'Model: {model_name} {date_model} - prediction on test dataset\nMetrics: '+', '.join(f"{name}; {value:.2f}" for name, value in metrics.items()))
    fig.legend()
    fig.tight_layout()
    fig.savefig(f'results/figures/{model_name}_{date_model}_prediction_on_test_set', bbox_inches = 'tight');


# %%
