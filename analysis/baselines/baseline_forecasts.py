# %% Autoreload functions for quicker checks of local module modifications

# %load_ext autoreload
# %autoreload 2

#%% Import libraries
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %% Change working directory

root_dir = Path(__file__).resolve().parent.parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

# %%

from src.training.save_checkpoint import make_checkpoint_dir
from src.training.reproducibility import set_seed
from src.data_pipeline.dataloaders import load_splits, build_dataloaders
from src.utils import rclone_bin

SEED = 2026
set_seed(SEED)

if 'filepaths' not in globals():
    processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')

    filepaths = list(processed_data_dir.glob('**/*60*.parquet'))
    print(filepaths)

# %% Create model config if not provided
if 'model_config' not in globals():
    model_config = dict()
    model_config['input_len'] = 48
    model_config['horizon'] = 24
    model_config['date_run'] = datetime.today().date().isoformat()

# %% Load data for moving average

features_column_names = ['DE_price_ahead']
targets_column_names = ['DE_price_ahead']

_, _, test_dataloader = build_dataloaders(
    filepaths=filepaths,
    input_len=model_config['input_len'],
    horizon=model_config['horizon'],
    features_column_names=features_column_names,
    targets_column_names=targets_column_names,
    batch_size=256,
)

X_test, y_test = test_dataloader.dataset.tensors
X_test, y_test = X_test.numpy(), y_test.numpy()

_, _, df_test = load_splits(filepaths=filepaths)

# %% Create directories to save results 

model_name = 'baselines'

save_checkpoint_dir = make_checkpoint_dir(model_name)
# %%
seasonal_decompose?

# %%

df_test['utc_timestamp']
# %% Decompose into trend, seasonality and residuals (NOTE: Time period of test data not long enough to get seasonality)
from statsmodels.tsa.seasonal import seasonal_decompose


series = df_test.set_index('utc_timestamp')[targets_column_names[0]]

# additive model
decomposition_additive = seasonal_decompose(series, model = 'additive', )

fig = decomposition_additive.plot()
fig.suptitle('Decomposition of Day Ahead Price using Additive Model', fontsize=14)
ax = fig.axes[-1]
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax = fig.axes[0]
ax.set_ylabel('Price (€/MWh)')

# %%

# multiplicative model
# decomposition_mult = seasonal_decompose(df_test[targets_column_names], model = 'multiplicative', period=365)

# plt.figure()
# decomposition_mult.plot()
# plt.suptitle('Decomposition of Day Ahead Price using Multiplicative Model', fontsize=14)




# %%

y_pred = list()
for i in range(X_test.shape[0]):
    y_pred.append(X_test[i,:,:].mean())

y_pred = np.array(y_pred)
y_pred.shape

# %% Plot

test_dates = df_test['utc_timestamp'].iloc[model_config['input_len']: model_config['input_len'] + len(y_test)]

plt.plot(test_dates, y_test.mean(axis=(1,2)), label='Data')
plt.plot(test_dates, y_pred, label = 'Moving Avg.')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.title('Moving average')





# %%
