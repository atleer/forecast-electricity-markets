# %% Autoreload functions for quicker checks of local module modifications

# %load_ext autoreload
# %autoreload 2

#%% Import libraries
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsforecast.models import MSTL, AutoARIMA
from statsforecast import StatsForecast

# %% Change working directory
root_dir = Path(__file__).resolve().parent.parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

# %% Create model config if not provided
if 'model_config' not in globals():
    model_config = dict()
    model_config['input_len'] = 48
    model_config['horizon'] = 24
    model_config['date_run'] = datetime.today().date().isoformat()
    model_config['data_split'] = '50_0_50'

# %% Load functions
from src.training.save_checkpoint import make_checkpoint_dir
from src.training.reproducibility import set_seed
from src.data_pipeline.dataloaders import load_splits, build_dataloaders
from src.utils import rclone_bin

SEED = 2026
set_seed(SEED)

if 'filepaths' not in globals():
    processed_data_dir = Path('data/processed/opsd-time_series-2020-10-06')
    filepaths = list(processed_data_dir.glob(f'**/{model_config['data_split']}/**/*60*.parquet'))


# %% Load data for moving average
features_column_names = ['DE_price_ahead']
targets_column_names = ['DE_price_ahead']

train_dataloader_, _, test_dataloader = build_dataloaders(
    filepaths=filepaths,
    input_len=model_config['input_len'],
    horizon=model_config['horizon'],
    features_column_names=features_column_names,
    targets_column_names=targets_column_names,
    batch_size=256,
)

X_test, y_test = test_dataloader.dataset.tensors
X_test, y_test = X_test.numpy(), y_test.numpy()

df_train, _, df_test = load_splits(filepaths=filepaths)

# %% Create directories to save results 

model_name = 'baseline'
save_checkpoint_dir = make_checkpoint_dir(model_name)

# %% Decompose into trend, seasonality and residuals (NOTE: Time period of test data not long enough to get yearly seasonality)
from statsmodels.tsa.seasonal import seasonal_decompose
series = df_train.set_index('utc_timestamp')[targets_column_names[0]]

# additive model
decomposition_additive = seasonal_decompose(series, model = 'additive', period = 24*7)

# %% Plot decomposition on training data

fig = decomposition_additive.plot()
fig.suptitle('Decomposition of Day Ahead Price using Additive Model', fontsize=14)
ax = fig.axes[-1]
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax = fig.axes[0]
ax.set_ylabel('Price (€/MWh)')


# %% Multiple Seasonal-Trend Decomposition
season_lengths = [24, 24 * 7] # one day and seven days
models = [MSTL(season_length=season_lengths, trend_forecaster = AutoARIMA())]
sf = StatsForecast(models = models, freq = 'h')

# %% Reorganize the train

df_train_target = df_train[['utc_timestamp', 'DE_price_ahead']]
df_train_target = df_train_target.rename(columns={'utc_timestamp': 'ds', 'DE_price_ahead': 'y'})
df_train_target.insert(0, 'unique_id', 'DE_price_ahead')
#df_train_target = df_train_target.sort_values(['unique_id', 'ds']).reset_index(drop=True)

df_test_target = df_test[['utc_timestamp', 'DE_price_ahead']]
df_test_target = df_test_target.rename(columns = {'utc_timestamp': 'ds', 'DE_price_ahead': 'y'})
df_test_target.insert(0, 'unique_id', 'DE_price_ahead')

# merge
df_all = pd.concat([df_train_target, df_test_target])
df_all

# %%
idx_origin_prediction_start

# %% Do rolling evaluation

# This is with expanding window. Can be done with sliding window
idx_origin_prediction_start = len(df_train_target) + 48
horizon = 24
for idx_end_input_window in range(idx_origin_prediction_start, idx_origin_prediction_start + 1): # len(df_all)):
    df_input = df_all.iloc[:idx_end_input_window]
    print(idx_end_input_window, len(df_input))
    sf.fit(df = df_input)
    df_seasonal_forecast = sf.predict(h=horizon)

    df_to_predict = df_all.iloc[idx_end_input_window:idx_end_input_window + horizon]

    fig, axes = plt.subplots()

    axes.plot(df_to_predict['ds'][:horizon], df_to_predict['y'][:horizon])
    axes.plot(df_seasonal_forecast['ds'][:horizon], df_seasonal_forecast['MSTL'][:horizon])
    
    axes.grid()
    axes.tick_params(axis='x', labelrotation = 45)
    #axes.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    # axes.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    fig.autofmt_xdate()

# %%
sf.fitted_[0, 0].model_.tail(24 * 14).plot(subplots=True, grid=True)
plt.tight_layout()
plt.show()

# %%
len(df_test)


# %%
cv_df = sf.cross_validation(df=df_all[:idx_origin_prediction_start + horizon], h = horizon, step_size = 24, refit = False, n_windows=1)
cv_df

# %%

plt.plot(cv_df['y'])
plt.plot(cv_df['MSTL'])

# %%
sf = sf.fit(df=df_train_target)
sf

# %%

sf.fitted_[0, 0].model_.tail(24 * 28).plot(subplots=True, grid=True)
plt.tight_layout()
plt.show()

# %% Seasonal forecast

horizon = 24 * 14 # len(df_test)

df_seasonal_forecast = sf.predict(h = horizon, level = [90])
df_seasonal_forecast

# %%
plt.plot(sf.fitted_[0, 0].model_['trend'])


# %%


# %%

fig, axes = plt.subplots()

axes.plot(df_seasonal_forecast['ds'][:horizon], df_seasonal_forecast['MSTL'][:horizon])
axes.plot(df_test['utc_timestamp'][:horizon], df_test['DE_price_ahead'][:horizon])
axes.grid()

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
