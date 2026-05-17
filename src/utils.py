
import numpy as np


def create_sequences(features, targets, input_len: int = 48, horizon: int = 24):
    """Create sequences for seq2seq forecasting
    
    Args:
        features: data to use in forecast
        targets: data to forecast
        input_len: past timesteps to use in forecast (encoder window length)
        horizon: future timesteps to forecast (decoder window length)

    Returns:
        X: (n_samples, input_len, n_features) inputs to encoder
        y: (n_samples, horizon, n_targets) targets (labels) for decoder
    """

    n_samples = len(features)

    X = np.empty((n_samples - input_len - horizon, input_len, features.shape[-1]))
    y = np.empty((n_samples - input_len - horizon, horizon, targets.shape[-1]))

    for i_sample in range(n_samples - input_len - horizon):
        for i_feature in range(features.shape[-1]):
            X[i_sample,:,i_feature] = features[i_sample:(i_sample + input_len), i_feature]

        for i_feature in range(targets.shape[-1]):
            y[i_sample,:,i_feature] = targets[(i_sample + input_len):(i_sample + input_len + horizon), i_feature]

    return np.array(X), np.array(y)