import pandas as pd

def clean_data(df: pd.DataFrame, features_column_names: list, targets_column_names: list):
    """
    Extract and clean columns that will make up features targets

    Args:
        df: dataset prior to extracting feature and target columns
        features_column_names: names of columns in dataframes that will make up features in training
        targets_column_names: names of columns in dataframes that will make up targets in training
    Returns:
        df_clean: dataset with NaNs dropped and feature and target columns extracted
    """
    df_clean = df[list(dict.fromkeys(['utc_timestamp'] + features_column_names + targets_column_names))].dropna()

    return df_clean


def scale_features_and_targets(df_train, df, features_column_names, targets_column_names):
    """
    Standardize features and targets

    Args:
        df_train: training subset
        features_column_names: names of columns in dataframes that will make up features in training
        targets_column_names: names of columns in dataframes that will make up targets in training
        df: subset for which the features and targets are scaled

    Returns:
        features: standardized features
        targets: standardized targets
    """

    # OPEN QUESTION: Should I use different scaling than standardization with mean?

    features_mean = df_train[features_column_names].mean(axis=0).values
    targets_mean = df_train[targets_column_names].mean(axis=0).values

    features_std = df_train[features_column_names].std(axis=0).values
    targets_std = df_train[targets_column_names].std(axis=0).values

    features = ((df[features_column_names].values - features_mean)/features_std)
    targets = ((df[targets_column_names].values - targets_mean)/targets_std)
    return features, targets