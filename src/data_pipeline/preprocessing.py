import pandas as pd

def clean_and_extract_data(features_column_names: list, targets_column_names: list, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """
    Extract columns that will make up features and columns that will make up targets from train, validation, and test dataframes

    Args:
        features_column_names: names of columns in dataframes that will make up features in training
        targets_column_names: names of columns in dataframes that will make up targets in training
        df_train: training subset
        df_val: validation subset
        df_test: test subset

    Returns:
        df_train: training data with NaNs dropped and feature and target columns extracted
        df_val: validation data --
        df_test: test data --
    
    """
    df_train = df_train[list(set(['utc_timestamp'] + features_column_names + targets_column_names))].dropna()
    df_val = df_val[list(set(['utc_timestamp'] + features_column_names + targets_column_names))].dropna()
    df_test = df_test[list(set(['utc_timestamp'] + features_column_names + targets_column_names))].dropna()

    return df_train, df_val, df_test


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

