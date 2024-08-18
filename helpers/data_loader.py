import numpy as np
from sklearn.preprocessing import StandardScaler

from helpers.prepocess_data import preprocess_df
import pandas as pd


def split_time_series_data(df, target_column, val_size=0.15, test_size=0.15, window_size=1):
    """
    Split a time series DataFrame into training, validation, and test sets.

    Parameters:
    - df: pandas DataFrame containing the time series data
    - target_column: string, name of the column to be predicted
    - val_size: float, proportion of data to use for validation (default 0.15)
    - test_size: float, proportion of data to use for testing (default 0.15)
    - window_size: int, size of the sliding window for feature creation (default 1)

    Returns:
    - x_train, y_train: Training data and labels as DataFrames
    - x_val, y_val: Validation data and labels as DataFrames
    - x_test, y_test: Test data and labels as DataFrames
    """

    # Preprocess the DataFrame
    df = preprocess_df(df)

    # Calculate sizes
    train_size = 1 - val_size - test_size
    total_size = len(df)
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    # Split the data
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    # Function to create features and labels
    def create_features(data):
        features = []
        labels = []
        for i in range(len(data) - window_size):
            features.append(data.iloc[i:i + window_size].drop(columns=[target_column]))
            labels.append(data.iloc[i + window_size][target_column])
        return pd.concat(features, keys=range(len(features))), pd.Series(labels, name=target_column)

    # Create features and labels for each set
    x_train, y_train = create_features(train_data)
    x_val, y_val = create_features(val_data)
    x_test, y_test = create_features(test_data)

    return x_train, y_train, x_val, y_val, x_test, y_test

def prepare_for_rnn(df, target_column, val_size=0.15, test_size=0.15, window_size=1):
    # Preprocess the DataFrame
    df = preprocess_df(df)

    # Calculate sizes
    train_size = 1 - val_size - test_size
    total_size = len(df)
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    # Split the data
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    # process the data
    def process_data(data):
        data = data[target_column].values.reshape(-1, 1)
        x_ = []
        y_ = []
        for i in range(window_size, len(data)):
            x_.append(data[i - window_size:i, 0])
            y_.append(data[i, 0])
        return np.array(x_), np.array(y_)

    # for training data (scaling and fittransform, the others will be transformed only)
    scaper = StandardScaler()
    x_train, y_train = process_data(train_data)
    x_train = scaper.fit_transform(x_train)

    # for validation data
    x_val, y_val = process_data(val_data)
    x_val = scaper.transform(x_val)

    # for test data
    x_test, y_test = process_data(test_data)
    x_test = scaper.transform(x_test)

    # reshape the input data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_val, y_val, x_test, y_test
