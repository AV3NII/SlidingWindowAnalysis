import numpy as np
from helpers.prepocess_data import preprocess_df


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
    - x_train, y_train: Training data and labels
    - x_val, y_val: Validation data and labels
    - x_test, y_test: Test data and labels
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
            features.append(data.iloc[i:i + window_size].drop(columns=[target_column]).values.flatten())
            labels.append(data.iloc[i + window_size][target_column])
        return np.array(features), np.array(labels)

    # Create features and labels for each set
    x_train, y_train = create_features(train_data)
    x_val, y_val = create_features(val_data)
    x_test, y_test = create_features(test_data)

    return x_train, y_train, x_val, y_val, x_test, y_test
