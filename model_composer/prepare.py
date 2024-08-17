import numpy as np


def prepare_for_model(
        model_name, model_func,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        window_size=7):
    """
    Prepare the model and data for training and evaluation. completely manual

    Parameters:
    - model_name: Name of the model
    - model_func: Function to create the model
    - x_train, y_train: Training data
    - x_val, y_val: Validation data
    - x_test, y_test: Testing data
    - window_size: Size of the sliding window used

    Returns:
    - model: The model
    - x_train, y_train: Training data and labels
    - x_val, y_val: Validation data and labels
    - x_test, y_test: Testing data and labels
    """

    feature_names = x_train.columns.tolist()
    model = None  # Placeholder override by the model_func

    if 'lstm' in model_name:
        input_shape = (window_size, len(feature_names))
        print(f"Input shape for {model_name}: {input_shape}")

        # Convert DataFrames to numpy arrays and reshape for RNN models
        x_train = x_train.values.reshape(-1, window_size, len(feature_names)).astype(np.float32)
        x_val = x_val.values.reshape(-1, window_size, len(feature_names)).astype(np.float32)
        x_test = x_test.values.reshape(-1, window_size, len(feature_names)).astype(np.float32)

        model = model_func(input_shape)

    elif 'cnn' in model_name:
        input_shape = (window_size, len(feature_names), 1)
        print(f"Input shape for {model_name}: {input_shape}")

        # Convert DataFrames to numpy arrays and reshape for CNN models
        x_train = x_train.values.reshape(-1, window_size, len(feature_names), 1).astype(np.float32)
        x_val = x_val.values.reshape(-1, window_size, len(feature_names), 1).astype(np.float32)
        x_test = x_test.values.reshape(-1, window_size, len(feature_names), 1).astype(np.float32)

        model = model_func(input_shape)

    return model, x_train, y_train, x_val, y_val, x_test, y_test
