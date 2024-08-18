import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from keras.api.callbacks import EarlyStopping, ModelCheckpoint


def train_and_evaluate_model(
        model,
        model_name,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        window_size,
        is_deep_learning=False,
        epochs=100,
        batch_size=32,
        hyperparams=None):
    """
    Train the model, optionally perform hyperparameter tuning, and evaluate it.

    Parameters:
    - model: The machine learning model to be trained
    - model_name: Name of the model
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - X_test, y_test: Testing data
    - window_size: Size of the sliding window used
    - is_deep_learning: Boolean indicating if it's a deep learning model
    - epochs, batch_size: Parameters for deep learning models
    - hyperparams: Dictionary of hyperparameters for traditional ML model tuning

    Returns:
    - model: The trained model
    - metrics: Dictionary containing evaluation metrics
    """
    start_time = time.time()

    if is_deep_learning:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(f"experiment_models/{model_name}_best_model.keras", save_best_only=True)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint])
    elif hyperparams is not None:


        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, hyperparams, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_

    else:
        model.fit(x_train, y_train)

    training_time = time.time() - start_time

    # Make predictions
    y_pred = model.predict(x_test)

    # Ensure y_pred is 1D
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)))
    r2 = r2_score(y_test, y_pred)
    forecast_bias = np.mean(y_pred - y_test)

    # Compile metrics dictionary
    metrics = {
        'model_name': model_name,
        'window_size': window_size,
        'rmse': rmse,
        'mae': mae,
        'smape': smape,
        'r2': r2,
        'forecast_bias': forecast_bias,
        'training_time': training_time
    }
    return model, metrics
