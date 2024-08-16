import keras.api.backend as kb
import lightgbm as lgb
import xgboost as xgb
from keras.api.layers import SimpleRNN, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Activation, SpatialDropout1D, \
    LayerNormalization, Add, Input
from keras.api.models import Sequential, Model
from keras.api.optimizers import Adam


# ------------------ RNN Model ------------------
def create_rnn_model(input_shape, output_units=1):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        SimpleRNN(32),
        Dense(output_units)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# ------------------ LSTM Model ------------------
def create_lstm_model(input_shape, output_units=1):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(output_units)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# ------------------ CNN Model ------------------
def create_cnn_model(input_shape, output_units=1):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(output_units)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# ------------------ LightGBM Model ------------------
def create_lightgbm_model():
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    return lgb.LGBMRegressor(**params)


# ------------------ XGBoost Model ------------------
def create_xgboost_model():
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    return xgb.XGBRegressor(**params)


# ------------------ TCM Model ------------------
def causal_padding(x):
    return kb.temporal_padding(x, (1, 0))


def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.1):
    prev_x = x
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding='causal')(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding='causal')(x)
    x = Add()([prev_x, x])
    return x


def create_tcn_model(input_shape, output_units=1, nb_filters=64, kernel_size=2,
                     nb_stacks=1, dilations=[1, 2, 4, 8], dropout_rate=0.2):
    input_layer = Input(shape=input_shape)

    x = Conv1D(nb_filters, kernel_size, padding='causal', name='initial_conv')(input_layer)

    for _ in range(nb_stacks):
        for dilation_rate in dilations:
            x = residual_block(x, dilation_rate, nb_filters,
                               kernel_size, dropout_rate)

    x = Activation('relu')(x)
    x = Dense(output_units)(x)

    model = Model(inputs=[input_layer], outputs=[x])
    model.compile(optimizer='adam', loss='mse')

    return model
