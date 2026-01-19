import numpy as np
import pandas as pd
from tensorflow.keras import layers, models


class LSTMModels:
    """
    LSTM-based forecasting models
    """

    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length

    def build_sequences(self, df, feature_cols, target_col):
        X, y = [], []

        for i in range(self.sequence_length, len(df)):
            X.append(df[feature_cols].iloc[i-self.sequence_length:i].values)
            y.append(df[target_col].iloc[i])

        return np.array(X), np.array(y)

    def build_lstm(self, input_shape):
        model = models.Sequential([
            layers.LSTM(64, return_sequences=False, input_shape=input_shape),
            layers.Dense(1)
        ])

        model.compile(
            optimizer="adam",
            loss="mse"
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        model = self.build_lstm(X_train.shape[1:])

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            verbose=0
        )

        return model
