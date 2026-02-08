# lstm_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Optional


class LSTMResidualModel:
    """
    Узагальнена багатовимірна LSTM-модель.

    - вхід:  (window_size, num_features)
    - вихід: вектор довжини output_dim (наприклад, [beta, sigma, gamma])

    Використовується:
      * як чиста LSTM для прогнозу цільового ряду (output_dim = 1),
      * як генератор time-varying параметрів SEIR (output_dim = 3).
    """

    def __init__(
        self,
        window_size: int = 14,
        num_features: int = 3,
        lstm_units: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        output_dim: int = 1,
    ):
        self.window_size = window_size
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.output_dim = output_dim

        self.model: Optional[tf.keras.Model] = None

    def build_model(self):
        inputs = layers.Input(shape=(self.window_size, self.num_features))
        x = inputs

        # Проміжні LSTM-шари з return_sequences=True
        for _ in range(self.num_layers - 1):
            x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.Dropout(self.dropout)(x)

        # Останній LSTM-шар – повертає вектор ознак
        x = layers.LSTM(self.lstm_units)(x)
        x = layers.Dropout(self.dropout)(x)

        # Вихід – вектор параметрів (output_dim)
        outputs = layers.Dense(self.output_dim)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        self.model = model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0,
    ):
        if self.model is None:
            self.build_model()
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель LSTM не навчена / не побудована.")
        return self.model.predict(X)
