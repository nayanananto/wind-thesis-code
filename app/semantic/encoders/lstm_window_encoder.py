from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.semantic.windowing.window_builder import SemanticWindow

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
except Exception:
    tf = None


@dataclass
class LSTMWindowEncoder:
    n_components: int = 8
    units: int = 32
    epochs: int = 10
    batch_size: int = 32
    dropout: float = 0.0
    random_state: int = 42
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_columns: list[str] = field(default_factory=list)
    fill_values: dict[str, float] = field(default_factory=dict)
    window_size: int | None = None
    encoder_model: Model | None = field(default=None, init=False)
    autoencoder_model: Model | None = field(default=None, init=False)

    def _fit_feature_schema(self, windows: list[SemanticWindow]) -> None:
        candidate_columns = [c for c in windows[0].frame.columns if c != "datetime"]
        keep_columns: list[str] = []
        fill_values: dict[str, float] = {}

        for column in candidate_columns:
            series_parts: list[pd.Series] = []
            for window in windows:
                if column not in window.frame.columns:
                    continue
                numeric = pd.to_numeric(window.frame[column], errors="coerce")
                if numeric.notna().any():
                    series_parts.append(numeric)

            if not series_parts:
                continue

            combined = pd.concat(series_parts, ignore_index=True).dropna()
            if combined.empty:
                continue

            keep_columns.append(column)
            fill_values[column] = float(combined.median())

        if not keep_columns:
            raise ValueError("No numeric sequence columns were available for LSTM encoding.")

        self.feature_columns = keep_columns
        self.fill_values = fill_values
        self.window_size = int(len(windows[0].frame))

    def _window_matrix(self, window: SemanticWindow) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("Feature schema has not been initialized.")

        frame = window.frame.reindex(columns=["datetime"] + self.feature_columns, fill_value=np.nan).copy()
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(pd.Series(self.fill_values)).fillna(0.0)
        matrix = numeric.to_numpy(dtype=float)

        expected_rows = int(self.window_size or len(matrix))
        if len(matrix) > expected_rows:
            matrix = matrix[-expected_rows:]
        elif len(matrix) < expected_rows:
            pad = np.repeat(matrix[:1], expected_rows - len(matrix), axis=0) if len(matrix) else np.zeros((expected_rows, len(self.feature_columns)), dtype=float)
            matrix = np.vstack([pad, matrix])

        return matrix

    def _windows_to_tensor(self, windows: list[SemanticWindow], fitting: bool) -> np.ndarray:
        if not windows:
            raise ValueError("No windows were provided to the LSTM encoder.")

        if fitting:
            self._fit_feature_schema(windows)

        tensor = np.asarray([self._window_matrix(window) for window in windows], dtype=float)
        flattened = tensor.reshape(-1, tensor.shape[-1])

        if fitting:
            flattened = self.scaler.fit_transform(flattened)
        else:
            flattened = self.scaler.transform(flattened)

        return flattened.reshape(tensor.shape)

    def fit(self, windows: list[SemanticWindow]) -> "LSTMWindowEncoder":
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; LSTM window encoder cannot run.")

        tensor = self._windows_to_tensor(windows, fitting=True)
        tf.keras.utils.set_random_seed(self.random_state)

        time_steps = tensor.shape[1]
        feature_count = tensor.shape[2]
        latent_dim = max(1, int(self.n_components))
        hidden_units = max(latent_dim, int(self.units))

        inputs = Input(shape=(time_steps, feature_count))
        encoded = LSTM(hidden_units, dropout=self.dropout)(inputs)
        latent = Dense(latent_dim, name="latent_embedding")(encoded)
        decoded = RepeatVector(time_steps)(latent)
        decoded = LSTM(hidden_units, return_sequences=True, dropout=self.dropout)(decoded)
        outputs = TimeDistributed(Dense(feature_count))(decoded)

        autoencoder = Model(inputs=inputs, outputs=outputs)
        encoder = Model(inputs=inputs, outputs=latent)

        autoencoder.compile(optimizer="adam", loss="mse")
        early_stopping = EarlyStopping(
            monitor="loss",
            patience=2,
            min_delta=1e-4,
            restore_best_weights=True,
        )
        autoencoder.fit(
            tensor,
            tensor,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping],
        )

        self.autoencoder_model = autoencoder
        self.encoder_model = encoder
        return self

    def transform(self, windows: list[SemanticWindow]) -> np.ndarray:
        if self.encoder_model is None:
            raise ValueError("LSTMWindowEncoder has not been fitted yet.")

        tensor = self._windows_to_tensor(windows, fitting=False)
        embeddings = self.encoder_model.predict(tensor, verbose=0)
        return np.asarray(embeddings, dtype=float)

    def fit_transform(self, windows: list[SemanticWindow]) -> np.ndarray:
        self.fit(windows)
        return self.transform(windows)
