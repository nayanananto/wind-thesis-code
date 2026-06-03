from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.models.semantic_forecaster.semantic_input_builder import (
    SEMANTIC_META_COLUMNS,
    SemanticSampleBundle,
    build_semantic_training_samples,
)
from app.semantic.encoders.lstm_window_encoder import LSTMWindowEncoder
from app.semantic.encoders.statistical_encoder import StatisticalWindowEncoder
from app.semantic.tokenization.cluster_tokenizer import ClusterTokenizer

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
except Exception:
    tf = None


def _token_one_hot(labels: np.ndarray, n_tokens: int) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    matrix = np.zeros((len(labels), n_tokens), dtype=float)
    if len(labels):
        matrix[np.arange(len(labels)), labels] = 1.0
    columns = [f"token_{idx:02d}" for idx in range(n_tokens)]
    return pd.DataFrame(matrix, columns=columns)


def _build_semantic_state_frame(
    feature_frame: pd.DataFrame,
    windows: list,
    encoder: StatisticalWindowEncoder | LSTMWindowEncoder,
    tokenizer: ClusterTokenizer,
    fit: bool,
) -> pd.DataFrame:
    if len(feature_frame) != len(windows):
        raise ValueError("Semantic state construction requires aligned feature rows and windows.")

    if isinstance(encoder, StatisticalWindowEncoder):
        numeric_frame = feature_frame.drop(columns=SEMANTIC_META_COLUMNS, errors="ignore")
        if fit:
            embeddings = encoder.fit_transform(numeric_frame)
            tokens = tokenizer.fit_transform(embeddings)
        else:
            embeddings = encoder.transform(numeric_frame)
            tokens = tokenizer.transform(embeddings)
    else:
        if fit:
            embeddings = encoder.fit_transform(windows)
            tokens = tokenizer.fit_transform(embeddings)
        else:
            embeddings = encoder.transform(windows)
            tokens = tokenizer.transform(embeddings)

    embedding_columns = [f"embedding_{idx:02d}" for idx in range(embeddings.shape[1])]
    n_tokens = int(tokenizer.model.n_clusters) if tokenizer.model is not None else int(tokens["token_id"].nunique())
    token_one_hot = _token_one_hot(tokens["token_id"].to_numpy(), n_tokens)

    return pd.concat(
        [
            feature_frame[["window_id"]].reset_index(drop=True),
            pd.DataFrame(embeddings, columns=embedding_columns),
            token_one_hot.reset_index(drop=True),
            tokens[["token_distance"]].reset_index(drop=True),
        ],
        axis=1,
    )


def _make_semantic_sequences(
    state_matrix: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for idx in range(sequence_length - 1, len(state_matrix)):
        x_rows.append(state_matrix[idx - sequence_length + 1 : idx + 1])
        y_rows.append(targets[idx])

    if not x_rows:
        return np.empty((0, sequence_length, state_matrix.shape[1])), np.empty((0, targets.shape[1]))

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


@dataclass
class SemanticLSTMForecaster:
    window_size: int = 48
    step_size: int = 1
    n_components: int = 8
    n_clusters: int = 16
    encoder_type: str = "statistical"
    encoder_units: int = 32
    encoder_epochs: int = 10
    encoder_batch_size: int = 32
    encoder_dropout: float = 0.0
    sequence_length: int = 12
    units: int = 32
    epochs: int = 10
    batch_size: int = 16
    dropout: float = 0.2
    loss: str = "mse"
    random_state: int = 42
    encoder: StatisticalWindowEncoder | LSTMWindowEncoder = field(init=False)
    tokenizer: ClusterTokenizer = field(init=False)
    scaler: MinMaxScaler = field(init=False)
    model: Sequential | None = field(default=None, init=False)
    state_columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        encoder_key = self.encoder_type.lower().strip()
        if encoder_key in {"statistical", "pca"}:
            self.encoder = StatisticalWindowEncoder(n_components=self.n_components)
        elif encoder_key == "lstm":
            self.encoder = LSTMWindowEncoder(
                n_components=self.n_components,
                units=self.encoder_units,
                epochs=self.encoder_epochs,
                batch_size=self.encoder_batch_size,
                dropout=self.encoder_dropout,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported semantic encoder type '{self.encoder_type}'.")
        self.tokenizer = ClusterTokenizer(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        self.scaler = MinMaxScaler()

    def fit(self, bundle: SemanticSampleBundle) -> "SemanticLSTMForecaster":
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; semantic LSTM cannot run.")

        training_states = _build_semantic_state_frame(
            bundle.feature_frame,
            bundle.training_windows,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            fit=True,
        )
        state_matrix = training_states.drop(columns=["window_id"], errors="ignore").to_numpy(dtype=float)
        if len(state_matrix) < self.sequence_length:
            raise ValueError("Not enough semantic states for the requested sequence length.")

        self.state_columns = list(training_states.drop(columns=["window_id"], errors="ignore").columns)
        scaled_states = self.scaler.fit_transform(state_matrix)
        x_train, y_train = _make_semantic_sequences(
            scaled_states,
            bundle.targets,
            sequence_length=self.sequence_length,
        )
        if len(x_train) == 0:
            raise ValueError("No semantic LSTM sequences were created.")

        tf.keras.utils.set_random_seed(self.random_state)
        horizon = int(y_train.shape[1])
        keras_loss = tf.keras.losses.Huber() if self.loss.lower() == "huber" else "mse"

        model = Sequential(
            [
                Input(shape=(self.sequence_length, x_train.shape[2])),
                LSTM(self.units),
                Dropout(self.dropout),
                Dense(horizon),
            ]
        )
        model.compile(optimizer="adam", loss=keras_loss)
        early_stopping = EarlyStopping(
            monitor="loss",
            patience=2,
            min_delta=1e-4,
            restore_best_weights=True,
        )
        model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping],
        )
        self.model = model
        return self

    def predict(self, bundle: SemanticSampleBundle) -> pd.DataFrame:
        if self.model is None or not self.state_columns:
            raise ValueError("SemanticLSTMForecaster must be fitted before prediction.")

        all_features = bundle.all_feature_frame.copy()
        all_windows = list(bundle.all_windows)
        latest_window_id = str(bundle.latest_feature_row.iloc[0]["window_id"])
        if latest_window_id not in set(all_features["window_id"].astype(str)):
            all_features = pd.concat(
                [all_features, bundle.latest_feature_row],
                ignore_index=True,
            )
            all_windows.append(bundle.latest_window)

        all_states = _build_semantic_state_frame(
            all_features,
            all_windows,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            fit=False,
        )
        state_matrix = all_states.drop(columns=["window_id"], errors="ignore")
        state_matrix = state_matrix.reindex(columns=self.state_columns, fill_value=0.0)
        scaled_states = self.scaler.transform(state_matrix.to_numpy(dtype=float))
        if len(scaled_states) < self.sequence_length:
            raise ValueError("Not enough semantic states to create the prediction sequence.")

        latest_sequence = scaled_states[-self.sequence_length :].reshape(1, self.sequence_length, -1)
        prediction = self.model.predict(latest_sequence, verbose=0)[0]
        prediction = np.asarray(prediction, dtype=float).clip(min=0.0)

        return pd.DataFrame(
            {
                "datetime": bundle.future_index,
                "wind_speed": prediction,
            }
        )

    def fit_predict(self, train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        bundle = build_semantic_training_samples(
            train_df,
            horizon=horizon,
            window_size=self.window_size,
            step_size=self.step_size,
        )
        self.fit(bundle)
        return self.predict(bundle)


def semantic_lstm_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    window_size: int = 48,
    step_size: int = 1,
    n_components: int = 8,
    n_clusters: int = 16,
    encoder_type: str = "statistical",
    encoder_units: int = 32,
    encoder_epochs: int = 10,
    encoder_batch_size: int = 32,
    encoder_dropout: float = 0.0,
    sequence_length: int = 12,
    units: int = 32,
    epochs: int = 10,
    batch_size: int = 16,
    dropout: float = 0.2,
    loss: str = "mse",
    random_state: int = 42,
) -> pd.DataFrame:
    forecaster = SemanticLSTMForecaster(
        window_size=window_size,
        step_size=step_size,
        n_components=n_components,
        n_clusters=n_clusters,
        encoder_type=encoder_type,
        encoder_units=encoder_units,
        encoder_epochs=encoder_epochs,
        encoder_batch_size=encoder_batch_size,
        encoder_dropout=encoder_dropout,
        sequence_length=sequence_length,
        units=units,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        loss=loss,
        random_state=random_state,
    )
    return forecaster.fit_predict(train_df=train_df, horizon=horizon)
