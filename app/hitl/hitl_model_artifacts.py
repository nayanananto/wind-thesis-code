from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


NUMERIC_FEATURE_COLUMNS = [
    "wind_speed",
    "wind_gust_10m_ms",
    "u100",
    "v100",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

PHASE_FEATURE_CANDIDATES = [
    "token_distance",
    "wind_speed_mean",
    "wind_speed_std",
    "wind_speed_min",
    "wind_speed_max",
    "wind_speed_range",
    "ramp_abs_mean",
    "ramp_abs_max",
    "gust_factor",
    "direction_abs_change_mean_deg",
    "direction_net_turn_deg",
    "vector_resultant_strength",
    "hour_sin_mean",
    "hour_cos_mean",
    "dow_sin_mean",
    "dow_cos_mean",
]


def set_reproducible(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import Input, Model, Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Embedding, Concatenate
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("TensorFlow is required for HITL LSTM/GRU artifacts.") from exc
    return {
        "tf": tf,
        "Input": Input,
        "Model": Model,
        "Sequential": Sequential,
        "EarlyStopping": EarlyStopping,
        "GRU": GRU,
        "LSTM": LSTM,
        "Dense": Dense,
        "Dropout": Dropout,
        "Embedding": Embedding,
        "Concatenate": Concatenate,
        "Adam": Adam,
    }


def infer_time_step(frame: pd.DataFrame) -> pd.Timedelta:
    times = pd.to_datetime(frame["datetime"], errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return pd.Timedelta(hours=1)
    diffs = times.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    mode = diffs.mode()
    return pd.to_timedelta(mode.iloc[0] if not mode.empty else diffs.median())


def prepare_numeric_frame(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    data = frame.copy()
    if "observation_time_utc" in data.columns and "datetime" not in data.columns:
        data = data.rename(columns={"observation_time_utc": "datetime"})
    if "datetime" not in data.columns:
        raise ValueError("Numeric LSTM input needs datetime or observation_time_utc.")
    if "wind_speed" not in data.columns:
        raise ValueError("Numeric LSTM input needs wind_speed.")

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
    data["wind_speed"] = pd.to_numeric(data["wind_speed"], errors="coerce")
    data = data.dropna(subset=["datetime", "wind_speed"]).sort_values("datetime")
    data = data.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)

    if "wind_direction" in data.columns:
        data["wind_direction"] = pd.to_numeric(data["wind_direction"], errors="coerce")
        data["wind_direction"] = data["wind_direction"].ffill().bfill().fillna(0.0)
    else:
        data["wind_direction"] = 0.0

    if "wind_gust_10m_ms" in data.columns:
        data["wind_gust_10m_ms"] = pd.to_numeric(data["wind_gust_10m_ms"], errors="coerce")
    else:
        data["wind_gust_10m_ms"] = np.nan
    data["wind_gust_10m_ms"] = data["wind_gust_10m_ms"].fillna(data["wind_speed"])

    rad = np.deg2rad(data["wind_direction"].astype(float))
    data["u100"] = data["wind_speed"].astype(float) * np.cos(rad)
    data["v100"] = data["wind_speed"].astype(float) * np.sin(rad)

    dt = data["datetime"]
    seconds = dt.dt.hour.to_numpy() * 3600.0 + dt.dt.minute.to_numpy() * 60.0 + dt.dt.second.to_numpy()
    data["hour_sin"] = np.sin(2 * np.pi * seconds / 86400.0)
    data["hour_cos"] = np.cos(2 * np.pi * seconds / 86400.0)
    dow = dt.dt.dayofweek.to_numpy()
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    columns = ["datetime"] + list(feature_columns or NUMERIC_FEATURE_COLUMNS)
    for column in columns:
        if column == "datetime":
            continue
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[columns[1:]] = data[columns[1:]].ffill().bfill().fillna(0.0)
    return data[columns].reset_index(drop=True)


def _numeric_sequences(values: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    target_end_indices: list[int] = []
    for start in range(lookback, len(values) - horizon + 1):
        x_rows.append(values[start - lookback : start, :])
        y_rows.append(values[start : start + horizon, 0])
        target_end_indices.append(start + horizon - 1)
    return np.asarray(x_rows), np.asarray(y_rows), np.asarray(target_end_indices)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.where(denom == 0.0, 0.0, np.abs(err) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "smape": smape}


def _inverse_target(scaler: MinMaxScaler, y_scaled: np.ndarray, exog_scaled: np.ndarray | None = None) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=float)
    flat = y_scaled.reshape(-1)
    n_features = int(getattr(scaler, "n_features_in_", 1))
    dummy = np.zeros((len(flat), n_features), dtype=float)
    dummy[:, 0] = flat
    if n_features > 1 and exog_scaled is not None:
        exog = np.asarray(exog_scaled, dtype=float).reshape(len(flat), n_features - 1)
        dummy[:, 1:] = exog
    inv = scaler.inverse_transform(dummy)[:, 0]
    return inv.reshape(y_scaled.shape)


def _build_numeric_model(lookback: int, n_features: int, horizon: int, config: dict[str, Any]):
    keras = _require_tensorflow()
    model = keras["Sequential"]()
    model.add(keras["LSTM"](int(config["units"]), input_shape=(lookback, n_features)))
    dropout = float(config.get("dropout", 0.0))
    if dropout > 0:
        model.add(keras["Dropout"](dropout))
    model.add(keras["Dense"](horizon))
    loss_name = str(config.get("loss", "huber")).lower()
    loss = keras["tf"].keras.losses.Huber() if loss_name == "huber" else "mse"
    model.compile(optimizer=keras["Adam"](learning_rate=float(config.get("learning_rate", 0.001))), loss=loss)
    return model


def default_numeric_configs() -> list[dict[str, Any]]:
    return [
        {"lookback": 24, "units": 32, "dropout": 0.10, "batch_size": 16, "epochs": 20, "loss": "huber", "learning_rate": 0.001},
        {"lookback": 24, "units": 64, "dropout": 0.10, "batch_size": 16, "epochs": 20, "loss": "huber", "learning_rate": 0.001},
        {"lookback": 48, "units": 32, "dropout": 0.10, "batch_size": 16, "epochs": 20, "loss": "huber", "learning_rate": 0.001},
        {"lookback": 48, "units": 64, "dropout": 0.20, "batch_size": 16, "epochs": 20, "loss": "huber", "learning_rate": 0.001},
        {"lookback": 72, "units": 64, "dropout": 0.20, "batch_size": 32, "epochs": 20, "loss": "mse", "learning_rate": 0.001},
    ]


def train_tuned_numeric_lstm(
    data_path: str | Path,
    output_dir: str | Path,
    horizon: int = 6,
    configs: list[dict[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    set_reproducible(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(data_path) if str(data_path).lower().endswith(".parquet") else pd.read_csv(data_path)
    frame = prepare_numeric_frame(raw)
    values_raw = frame[NUMERIC_FEATURE_COLUMNS].astype(float).to_numpy()

    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    scaler = MinMaxScaler().fit(values_raw[:train_end])
    values_scaled = scaler.transform(values_raw)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for idx, config in enumerate(configs or default_numeric_configs(), start=1):
        lookback = int(config["lookback"])
        if n_rows <= lookback + horizon + 20:
            continue
        x_all, y_all, target_end = _numeric_sequences(values_scaled, lookback, horizon)
        train_mask = target_end < train_end
        val_mask = (target_end >= train_end) & (target_end < val_end)
        if train_mask.sum() < 20 or val_mask.sum() < 5:
            continue

        model = _build_numeric_model(lookback, len(NUMERIC_FEATURE_COLUMNS), horizon, config)
        keras = _require_tensorflow()
        callbacks = [
            keras["EarlyStopping"](
                monitor="val_loss",
                patience=3,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            x_all[train_mask],
            y_all[train_mask],
            validation_data=(x_all[val_mask], y_all[val_mask]),
            epochs=int(config.get("epochs", 20)),
            batch_size=int(config.get("batch_size", 16)),
            verbose=0,
            shuffle=False,
            callbacks=callbacks,
        )

        val_pred_scaled = model.predict(x_all[val_mask], verbose=0)
        y_val = _inverse_target(scaler, y_all[val_mask])
        pred_val = _inverse_target(scaler, val_pred_scaled)
        metrics = _regression_metrics(y_val, pred_val)
        row = {
            "rank_candidate": idx,
            **config,
            "epochs_ran": int(len(history.history.get("loss", []))),
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
            "val_smape": metrics["smape"],
        }
        rows.append(row)
        if best_row is None or row["val_rmse"] < best_row["val_rmse"]:
            best_row = row

    if not rows or best_row is None:
        raise ValueError("No numeric LSTM configuration could be trained.")

    tuning = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    tuning.to_csv(output / "tuning_results.csv", index=False)

    best_config = {key: best_row[key] for key in ["lookback", "units", "dropout", "batch_size", "epochs", "loss", "learning_rate"]}
    lookback = int(best_config["lookback"])
    x_all, y_all, target_end = _numeric_sequences(values_scaled, lookback, horizon)
    train_val_mask = target_end < val_end
    test_mask = target_end >= val_end
    model = _build_numeric_model(lookback, len(NUMERIC_FEATURE_COLUMNS), horizon, best_config)
    keras = _require_tensorflow()
    callbacks = [
        keras["EarlyStopping"](
            monitor="val_loss",
            patience=4,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    model.fit(
        x_all[train_val_mask],
        y_all[train_val_mask],
        validation_split=0.12,
        epochs=int(best_config.get("epochs", 20)),
        batch_size=int(best_config.get("batch_size", 16)),
        verbose=0,
        shuffle=False,
        callbacks=callbacks,
    )

    test_pred_scaled = model.predict(x_all[test_mask], verbose=0)
    y_test = _inverse_target(scaler, y_all[test_mask])
    pred_test = _inverse_target(scaler, test_pred_scaled)
    test_metrics = _regression_metrics(y_test, pred_test)

    last_observed_scaled = x_all[test_mask][:, -1, 0].reshape(-1, 1)
    persistence_scaled = np.repeat(last_observed_scaled, horizon, axis=1)
    persistence = _inverse_target(scaler, persistence_scaled)
    persistence_metrics = _regression_metrics(y_test, persistence)
    skill = 1.0 - (test_metrics["rmse"] / persistence_metrics["rmse"]) if persistence_metrics["rmse"] > 0 else None

    model.save(output / "model.keras")
    joblib.dump(scaler, output / "scaler.joblib")
    metadata = {
        "artifact_type": "hitl_numeric_lstm",
        "station": "KBOS",
        "data_path": str(data_path),
        "time_col": "datetime",
        "target_col": "wind_speed",
        "feature_columns": NUMERIC_FEATURE_COLUMNS,
        "horizon": int(horizon),
        "lookback": int(lookback),
        "time_step": str(infer_time_step(frame)),
        "best_config": _json_safe(best_config),
        "best_validation": _json_safe(best_row),
        "test_metrics": _json_safe(test_metrics),
        "persistence_metrics": _json_safe(persistence_metrics),
        "skill_vs_persistence_rmse": None if skill is None else float(skill),
        "model_path": str(output / "model.keras"),
        "scaler_path": str(output / "scaler.joblib"),
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def _predicted_window_sketch(observed: np.ndarray, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.array([float(row["wind_speed"]) for row in predictions], dtype=float)
    values = np.concatenate([observed[-min(6, len(observed)) :], predicted])
    diffs = np.diff(values) if len(values) > 1 else np.array([0.0])
    return {
        "evidence_id": "predicted_numeric_window",
        "construction": "recent_observed_raw_values_plus_numeric_lstm_forecast",
        "observed_points_used": int(min(6, len(observed))),
        "forecast_points_used": int(len(predictions)),
        "window_rows": int(len(values)),
        "wind_speed_mean": round(float(np.mean(values)), 6),
        "wind_speed_std": round(float(np.std(values)), 6),
        "wind_speed_min": round(float(np.min(values)), 6),
        "wind_speed_max": round(float(np.max(values)), 6),
        "wind_speed_p10": round(float(np.quantile(values, 0.10)), 6),
        "wind_speed_p90": round(float(np.quantile(values, 0.90)), 6),
        "wind_speed_range": round(float(np.max(values) - np.min(values)), 6),
        "ramp_abs_mean": round(float(np.mean(np.abs(diffs))), 6),
        "ramp_abs_max": round(float(np.max(np.abs(diffs))), 6),
    }


def predict_numeric_lstm(frame: pd.DataFrame, artifact_dir: str | Path, steps: int) -> dict[str, Any]:
    keras = _require_tensorflow()
    artifact = Path(artifact_dir)
    metadata = _read_json(artifact / "metadata.json")
    horizon = int(metadata["horizon"])
    lookback = int(metadata["lookback"])
    steps = int(steps)
    if steps > horizon:
        raise ValueError(f"HITL numeric LSTM was trained for horizon={horizon}, requested steps={steps}.")

    feature_columns = list(metadata.get("feature_columns") or NUMERIC_FEATURE_COLUMNS)
    prepared = prepare_numeric_frame(frame, feature_columns=feature_columns)
    if len(prepared) < lookback:
        raise ValueError(f"Need at least {lookback} live rows for LSTM forecast; got {len(prepared)}.")

    scaler = joblib.load(artifact / "scaler.joblib")
    model = keras["tf"].keras.models.load_model(artifact / "model.keras")
    values = prepared[feature_columns].astype(float).to_numpy()
    scaled = scaler.transform(values)
    x_input = scaled[-lookback:, :].reshape(1, lookback, len(feature_columns))
    pred_scaled = model.predict(x_input, verbose=0)[0][:steps]

    n_features = len(feature_columns)
    dummy = np.zeros((steps, n_features), dtype=float)
    dummy[:, 0] = pred_scaled
    if n_features > 1:
        dummy[:, 1:] = np.repeat(scaled[-1, 1:].reshape(1, -1), steps, axis=0)
    pred = scaler.inverse_transform(dummy)[:, 0].clip(min=0.0)

    time_step = infer_time_step(prepared)
    last_time = pd.Timestamp(prepared["datetime"].iloc[-1])
    recent_residuals = np.diff(values[-min(24, len(values)) :, 0]) if len(values) > 1 else np.array([0.0])
    residual_std = float(np.std(recent_residuals)) if len(recent_residuals) else 0.0
    predictions: list[dict[str, Any]] = []
    for idx, value in enumerate(pred, start=1):
        uncertainty = max(0.35, residual_std * (idx**0.5))
        timestamp = last_time + (time_step * idx)
        predictions.append(
            {
                "step": idx,
                "datetime": timestamp.isoformat(),
                "wind_speed": round(float(value), 3),
                "lower": round(max(0.0, float(value - uncertainty)), 3),
                "upper": round(float(value + uncertainty), 3),
                "uncertainty": round(float(uncertainty), 3),
            }
        )

    context_columns = [column for column in ["datetime", "wind_speed", "wind_gust_10m_ms"] if column in prepared.columns]
    recent_context = prepared.tail(min(8, len(prepared)))[context_columns].copy()
    recent_context["datetime"] = recent_context["datetime"].astype(str)
    latest = prepared.iloc[-1].to_dict()
    return {
        "method": "hitl_numeric_lstm",
        "forecast_model_source": "hitl_numeric_lstm",
        "model_artifact_path": str(artifact),
        "steps": int(steps),
        "source_rows": int(len(prepared)),
        "recent_rows_used": int(lookback),
        "feature_columns": feature_columns,
        "last_observation": {
            "datetime": str(latest.get("datetime")),
            "wind_speed": round(float(latest.get("wind_speed")), 3),
        },
        "predictions": predictions,
        "predicted_window_sketch": _predicted_window_sketch(values[:, 0], predictions),
        "recent_context": recent_context.to_dict(orient="records"),
        "grounding_rules": [
            "This numeric forecast uses a trained hourly LSTM artifact.",
            "Future exogenous variables are held at the latest observed value for live inference.",
            "Prediction intervals are heuristic bands from recent one-step variation.",
        ],
    }


def _phase_feature_columns(frame: pd.DataFrame) -> list[str]:
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    selected = [column for column in PHASE_FEATURE_CANDIDATES if column in frame.columns]
    return selected + embedding_columns


def _load_semantic_states(metadata_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    from app.hitl.semantic_review import load_semantic_metadata

    metadata = load_semantic_metadata(metadata_path)
    path = Path(metadata["output_paths"]["semantic_states"])
    if not path.exists():
        matches = list(Path.cwd().glob(f"**/{path.name}"))
        if matches:
            path = matches[0]
    states = pd.read_csv(path)
    if "window_start" in states.columns:
        states["window_start"] = pd.to_datetime(states["window_start"], errors="coerce")
        states = states.sort_values("window_start").reset_index(drop=True)
    states["token_id"] = pd.to_numeric(states["token_id"], errors="coerce")
    states = states.dropna(subset=["token_id"]).reset_index(drop=True)
    states["token_id"] = states["token_id"].astype(int)
    return metadata, states


def _phase_sequences(
    tokens: np.ndarray,
    features: np.ndarray,
    sequence_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tokens: list[np.ndarray] = []
    x_features: list[np.ndarray] = []
    y_rows: list[int] = []
    target_indices: list[int] = []
    for end_idx in range(sequence_length - 1, len(tokens) - horizon):
        start_idx = end_idx - sequence_length + 1
        target_idx = end_idx + horizon
        x_tokens.append(tokens[start_idx : end_idx + 1])
        x_features.append(features[start_idx : end_idx + 1])
        y_rows.append(int(tokens[target_idx]))
        target_indices.append(int(target_idx))
    return np.asarray(x_tokens), np.asarray(x_features), np.asarray(y_rows), np.asarray(target_indices)


def _build_phase_model(
    sequence_length: int,
    n_features: int,
    n_tokens: int,
    config: dict[str, Any],
):
    keras = _require_tensorflow()
    token_input = keras["Input"](shape=(sequence_length,), name="token_sequence")
    feature_input = keras["Input"](shape=(sequence_length, n_features), name="feature_sequence")
    token_embedding = keras["Embedding"](
        input_dim=n_tokens,
        output_dim=int(config["embedding_dim"]),
        name="token_embedding",
    )(token_input)
    merged = keras["Concatenate"](axis=-1)([token_embedding, feature_input])
    encoded = keras["GRU"](int(config["units"]))(merged)
    dropout = float(config.get("dropout", 0.0))
    if dropout > 0:
        encoded = keras["Dropout"](dropout)(encoded)
    output = keras["Dense"](n_tokens, activation="softmax")(encoded)
    model = keras["Model"](inputs=[token_input, feature_input], outputs=output)
    model.compile(
        optimizer=keras["Adam"](learning_rate=float(config.get("learning_rate", 0.001))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def default_phase_configs() -> list[dict[str, Any]]:
    return [
        {"sequence_length": 3, "embedding_dim": 8, "units": 24, "dropout": 0.10, "batch_size": 16, "epochs": 25, "learning_rate": 0.001},
        {"sequence_length": 4, "embedding_dim": 8, "units": 32, "dropout": 0.10, "batch_size": 16, "epochs": 25, "learning_rate": 0.001},
        {"sequence_length": 6, "embedding_dim": 8, "units": 32, "dropout": 0.15, "batch_size": 16, "epochs": 25, "learning_rate": 0.001},
        {"sequence_length": 6, "embedding_dim": 12, "units": 48, "dropout": 0.20, "batch_size": 32, "epochs": 25, "learning_rate": 0.001},
    ]


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, top_k: int = 3) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    ranked = np.argsort(-probabilities, axis=1)
    top1 = ranked[:, 0]
    topk = ranked[:, :top_k]
    reciprocal = []
    for actual, row in zip(y_true, ranked):
        matches = np.where(row == actual)[0]
        reciprocal.append(1.0 / float(matches[0] + 1) if len(matches) else 0.0)
    return {
        "top1_accuracy": round(float(np.mean(top1 == y_true)), 4),
        "topk_accuracy": round(float(np.mean([actual in row for actual, row in zip(y_true, topk)])), 4),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal)), 4),
        "macro_f1": round(float(f1_score(y_true, top1, average="macro", zero_division=0)), 4),
    }


def _transition_distribution(
    train_tokens: np.ndarray,
    sequence: np.ndarray,
    n_tokens: int,
    horizon: int,
    min_support: int,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=int)
    counts = np.zeros(n_tokens, dtype=float)
    history = len(sequence)
    for end_idx in range(history - 1, len(train_tokens) - horizon):
        start_idx = end_idx - history + 1
        if start_idx < 0:
            continue
        if np.array_equal(train_tokens[start_idx : end_idx + 1], sequence):
            counts[int(train_tokens[end_idx + horizon])] += 1
    if counts.sum() >= min_support:
        return counts / counts.sum()

    last = int(sequence[-1])
    markov = np.zeros(n_tokens, dtype=float)
    for end_idx in range(0, len(train_tokens) - horizon):
        if int(train_tokens[end_idx]) == last:
            markov[int(train_tokens[end_idx + horizon])] += 1
    if markov.sum() > 0:
        return markov / markov.sum()

    prior = np.bincount(train_tokens.astype(int), minlength=n_tokens).astype(float)
    return prior / prior.sum()


def train_tuned_phase_gru(
    metadata_path: str | Path,
    output_dir: str | Path,
    horizon: int = 1,
    configs: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    set_reproducible(seed)
    metadata, states = _load_semantic_states(metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens = states["token_id"].astype(int).to_numpy()
    n_tokens = int(max(tokens)) + 1
    feature_columns = _phase_feature_columns(states)
    feature_frame = states[feature_columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    n_rows = len(states)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    scaler = StandardScaler().fit(feature_frame.iloc[:train_end])
    features = scaler.transform(feature_frame)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for idx, config in enumerate(configs or default_phase_configs(), start=1):
        seq_len = int(config["sequence_length"])
        x_tok, x_feat, y_all, target_idx = _phase_sequences(tokens, features, seq_len, horizon)
        train_mask = target_idx < train_end
        val_mask = (target_idx >= train_end) & (target_idx < val_end)
        if train_mask.sum() < 20 or val_mask.sum() < 5:
            continue
        model = _build_phase_model(seq_len, len(feature_columns), n_tokens, config)
        keras = _require_tensorflow()
        callbacks = [
            keras["EarlyStopping"](
                monitor="val_loss",
                patience=4,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            [x_tok[train_mask], x_feat[train_mask]],
            y_all[train_mask],
            validation_data=([x_tok[val_mask], x_feat[val_mask]], y_all[val_mask]),
            epochs=int(config.get("epochs", 25)),
            batch_size=int(config.get("batch_size", 16)),
            verbose=0,
            shuffle=False,
            callbacks=callbacks,
        )
        probs = model.predict([x_tok[val_mask], x_feat[val_mask]], verbose=0)
        metrics = _classification_metrics(y_all[val_mask], probs, top_k=top_k)
        row = {
            "rank_candidate": idx,
            **config,
            "epochs_ran": int(len(history.history.get("loss", []))),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)
        if best_row is None or row["val_mean_reciprocal_rank"] > best_row["val_mean_reciprocal_rank"]:
            best_row = row

    if not rows or best_row is None:
        raise ValueError("No phase GRU configuration could be trained.")

    pd.DataFrame(rows).sort_values("val_mean_reciprocal_rank", ascending=False).to_csv(
        output / "tuning_results.csv",
        index=False,
    )

    best_config = {
        key: best_row[key]
        for key in ["sequence_length", "embedding_dim", "units", "dropout", "batch_size", "epochs", "learning_rate"]
    }
    seq_len = int(best_config["sequence_length"])
    x_tok, x_feat, y_all, target_idx = _phase_sequences(tokens, features, seq_len, horizon)
    train_val_mask = target_idx < val_end
    test_mask = target_idx >= val_end
    model = _build_phase_model(seq_len, len(feature_columns), n_tokens, best_config)
    keras = _require_tensorflow()
    callbacks = [
        keras["EarlyStopping"](
            monitor="val_loss",
            patience=4,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    model.fit(
        [x_tok[train_val_mask], x_feat[train_val_mask]],
        y_all[train_val_mask],
        validation_split=0.12,
        epochs=int(best_config.get("epochs", 25)),
        batch_size=int(best_config.get("batch_size", 16)),
        verbose=0,
        shuffle=False,
        callbacks=callbacks,
    )

    test_probs = model.predict([x_tok[test_mask], x_feat[test_mask]], verbose=0)
    test_metrics = _classification_metrics(y_all[test_mask], test_probs, top_k=top_k)
    persistence_probs = np.zeros_like(test_probs)
    last_tokens = x_tok[test_mask][:, -1].astype(int)
    persistence_probs[np.arange(len(last_tokens)), last_tokens] = 1.0
    persistence_metrics = _classification_metrics(y_all[test_mask], persistence_probs, top_k=top_k)

    train_tokens = tokens[:val_end]
    transition_probs = np.asarray(
        [
            _transition_distribution(train_tokens, sequence, n_tokens=n_tokens, horizon=horizon, min_support=5)
            for sequence in x_tok[test_mask]
        ]
    )
    transition_metrics = _classification_metrics(y_all[test_mask], transition_probs, top_k=top_k)

    model.save(output / "model.keras")
    joblib.dump(scaler, output / "feature_scaler.joblib")
    token_labels = _token_labels(states)
    artifact_metadata = {
        "artifact_type": "hitl_phase_gru",
        "station": "KBOS",
        "semantic_metadata_path": str(metadata_path),
        "semantic_run_name": metadata.get("run_name"),
        "horizon": int(horizon),
        "sequence_length": int(seq_len),
        "n_tokens": int(n_tokens),
        "feature_columns": feature_columns,
        "best_config": _json_safe(best_config),
        "best_validation": _json_safe(best_row),
        "test_metrics": _json_safe(test_metrics),
        "persistence_metrics": _json_safe(persistence_metrics),
        "transition_metrics": _json_safe(transition_metrics),
        "token_labels": token_labels,
        "model_path": str(output / "model.keras"),
        "feature_scaler_path": str(output / "feature_scaler.joblib"),
    }
    _write_json(output / "metadata.json", artifact_metadata)
    return artifact_metadata


def _token_labels(states: pd.DataFrame) -> dict[str, str]:
    if "regime_name" not in states.columns:
        return {str(token): f"token {token}" for token in sorted(states["token_id"].astype(int).unique())}
    labels: dict[str, str] = {}
    for token, group in states.groupby("token_id"):
        names = group["regime_name"].dropna().astype(str)
        labels[str(int(token))] = str(names.mode().iloc[0]) if not names.empty else f"token {int(token)}"
    return labels


def predict_phase_gru(
    live_history: pd.DataFrame,
    artifact_dir: str | Path,
    top_k: int = 3,
) -> dict[str, Any]:
    keras = _require_tensorflow()
    artifact = Path(artifact_dir)
    metadata = _read_json(artifact / "metadata.json")
    seq_len = int(metadata["sequence_length"])
    n_tokens = int(metadata["n_tokens"])
    feature_columns = list(metadata["feature_columns"])

    frame = live_history.copy()
    if "window_end" in frame.columns:
        frame["window_end"] = pd.to_datetime(frame["window_end"], errors="coerce", utc=True)
        frame = frame.sort_values("window_end")
    frame["token_id"] = pd.to_numeric(frame["token_id"], errors="coerce")
    frame = frame.dropna(subset=["token_id"]).reset_index(drop=True)
    if len(frame) < seq_len:
        raise ValueError(f"Need at least {seq_len} live semantic windows for phase GRU; got {len(frame)}.")

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    features = frame[feature_columns].ffill().bfill().fillna(0.0)
    scaler = joblib.load(artifact / "feature_scaler.joblib")
    scaled_features = scaler.transform(features)
    tokens = frame["token_id"].astype(int).to_numpy()
    if int(tokens.max()) >= n_tokens:
        raise ValueError("Live token id exceeds phase GRU token vocabulary.")

    x_tok = tokens[-seq_len:].reshape(1, seq_len)
    x_feat = scaled_features[-seq_len:, :].reshape(1, seq_len, len(feature_columns))
    model = keras["tf"].keras.models.load_model(artifact / "model.keras")
    probs = model.predict([x_tok, x_feat], verbose=0)[0]
    ranked = np.argsort(-probs)[: max(1, int(top_k))]
    labels = {int(key): value for key, value in (metadata.get("token_labels") or {}).items()}
    candidates = []
    for rank, token_id in enumerate(ranked, start=1):
        candidates.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "regime_name": labels.get(int(token_id), f"token {int(token_id)}"),
                "probability": round(float(probs[token_id]), 4),
                "count": None,
                "support": None,
                "transition_source": "hitl_phase_gru",
            }
        )

    latest = frame.iloc[-1].to_dict()
    return {
        "forecast_model_source": "hitl_phase_gru",
        "model_artifact_path": str(artifact),
        "history_length": int(seq_len),
        "horizon_steps": int(metadata.get("horizon", 1)),
        "live_token_sequence": [int(token) for token in tokens[-seq_len:].tolist()],
        "current_live_state": {
            key: latest.get(key)
            for key in [
                "window_id",
                "window_start",
                "window_end",
                "window_rows",
                "time_span_hours",
                "token_id",
                "token_distance",
                "regime_name",
            ]
            if key in latest
        },
        "candidate_next_phases": candidates,
        "support": None,
        "support_is_sufficient": True,
        "similar_transition_analogs": [],
        "grounding_rules": [
            "Live phase prediction uses a trained hourly GRU artifact over semantic token history.",
            "Probabilities are softmax scores over known semantic tokens.",
            "This model was selected offline against last-token persistence and transition-count baselines.",
        ],
    }
