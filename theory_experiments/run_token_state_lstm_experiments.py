from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_wind import infer_time_step, metrics, seasonal_persistence, set_reproducible  # noqa: E402


try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover - handled at runtime
    tf = None
    EarlyStopping = Dense = Dropout = LSTM = Sequential = None


SEEDS = [42, 123]
HORIZON_HOURS = [1, 3, 6, 12]
STEPS_PER_HOUR = 12
HORIZON_STEPS = {hour: hour * STEPS_PER_HOUR for hour in HORIZON_HOURS}
TRAIN_DAYS = 30
STEP_HOURS = 168
FINAL_SPLITS = 2
SEQUENCE_LENGTH = 12
UNITS = 32
EPOCHS = 12
BATCH_SIZE = 32
DROPOUT = 0.15


STATIONS = {
    "KBOS": {
        "raw_path": ROOT / "data" / "noaa_5min" / "KBOS_2024_5min.parquet",
        "states_path": ROOT / "data" / "semantic" / "kbos_5min_phase_semantic_states.csv",
    },
    "DDC": {
        "raw_path": ROOT / "data" / "noaa_5min" / "DDC_2024_5min.parquet",
        "states_path": ROOT / "data" / "semantic" / "ddc_5min_phase_semantic_states.csv",
    },
}


OUT = ROOT / "results" / "token_state_lstm_experiments"


@dataclass
class StationData:
    station: str
    raw: pd.DataFrame
    states: pd.DataFrame
    feature_matrix: np.ndarray
    feature_columns: list[str]
    target: pd.Series
    time_step: pd.Timedelta


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_station(station: str, raw_path: Path, states_path: Path) -> StationData:
    raw = pd.read_parquet(raw_path)
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw = raw.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    raw["wind_speed"] = pd.to_numeric(raw["wind_speed"], errors="coerce")
    raw["wind_speed"] = raw["wind_speed"].interpolate(limit_direction="both").ffill().bfill()
    time_step = infer_time_step(raw)

    states = pd.read_csv(states_path)
    states["window_end"] = pd.to_datetime(states["window_end"], errors="coerce")
    states = states.dropna(subset=["window_end"]).sort_values("window_end").reset_index(drop=True)
    states["token_id"] = pd.to_numeric(states["token_id"], errors="coerce").astype(int)
    states["token_distance"] = pd.to_numeric(states["token_distance"], errors="coerce").fillna(0.0)

    n_tokens = int(states["token_id"].max()) + 1
    token = states["token_id"].to_numpy(dtype=int)
    one_hot = np.zeros((len(states), n_tokens), dtype=float)
    valid = (token >= 0) & (token < n_tokens)
    one_hot[np.arange(len(states))[valid], token[valid]] = 1.0
    distance = states["token_distance"].to_numpy(dtype=float).reshape(-1, 1)
    matrix = np.hstack([one_hot, distance])
    columns = [f"token_{idx}" for idx in range(n_tokens)] + ["token_distance"]

    target = raw.set_index("datetime").sort_index()["wind_speed"].astype(float)
    return StationData(station, raw, states, matrix, columns, target, time_step)


def make_cutoffs(raw: pd.DataFrame, horizon_steps: int) -> pd.DatetimeIndex:
    time_step = infer_time_step(raw)
    cut_start = raw["datetime"].min() + pd.Timedelta(days=TRAIN_DAYS)
    cut_end = raw["datetime"].max() - (time_step * int(horizon_steps))
    candidates = pd.date_range(cut_start, cut_end, freq=pd.Timedelta(hours=STEP_HOURS))
    data_by_time = raw.set_index("datetime").sort_index()
    valid: list[pd.Timestamp] = []
    for cutoff in candidates:
        fut_idx = pd.date_range(cutoff + time_step, periods=int(horizon_steps), freq=time_step)
        truth = data_by_time.reindex(fut_idx)["wind_speed"]
        if not truth.isna().any():
            valid.append(pd.Timestamp(cutoff))
    if not valid:
        raise ValueError("No valid rolling cutoffs.")
    return pd.DatetimeIndex(valid)


def select_even(cutoffs: pd.DatetimeIndex, count: int) -> pd.DatetimeIndex:
    if len(cutoffs) <= count:
        return cutoffs
    idx = np.linspace(0, len(cutoffs) - 1, num=count, dtype=int)
    return cutoffs[idx]


def build_training_data(
    data: StationData,
    cutoff: pd.Timestamp,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    state_times = pd.DatetimeIndex(data.states["window_end"])
    start_time = cutoff - pd.Timedelta(days=TRAIN_DAYS)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for end_idx in range(SEQUENCE_LENGTH - 1, len(state_times)):
        origin = state_times[end_idx]
        if origin < start_time:
            continue
        target_times = pd.date_range(origin + data.time_step, periods=horizon_steps, freq=data.time_step)
        if target_times[-1] > cutoff:
            continue
        y = data.target.reindex(target_times).to_numpy(dtype=float)
        if not np.isfinite(y).all():
            continue
        x_rows.append(data.feature_matrix[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1])
        y_rows.append(y)

    if not x_rows:
        raise ValueError(f"No token-state LSTM training rows before cutoff {cutoff}")

    X = np.asarray(x_rows, dtype=float)
    Y = np.asarray(y_rows, dtype=float)

    x_scaler = MinMaxScaler().fit(X.reshape(-1, X.shape[-1]))
    y_scaler = MinMaxScaler().fit(Y.reshape(-1, 1))
    Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    Ys = y_scaler.transform(Y.reshape(-1, 1)).reshape(Y.shape)
    return Xs, Ys, x_scaler, y_scaler


def build_model(n_features: int, horizon_steps: int) -> Sequential:
    if tf is None:
        raise RuntimeError("TensorFlow not installed; cannot run token-state LSTM")
    model = Sequential()
    model.add(LSTM(UNITS, input_shape=(SEQUENCE_LENGTH, n_features)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(horizon_steps))
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return model


def forecast_one(
    data: StationData,
    cutoff: pd.Timestamp,
    horizon_steps: int,
    seed: int,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    if tf is not None:
        tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    set_reproducible(seed)

    X, Y, x_scaler, y_scaler = build_training_data(data, cutoff, horizon_steps)
    state_times = pd.DatetimeIndex(data.states["window_end"])
    pos = state_times.searchsorted(cutoff, side="right") - 1
    if pos < SEQUENCE_LENGTH - 1:
        raise ValueError(f"Not enough token-state context before cutoff {cutoff}")
    X_forecast = data.feature_matrix[pos - SEQUENCE_LENGTH + 1 : pos + 1].reshape(
        1, SEQUENCE_LENGTH, -1
    )
    X_forecast = x_scaler.transform(X_forecast.reshape(-1, X_forecast.shape[-1])).reshape(X_forecast.shape)

    model = build_model(X.shape[-1], horizon_steps)
    es = EarlyStopping(monitor="loss", patience=2, min_delta=1e-4, restore_best_weights=True)
    history = model.fit(
        X,
        Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es],
        shuffle=False,
    )
    pred_scaled = model.predict(X_forecast, verbose=0)[0]
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred = np.clip(pred.astype(float), 0.0, None)
    future_idx = pd.date_range(cutoff + data.time_step, periods=horizon_steps, freq=data.time_step)
    fit_info = {
        "train_sequences": int(X.shape[0]),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "final_loss": float(history.history.get("loss", [np.nan])[-1]),
    }
    return pred, future_idx, fit_info


def evaluate_forecast(
    data: StationData,
    cutoff: pd.Timestamp,
    horizon_steps: int,
    pred: np.ndarray,
    future_idx: pd.DatetimeIndex,
) -> dict[str, float]:
    truth = data.target.reindex(future_idx).to_numpy(dtype=float)
    m = metrics(truth, pred)
    train_df = data.raw[
        (data.raw["datetime"] >= cutoff - pd.Timedelta(days=TRAIN_DAYS))
        & (data.raw["datetime"] <= cutoff)
    ].copy()
    base_pred = (
        seasonal_persistence(train_df, horizon_steps)
        .set_index("datetime")
        .reindex(future_idx)["wind_speed"]
        .to_numpy(dtype=float)
    )
    base_m = metrics(truth, base_pred)
    m["baseline_mae"] = base_m["mae"]
    m["skill_vs_persistence"] = 1.0 - (m["mae"] / (base_m["mae"] + 1e-9))
    return m


def run_station(station: str, paths: dict[str, Path]) -> pd.DataFrame:
    data = load_station(station, paths["raw_path"], paths["states_path"])
    station_out = OUT / station.lower()
    station_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    live_path = station_out / "token_state_lstm_live.csv"

    for seed in SEEDS:
        for horizon_hours in HORIZON_HOURS:
            horizon_steps = int(HORIZON_STEPS[horizon_hours])
            cutoffs = select_even(make_cutoffs(data.raw, horizon_steps), FINAL_SPLITS)
            start = time.time()
            split_metrics: list[dict[str, float]] = []
            status = "ok"
            error = ""
            fit_infos: list[dict[str, Any]] = []

            for split_id, cutoff in enumerate(cutoffs):
                try:
                    pred, future_idx, fit_info = forecast_one(data, cutoff, horizon_steps, seed)
                    m = evaluate_forecast(data, cutoff, horizon_steps, pred, future_idx)
                    split_row = {
                        "station": station,
                        "seed": int(seed),
                        "design": "token_state_lstm",
                        "horizon_hours": int(horizon_hours),
                        "horizon_steps": int(horizon_steps),
                        "split_id": int(split_id),
                        "cutoff": cutoff.isoformat(),
                        **m,
                        **fit_info,
                    }
                    split_rows.append(split_row)
                    split_metrics.append(m)
                    fit_infos.append(fit_info)
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    print(f"[token-lstm:error] {station} seed={seed} h={horizon_hours}: {exc}", flush=True)
                    break

            row: dict[str, Any] = {
                "station": station,
                "seed": int(seed),
                "design": "token_state_lstm",
                "horizon_hours": int(horizon_hours),
                "horizon_steps": int(horizon_steps),
                "status": status,
                "train_days": int(TRAIN_DAYS),
                "max_splits": int(FINAL_SPLITS),
                "sequence_length": int(SEQUENCE_LENGTH),
                "units": int(UNITS),
                "epochs": int(EPOCHS),
                "batch_size": int(BATCH_SIZE),
                "dropout": float(DROPOUT),
                "feature_columns": json.dumps(data.feature_columns),
                "elapsed_sec": round(time.time() - start, 2),
            }
            if split_metrics:
                for key in ["mae", "rmse", "smape", "nmae", "nrmse", "baseline_mae", "skill_vs_persistence"]:
                    row[key] = float(np.mean([m[key] for m in split_metrics]))
                row["valid_splits"] = int(len(split_metrics))
                row["mean_train_sequences"] = float(np.mean([fi["train_sequences"] for fi in fit_infos]))
                row["mean_epochs_ran"] = float(np.mean([fi["epochs_ran"] for fi in fit_infos]))
            if error:
                row["error"] = error

            rows.append(row)
            pd.DataFrame(rows).to_csv(live_path, index=False)
            print(
                f"[token-lstm:{status}] {station} seed={seed} h={horizon_hours}h "
                f"skill={row.get('skill_vs_persistence')} mae={row.get('mae')} "
                f"elapsed={row['elapsed_sec']}s",
                flush=True,
            )

    final = pd.DataFrame(rows)
    splits = pd.DataFrame(split_rows)
    final.to_csv(station_out / "token_state_lstm_results.csv", index=False)
    splits.to_csv(station_out / "token_state_lstm_split_metrics.csv", index=False)
    return final


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for station, paths in STATIONS.items():
        all_rows.append(run_station(station, paths))
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT / "token_state_lstm_all_results.csv", index=False)
    summary = (
        combined[combined["status"].eq("ok")]
        .groupby(["station", "design", "horizon_hours"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            skill_vs_persistence=("skill_vs_persistence", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(OUT / "token_state_lstm_summary.csv", index=False)
    metadata = {
        "design": "token_state_lstm",
        "purpose": "Continuous wind-speed forecast using regime-token state sequence as LSTM input.",
        "stations": list(STATIONS),
        "seeds": SEEDS,
        "horizon_hours": HORIZON_HOURS,
        "horizon_steps": HORIZON_STEPS,
        "train_days": TRAIN_DAYS,
        "final_splits": FINAL_SPLITS,
        "sequence_length": SEQUENCE_LENGTH,
        "units": UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "inputs": {
            "token_state": "recent sequence of one-hot regime token indicators plus token_distance",
            "target": "future wind_speed over every 5-minute lead step inside the evaluated horizon",
        },
    }
    (OUT / "token_state_lstm_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2),
        encoding="utf-8",
    )
    print("\n[summary]")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
