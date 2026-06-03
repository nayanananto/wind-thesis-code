# backtest_wind.py
from __future__ import annotations
import argparse, math, warnings
import numpy as np
import pandas as pd
from app.models.semantic_forecaster.semantic_lstm_service import semantic_lstm_forecast
from app.models.semantic_forecaster.semantic_token_service import semantic_token_forecast
# Optional: TensorFlow (guarded)
try:
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None  # we'll check later

# Optional deps (guarded imports)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

# ----------------- Config -----------------
EXOG_COLS_DEFAULT = [
    "u100", "v100", "wind_gust_10m_ms", "temperature_2m_c", "relative_humidity_2m",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]
# --- add imports at top ---
import os, random
import numpy as np

def set_reproducible(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # avoids oneDNN non-determinism on CPU
    try:
        import tensorflow as tf
        # TF 2.11+ helper (covers Python, NumPy, TF)
        tf.keras.utils.set_random_seed(seed)
        # Also force deterministic kernels where available
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

# ----------------- Utils ------------------
def infer_time_step(df: pd.DataFrame) -> pd.Timedelta:
    if "datetime" not in df.columns or len(df) < 2:
        return pd.Timedelta(hours=1)
    series = pd.to_datetime(df["datetime"], errors="coerce").dropna().sort_values()
    if len(series) < 2:
        return pd.Timedelta(hours=1)
    diffs = series.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    mode = diffs.mode()
    if not mode.empty:
        return pd.to_timedelta(mode.iloc[0])
    return pd.to_timedelta(diffs.median())


def load_features(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # --- NEW: cyclic time features if missing ---
    if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
        seconds = (
            df["datetime"].dt.hour.to_numpy() * 3600.0
            + df["datetime"].dt.minute.to_numpy() * 60.0
            + df["datetime"].dt.second.to_numpy()
        )
        df["hour_sin"] = np.sin(2 * np.pi * seconds / 86400.0)
        df["hour_cos"] = np.cos(2 * np.pi * seconds / 86400.0)
    if "dow_sin" not in df.columns or "dow_cos" not in df.columns:
        dow = df["datetime"].dt.dayofweek.to_numpy()
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # --- NEW: optional u/v from speed+direction if available and u100/v100 missing ---
    if "u100" not in df.columns and "v100" not in df.columns:
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            rad = np.deg2rad(df["wind_direction"].astype(float))
            df["u100"] = df["wind_speed"].astype(float) * np.cos(rad)
            df["v100"] = df["wind_speed"].astype(float) * np.sin(rad)

    return df

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    # basic errors
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))

    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(
        np.mean(
            np.where(denom == 0.0, 0.0, np.abs(y_true - y_pred) / denom)
        ) * 100.0
    )

    # ----- RANGE-BASED NORMALIZATION (main NMAE / NRMSE) -----
    data_range = float(np.max(y_true) - np.min(y_true))
    if data_range < 1e-9:
        nmae_range = float("nan")
        nrmse_range = float("nan")
    else:
        nmae_range = mae / data_range          # MAE / (max-min)
        nrmse_range = rmse / data_range        # RMSE / (max-min)

    # ----- STD-BASED NORMALIZATION (extra info) -----
    std = float(np.std(y_true))
    if std < 1e-9:
        nrmse_std = float("nan")
        nmse = float("nan")
    else:
        nrmse_std = rmse / std                 # RMSE / σ
        nmse = mse / (std ** 2)                # MSE / σ²

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        # main normalized metrics (range-based)
        "nmae": nmae_range,
        "nrmse": nrmse_range,
        # extra diagnostics
        "nrmse_std": nrmse_std,
        "nmse": nmse,
    }





def seasonal_persistence(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    base = train_df.set_index("datetime")["wind_speed"]
    time_step = infer_time_step(train_df)
    idx = pd.date_range(train_df["datetime"].max() + time_step, periods=int(horizon), freq=time_step)
    preds = [float(base.get(ts - pd.Timedelta(hours=24), base.iloc[-1])) for ts in idx]
    return pd.DataFrame({"datetime": idx, "wind_speed": preds})


def trim_history(df: pd.DataFrame, cutoff: pd.Timestamp, train_days: int) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=train_days)
    return df[(df["datetime"] >= start) & (df["datetime"] <= cutoff)].copy()

def build_exog(train_df: pd.DataFrame, fut_idx: pd.DatetimeIndex, exog_cols: list[str], mode: str) -> tuple[pd.DataFrame|None, pd.DataFrame|None]:
    """mode: 'actual' uses actual future exog, 'persistence' repeats last observed."""
    have = [c for c in exog_cols if c in train_df.columns]
    if not have:
        return None, None
    X_hist = train_df[["datetime"] + have].rename(columns={"datetime": "ds"}).copy()

    # future exog
    if mode == "actual":
        # we will merge from the global df later (caller provides)
        X_future = pd.DataFrame({"ds": fut_idx})
        # placeholder: caller should fill actual values before calling model
        return X_hist, X_future  # caller will fill columns
    else:
        # persistence from last row of train
        last = train_df.iloc[-1]
        X_future = pd.DataFrame({"ds": fut_idx})
        for c in have:
            X_future[c] = float(last[c])
        return X_hist, X_future

# -------------- Models --------------------
def prophet_forecast(train_df: pd.DataFrame,
                     fut_actual_exog_df: pd.DataFrame|None,
                     horizon: int,
                     exog_cols: list[str],
                     enforce_nonneg: bool = True,
                     daily_fourier: int = 12,
                     weekly_fourier: int = 6,
                     cps: float = 0.10,
                     n_changepoints: int = 10) -> pd.DataFrame:
    if Prophet is None:
        raise RuntimeError("prophet not installed")

    y = train_df[["datetime","wind_speed"]].rename(columns={"datetime":"ds","wind_speed":"y"}).copy()
    time_step = infer_time_step(train_df)
    fut_idx = pd.date_range(train_df["datetime"].max() + time_step,
                            periods=int(horizon), freq=time_step)
    X_hist, X_future = build_exog(train_df, fut_idx, exog_cols,
                                  mode="actual" if fut_actual_exog_df is not None else "persistence")

    if X_future is not None and fut_actual_exog_df is not None:
        fut_actual = fut_actual_exog_df.reindex(fut_idx)[exog_cols].reset_index().rename(columns={"index":"ds"})
        for c in [c for c in exog_cols if c in fut_actual.columns]:
            X_future[c] = fut_actual[c].values

    m = Prophet(
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
        n_changepoints=n_changepoints, changepoint_prior_scale=cps,
        seasonality_mode="additive", uncertainty_samples=0
    )
    m.add_seasonality(name="daily", period=1, fourier_order=daily_fourier)
    m.add_seasonality(name="weekly", period=7,  fourier_order=weekly_fourier)

    if X_hist is not None:
        for c in [col for col in X_hist.columns if col != "ds"]:
            m.add_regressor(c, standardize=True)
        train = y.merge(X_hist, on="ds", how="left")
    else:
        train = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train)

    future = pd.DataFrame({"ds": fut_idx})
    if X_future is not None:
        future = future.merge(X_future, on="ds", how="left")

    fc = m.predict(future)
    out = fc[["ds","yhat"]].rename(columns={"ds":"datetime","yhat":"wind_speed"})
    if enforce_nonneg:
        out["wind_speed"] = out["wind_speed"].clip(lower=0.0)
    return out

def lstm_forecast(train_df: pd.DataFrame,
                  fut_actual_exog_df: pd.DataFrame|None,
                  horizon: int,
                  exog_cols: list[str],
                  lookback: int = 48,
                  units: int = 64,
                  epochs: int = 15,
                  batch_size: int = 32,
                  dropout: float = 0.2,
                  loss: str = "mse",          # "mse" or "huber"
                  direct: bool = False        # NEW: True = direct multi-horizon
                  ) -> pd.DataFrame:
    if tf is None:
        raise RuntimeError("TensorFlow not installed; cannot run LSTM")

    df = train_df.sort_values("datetime").copy()
    time_step = infer_time_step(df)
    have = [c for c in exog_cols if c in df.columns]
    cols = ["wind_speed"] + have
    Z = df[cols].astype(float).values

    scaler = MinMaxScaler().fit(Z)
    Zs = scaler.transform(Z)

    def make_seq_single(series: np.ndarray, lb: int):
        X, y = [], []
        for i in range(lb, len(series)):
            X.append(series[i-lb:i, :]); y.append(series[i, 0])
        return np.asarray(X), np.asarray(y)

    def make_seq_multi(series: np.ndarray, lb: int, H: int):
        X, Y = [], []
        # ensure we have H steps ahead
        for i in range(lb, len(series) - H + 1):
            X.append(series[i-lb:i, :])
            Y.append(series[i:i+H, 0])   # only target channel
        return np.asarray(X), np.asarray(Y)

    # Not enough data
    if (not direct and len(Zs) <= lookback + 1) or (direct and len(Zs) <= lookback + horizon):
        return seasonal_persistence(train_df, horizon)

    if direct:
        X, Y = make_seq_multi(Zs, lookback, int(horizon))
        out_dim = int(horizon)
    else:
        X, Y = make_seq_single(Zs, lookback)
        out_dim = 1

    model = Sequential()
    model.add(LSTM(units, input_shape=(lookback, X.shape[2])))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(out_dim))

    keras_loss = tf.keras.losses.Huber() if loss.lower() == "huber" else "mse"
    model.compile(optimizer="adam", loss=keras_loss)
    es = EarlyStopping(monitor="loss", patience=2, min_delta=1e-4, restore_best_weights=True)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[es], shuffle=False)

    fut_idx = pd.date_range(df["datetime"].max() + time_step, periods=int(horizon), freq=time_step)

    # direct multi-horizon: single forward pass
    if direct:
        window = Zs[-lookback:, :].reshape(1, lookback, -1)
        yhat_scaled_vec = model.predict(window, verbose=0)[0]  # shape (H,)
        if have:
            # Build matrix to invert scaling (pred target + exog)
            if fut_actual_exog_df is not None:
                fut_df = fut_actual_exog_df.reindex(fut_idx)[have].astype(float)
                for c in have:
                    if fut_df[c].isna().any():
                        fut_df[c] = fut_df[c].fillna(df[c].iloc[-1])
                exog_part = scaler.transform(
                    np.column_stack([np.repeat(df["wind_speed"].iloc[-1], len(fut_df)), fut_df.values])
                )[:, 1:]
            else:
                last_exog = scaler.transform(
                    np.hstack([[df["wind_speed"].iloc[-1]], df[have].iloc[-1].values]).reshape(1, -1)
                )[0, 1:]
                exog_part = np.repeat(last_exog.reshape(1, -1), len(yhat_scaled_vec), axis=0)
            inv_mat = np.column_stack([yhat_scaled_vec, exog_part])
        else:
            inv_mat = yhat_scaled_vec.reshape(-1, 1)
        inv = scaler.inverse_transform(inv_mat)
        preds = inv[:, 0].clip(min=0.0)
        return pd.DataFrame({"datetime": fut_idx, "wind_speed": preds})

    # recursive (original) path
    window = Zs[-lookback:, :].copy()
    preds_scaled = []

    fut_exog_arr = None
    if fut_actual_exog_df is not None and have:
        fut_df = fut_actual_exog_df.reindex(fut_idx)[have].astype(float)
        for c in have:
            if fut_df[c].isna().any():
                fut_df[c] = fut_df[c].fillna(df[c].iloc[-1])
        last_target = df["wind_speed"].iloc[-1]
        tmp = np.column_stack([np.repeat(last_target, len(fut_df)), fut_df.values])
        fut_exog_arr = scaler.transform(tmp)[:, 1:]

    for t in range(int(horizon)):
        yhat_scaled = model.predict(window.reshape(1, lookback, -1), verbose=0)[0, 0]
        preds_scaled.append(yhat_scaled)
        next_row = window[-1, :].copy()
        next_row[0] = yhat_scaled
        if have and fut_exog_arr is not None:
            next_row[1:1+len(have)] = fut_exog_arr[t, :]
        window = np.vstack([window[1:], next_row])

    if have:
        if fut_exog_arr is not None:
            exog_part = fut_exog_arr
        else:
            last_exog = scaler.transform(
                np.hstack([[df["wind_speed"].iloc[-1]], df[have].iloc[-1].values]).reshape(1, -1)
            )[0, 1:]
            exog_part = np.repeat(last_exog.reshape(1, -1), len(preds_scaled), axis=0)
        inv_mat = np.column_stack([preds_scaled, exog_part])
    else:
        inv_mat = np.array(preds_scaled).reshape(-1, 1)

    inv = scaler.inverse_transform(inv_mat)
    preds = inv[:, 0].clip(min=0.0)
    return pd.DataFrame({"datetime": fut_idx, "wind_speed": preds})

def sarimax_forecast(train_df: pd.DataFrame,
                     fut_actual_exog_df: pd.DataFrame|None,
                     horizon: int,
                     exog_cols: list[str],
                     order=(2,1,2), seasonal_order=(1,1,1,24)) -> pd.DataFrame:
    if SARIMAX is None:
        raise RuntimeError("statsmodels not installed")
    y = train_df.set_index("datetime")["wind_speed"].astype(float)
    time_step = infer_time_step(train_df)
    fut_idx = pd.date_range(train_df["datetime"].max() + time_step, periods=int(horizon), freq=time_step)

    # exog: history
    have = [c for c in exog_cols if c in train_df.columns]
    X_hist = train_df.set_index("datetime")[have] if have else None

    # exog: future
    if have:
        if fut_actual_exog_df is not None:
            exog_fut = fut_actual_exog_df.reindex(fut_idx)[have]
        else:
            last = train_df.iloc[-1]
            exog_fut = pd.DataFrame({c: float(last[c]) for c in have}, index=fut_idx)
    else:
        exog_fut = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, exog=X_hist, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = res.get_forecast(steps=int(horizon), exog=exog_fut)
    out = pd.DataFrame({"datetime": fut_idx, "wind_speed": pred.predicted_mean.values})
    out["wind_speed"] = out["wind_speed"].clip(lower=0.0)
    return out


def semantic_forecast(train_df: pd.DataFrame,
                      horizon: int,
                      semantic_cfg: dict | None = None) -> pd.DataFrame:
    cfg = semantic_cfg or {}
    try:
        return semantic_token_forecast(
            train_df=train_df,
            horizon=horizon,
            window_size=cfg.get("window_size", 48),
            step_size=cfg.get("step_size", 1),
            n_components=cfg.get("n_components", 8),
            n_clusters=cfg.get("n_clusters", 16),
            n_estimators=cfg.get("n_estimators", 200),
            min_samples_leaf=cfg.get("min_samples_leaf", 2),
            random_state=cfg.get("random_state", 42),
        )
    except Exception:
        return seasonal_persistence(train_df, horizon)


def semantic_lstm_model_forecast(train_df: pd.DataFrame,
                                 horizon: int,
                                 semantic_lstm_cfg: dict | None = None) -> pd.DataFrame:
    cfg = semantic_lstm_cfg or {}
    try:
        return semantic_lstm_forecast(
            train_df=train_df,
            horizon=horizon,
            window_size=cfg.get("window_size", 48),
            step_size=cfg.get("step_size", 1),
            n_components=cfg.get("n_components", 8),
            n_clusters=cfg.get("n_clusters", 16),
            encoder_type=cfg.get("encoder_type", "statistical"),
            encoder_units=cfg.get("encoder_units", 32),
            encoder_epochs=cfg.get("encoder_epochs", 10),
            encoder_batch_size=cfg.get("encoder_batch_size", 32),
            encoder_dropout=cfg.get("encoder_dropout", 0.0),
            sequence_length=cfg.get("sequence_length", 12),
            units=cfg.get("units", 32),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 16),
            dropout=cfg.get("dropout", 0.2),
            loss=cfg.get("loss", "mse"),
            random_state=cfg.get("random_state", 42),
        )
    except Exception:
        return seasonal_persistence(train_df, horizon)

# -------------- Backtest ------------------
def rolling_backtest(df: pd.DataFrame,
                     model: str,
                     horizon: int,
                     train_days: int,
                     step_hours: int,
                     use_future_exog: str,
                     exog_cols: list[str],
                     max_splits: int|None = None,
                     prophet_cfg: dict|None = None,
                     lstm_cfg: dict|None = None,
                     sarimax_cfg: dict|None = None,
                     semantic_cfg: dict|None = None,
                     semantic_lstm_cfg: dict|None = None) -> dict:
    """
    Rolling-origin: for t in [start+train_days ... end-horizon] stepping by step_hours:
        train <= t ; forecast t+1..t+H ; compare to actual
    """
    assert model in ("prophet", "sarimax", "baseline", "lstm", "semantic", "semantic_lstm")
    assert use_future_exog in ("actual", "persistence")

    cut_start = df["datetime"].min() + pd.Timedelta(days=train_days)
    time_step = infer_time_step(df)
    cut_end   = df["datetime"].max() - (time_step * int(horizon))
    if cut_start >= cut_end:
        raise ValueError("Not enough data for chosen train_days & horizon")

    data_by_time = df.set_index("datetime").sort_index()
    candidate_cutoffs = pd.date_range(cut_start, cut_end, freq=pd.Timedelta(hours=int(step_hours)))
    eligible_cutoffs = []
    for cutoff in candidate_cutoffs:
        fut_idx = pd.date_range(cutoff + time_step, periods=int(horizon), freq=time_step)
        truth_slice = data_by_time.reindex(fut_idx)["wind_speed"]
        if truth_slice.isna().any():
            continue
        eligible_cutoffs.append(cutoff)

    if not eligible_cutoffs:
        raise ValueError("No valid rolling cutoffs have a complete future truth window.")

    cutoffs = pd.DatetimeIndex(eligible_cutoffs)
    if max_splits and len(cutoffs) > max_splits:
        # thin the valid cutoffs to at most max_splits evenly spaced points
        idx = np.linspace(0, len(cutoffs) - 1, num=max_splits, dtype=int)
        cutoffs = cutoffs[idx]

    all_rows = []
    for i, cutoff in enumerate(cutoffs, 1):
        train_df = trim_history(df, cutoff, train_days=train_days)
        fut_idx = pd.date_range(cutoff + time_step, periods=int(horizon), freq=time_step)
        truth = data_by_time.reindex(fut_idx)["wind_speed"].values

        # future exog actuals if requested
        fut_exog = None
        if use_future_exog == "actual":
            have = [c for c in exog_cols if c in df.columns]
            if have:
                fut_exog = data_by_time[have]

        if model == "baseline":
            fc = seasonal_persistence(train_df, horizon)
        elif model == "prophet":
            fc = prophet_forecast(train_df, fut_exog, horizon, exog_cols, **(prophet_cfg or {}))
        elif model == "lstm":
            fc = lstm_forecast(train_df, fut_exog, horizon, exog_cols, **(lstm_cfg or {}))
        elif model == "semantic":
            fc = semantic_forecast(train_df, horizon, semantic_cfg=semantic_cfg)
        elif model == "semantic_lstm":
            fc = semantic_lstm_model_forecast(train_df, horizon, semantic_lstm_cfg=semantic_lstm_cfg)
        elif model == "sarimax":
            s_cfg = sarimax_cfg or {}
            fc = sarimax_forecast(train_df, fut_exog, horizon, exog_cols,
                                  order=s_cfg.get("order", (2, 1, 2)),
                                  seasonal_order=s_cfg.get("seasonal_order", (1, 1, 1, 24)))
        else:
            raise ValueError(f"Unknown model: {model}")

        # align & compute metrics
        pred = fc.set_index("datetime").reindex(fut_idx)["wind_speed"].ffill().fillna(0.0).values
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            print(f"[{i}/{len(cutoffs)}] cutoff={cutoff}  skipped invalid truth/pred window")
            continue
        m = metrics(truth, pred)
        base_pred = seasonal_persistence(train_df, horizon).set_index("datetime").reindex(fut_idx)["wind_speed"].values
        if not np.isfinite(base_pred).all():
            print(f"[{i}/{len(cutoffs)}] cutoff={cutoff}  skipped invalid persistence window")
            continue
        base_m = metrics(truth, base_pred)
        skill = 1.0 - (m["mae"] / (base_m["mae"] + 1e-9))

        all_rows.append({
            "cutoff": cutoff,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "smape": m["smape"],
            "nmae": m["nmae"],  # NEW
            "nrmse": m["nrmse"],  # NEW
            "baseline_mae": base_m["mae"],
            "skill_vs_persistence": skill
        })

        print(f"[{i}/{len(cutoffs)}] cutoff={cutoff}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  skill={skill:.2%}")

    if not all_rows:
        raise ValueError("No valid splits were evaluated after filtering incomplete windows.")

    table = pd.DataFrame(all_rows)
    summary = table[["mae", "rmse", "smape", "nmae", "nrmse", "skill_vs_persistence"]].mean().to_dict()
    summary["valid_splits"] = int(len(table))
    return {"splits": table, "summary": summary}

# -------------- CLI -----------------------
def main():
    p = argparse.ArgumentParser("Backtest wind-speed forecasting with exogenous inputs")

    # ---- Core args (restored) ----
    p.add_argument("--data", type=str, default="data/wind_data.csv",
                   help="features file (has columns: datetime, wind_speed, [exog...])")
    p.add_argument("--model", type=str, default="prophet",
                   choices=["prophet", "sarimax", "baseline", "lstm", "semantic", "semantic_lstm"])
    p.add_argument("--horizon", type=int, default=168,
                   help="forecast horizon in hours (e.g., 168 for 7 days)")
    p.add_argument("--train_days", type=int, default=120,
                   help="rolling training window length in days")
    p.add_argument("--step_hours", type=int, default=24,
                   help="step between cutoffs in hours")
    p.add_argument("--use_future_exog", type=str, default="persistence",
                   choices=["persistence", "actual"],
                   help="future exog: 'persistence' (live-like) or 'actual' (oracle upper bound)")
    p.add_argument("--max_splits", type=int, default=12,
                   help="limit number of rolling cutoffs (speed)")
    p.add_argument("--exog", type=str, nargs="*", default=EXOG_COLS_DEFAULT,
                   help="which exogenous columns to use; pass nothing for univariate")
    p.add_argument("--csv_out", type=str, default=None,
                   help="optional: write per-split metrics to CSV")

    # ---- LSTM knobs (already present) ----
    p.add_argument("--lstm_lookback", type=int, default=48)
    p.add_argument("--lstm_units", type=int, default=64)
    p.add_argument("--lstm_epochs", type=int, default=15)
    p.add_argument("--lstm_batch_size", type=int, default=32)
    p.add_argument("--lstm_dropout", type=float, default=0.2)
    p.add_argument("--lstm_loss", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--lstm_direct", action="store_true")

    # ---- Prophet knobs (already present) ----
    p.add_argument("--prophet_daily_fourier", type=int, default=12)
    p.add_argument("--prophet_weekly_fourier", type=int, default=6)
    p.add_argument("--prophet_cps", type=float, default=0.10)
    p.add_argument("--prophet_n_changepoints", type=int, default=10)

    # ---- SARIMAX knobs (already present) ----
    p.add_argument("--sarimax_order", type=str, default="2,1,2")
    p.add_argument("--sarimax_seasonal_order", type=str, default="1,1,1,24")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--semantic_window_size", type=int, default=48)
    p.add_argument("--semantic_step_size", type=int, default=1)
    p.add_argument("--semantic_components", type=int, default=8)
    p.add_argument("--semantic_clusters", type=int, default=16)
    p.add_argument("--semantic_estimators", type=int, default=200)
    p.add_argument("--semantic_min_samples_leaf", type=int, default=2)
    p.add_argument("--semantic_encoder_type", type=str, default="statistical", choices=["statistical", "pca", "lstm"])
    p.add_argument("--semantic_encoder_units", type=int, default=32)
    p.add_argument("--semantic_encoder_epochs", type=int, default=10)
    p.add_argument("--semantic_encoder_batch_size", type=int, default=32)
    p.add_argument("--semantic_encoder_dropout", type=float, default=0.0)
    p.add_argument("--semantic_lstm_sequence_length", type=int, default=12)
    p.add_argument("--semantic_lstm_units", type=int, default=32)
    p.add_argument("--semantic_lstm_epochs", type=int, default=10)
    p.add_argument("--semantic_lstm_batch_size", type=int, default=16)
    p.add_argument("--semantic_lstm_dropout", type=float, default=0.2)
    p.add_argument("--semantic_lstm_loss", type=str, default="mse", choices=["mse", "huber"])

    args = p.parse_args()
    set_reproducible(args.seed)

    # helper to parse tuples like "2,1,2" and "1,1,1,24"
    def _parse_tuple(s: str, n: int):
        vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        if len(vals) != n:
            raise ValueError(f"Expected {n} ints in '{s}'")
        return tuple(vals)

    sarimax_cfg = {
        "order": _parse_tuple(args.sarimax_order, 3),
        "seasonal_order": _parse_tuple(args.sarimax_seasonal_order, 4),
    }
    prophet_cfg = {
        "daily_fourier": args.prophet_daily_fourier,
        "weekly_fourier": args.prophet_weekly_fourier,
        "cps": args.prophet_cps,
        "n_changepoints": args.prophet_n_changepoints,
    }
    lstm_cfg = {
        "lookback": args.lstm_lookback,
        "units": args.lstm_units,
        "epochs": args.lstm_epochs,
        "batch_size": args.lstm_batch_size,
        "dropout": args.lstm_dropout,
        "loss": args.lstm_loss,
        "direct": args.lstm_direct,
    }
    semantic_cfg = {
        "window_size": args.semantic_window_size,
        "step_size": args.semantic_step_size,
        "n_components": args.semantic_components,
        "n_clusters": args.semantic_clusters,
        "n_estimators": args.semantic_estimators,
        "min_samples_leaf": args.semantic_min_samples_leaf,
        "random_state": args.seed,
    }
    semantic_lstm_cfg = {
        "window_size": args.semantic_window_size,
        "step_size": args.semantic_step_size,
        "n_components": args.semantic_components,
        "n_clusters": args.semantic_clusters,
        "encoder_type": args.semantic_encoder_type,
        "encoder_units": args.semantic_encoder_units,
        "encoder_epochs": args.semantic_encoder_epochs,
        "encoder_batch_size": args.semantic_encoder_batch_size,
        "encoder_dropout": args.semantic_encoder_dropout,
        "sequence_length": args.semantic_lstm_sequence_length,
        "units": args.semantic_lstm_units,
        "epochs": args.semantic_lstm_epochs,
        "batch_size": args.semantic_lstm_batch_size,
        "dropout": args.semantic_lstm_dropout,
        "loss": args.semantic_lstm_loss,
        "random_state": args.seed,
    }

    df = load_features(args.data)

    result = rolling_backtest(
        df=df,
        model=args.model,
        horizon=args.horizon,
        train_days=args.train_days,
        step_hours=args.step_hours,
        use_future_exog=args.use_future_exog,
        exog_cols=args.exog,
        max_splits=args.max_splits,
        prophet_cfg=prophet_cfg,
        lstm_cfg=lstm_cfg,
        sarimax_cfg=sarimax_cfg,
        semantic_cfg=semantic_cfg,
        semantic_lstm_cfg=semantic_lstm_cfg,
    )

    print("\n=== Summary ===")
    percent_keys = ("nmae", "nrmse", "nrmse_std")

    for k, v in result["summary"].items():
        if isinstance(v, float):
            if k in percent_keys:
                print(f"{k}: {v * 100:.2f}%")
            else:
                print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    if args.csv_out:
        result["splits"].to_csv(args.csv_out, index=False)
        print(f"Wrote per-split metrics to {args.csv_out}")


if __name__ == "__main__":
    main()
