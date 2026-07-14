from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_wind import infer_time_step, metrics, seasonal_persistence, set_reproducible  # noqa: E402


DATA_PATH = ROOT / "data" / "noaa_5min" / "DDC_2024_5min.parquet"
STATES_PATH = ROOT / "data" / "semantic" / "ddc_5min_phase_semantic_states.csv"
OUT = ROOT / "results" / "ddc_gradient_boosting_experiments"
LSTM_RESULTS = ROOT / "results" / "ddc_5min_two_seed_experiments"
KBOS_BEST_CONFIGS = ROOT / "results" / "kbos_gradient_boosting_experiments" / "gb_best_configs.csv"

SEEDS = [42, 123]
HORIZON_HOURS = [1, 3, 6, 12]
STEPS_PER_HOUR = 12
HORIZON_STEPS = {hour: hour * STEPS_PER_HOUR for hour in HORIZON_HOURS}
TRAIN_DAYS = 30
STEP_HOURS = 168
FINAL_SPLITS = 2
TUNING_SPLITS = 2
LOOKBACK_STEPS = 48
ORIGIN_FREQ = "1h"

RAW_COLUMNS = [
    "wind_speed",
    "wind_gust_10m_ms",
    "u100",
    "v100",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

META_COLUMNS = {
    "window_id",
    "window_start",
    "window_end",
    "window_rows",
    "time_span_hours",
    "token_id",
    "token_distance",
    "regime_name",
    "short_explanation",
    "meteorological_interpretation",
    "label_source",
}


@dataclass
class FeatureCache:
    design: str
    origin_times: pd.DatetimeIndex
    matrix: np.ndarray
    columns: list[str]


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


def load_raw_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    df = df.reset_index(drop=True)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[RAW_COLUMNS] = df[RAW_COLUMNS].interpolate(limit_direction="both").ffill().bfill()
    return df


def load_state_frame() -> pd.DataFrame:
    states = pd.read_csv(STATES_PATH)
    states["window_end"] = pd.to_datetime(states["window_end"], errors="coerce")
    states = states.dropna(subset=["window_end"]).sort_values("window_end").reset_index(drop=True)
    non_numeric = {
        "window_id",
        "window_start",
        "window_end",
        "regime_name",
        "short_explanation",
        "meteorological_interpretation",
        "label_source",
    }
    for col in states.columns:
        if col not in non_numeric:
            states[col] = pd.to_numeric(states[col], errors="coerce")
    return states


def make_cutoffs(df: pd.DataFrame, horizon_steps: int) -> pd.DatetimeIndex:
    time_step = infer_time_step(df)
    cut_start = df["datetime"].min() + pd.Timedelta(days=TRAIN_DAYS)
    cut_end = df["datetime"].max() - (time_step * int(horizon_steps))
    candidates = pd.date_range(cut_start, cut_end, freq=pd.Timedelta(hours=STEP_HOURS))
    data_by_time = df.set_index("datetime").sort_index()
    valid: list[pd.Timestamp] = []
    for cutoff in candidates:
        fut_idx = pd.date_range(cutoff + time_step, periods=int(horizon_steps), freq=time_step)
        truth = data_by_time.reindex(fut_idx)["wind_speed"]
        if not truth.isna().any():
            valid.append(cutoff)
    if not valid:
        raise ValueError("No valid rolling cutoffs.")
    return pd.DatetimeIndex(valid)


def select_even(cutoffs: pd.DatetimeIndex, count: int) -> pd.DatetimeIndex:
    if len(cutoffs) <= count:
        return cutoffs
    idx = np.linspace(0, len(cutoffs) - 1, num=count, dtype=int)
    return cutoffs[idx]


def select_tuning_cutoffs(cutoffs: pd.DatetimeIndex, final_cutoffs: pd.DatetimeIndex) -> pd.DatetimeIndex:
    final_set = set(pd.Timestamp(x) for x in final_cutoffs)
    available = [pd.Timestamp(x) for x in cutoffs if pd.Timestamp(x) not in final_set]
    if not available:
        return final_cutoffs[:1]
    start = max(1, int(0.35 * (len(available) - 1)))
    end = max(start + 1, int(0.75 * (len(available) - 1)))
    pool = pd.DatetimeIndex(available[start : end + 1])
    return select_even(pool, min(TUNING_SPLITS, len(pool)))


def build_origin_times(df: pd.DataFrame) -> pd.DatetimeIndex:
    time_step = infer_time_step(df)
    start = df["datetime"].min() + (LOOKBACK_STEPS - 1) * time_step
    end = df["datetime"].max()
    return pd.date_range(start, end, freq=ORIGIN_FREQ)


def build_raw_cache(df: pd.DataFrame, origin_times: pd.DatetimeIndex) -> FeatureCache:
    data = df.set_index("datetime").sort_index()[RAW_COLUMNS]
    rows: list[np.ndarray] = []
    valid_times: list[pd.Timestamp] = []
    time_step = infer_time_step(df)
    for origin in origin_times:
        window_idx = pd.date_range(origin - (LOOKBACK_STEPS - 1) * time_step, origin, freq=time_step)
        window = data.reindex(window_idx)
        if window.isna().any().any():
            continue
        rows.append(window.to_numpy(dtype=float).reshape(-1))
        valid_times.append(origin)
    columns = [f"{col}_lag_{lag:02d}" for lag in range(LOOKBACK_STEPS, 0, -1) for col in RAW_COLUMNS]
    return FeatureCache("raw_window_gb", pd.DatetimeIndex(valid_times), np.asarray(rows, dtype=float), columns)


def build_state_cache(states: pd.DataFrame, origin_times: pd.DatetimeIndex, design: str) -> FeatureCache:
    state_sorted = states.sort_values("window_end").copy()
    origin_frame = pd.DataFrame({"origin_time": origin_times}).sort_values("origin_time")
    aligned = pd.merge_asof(
        origin_frame,
        state_sorted,
        left_on="origin_time",
        right_on="window_end",
        direction="backward",
    ).dropna(subset=["window_end"])

    embedding_cols = [c for c in aligned.columns if c.startswith("embedding_")]
    stat_cols = [
        c
        for c in aligned.columns
        if c not in META_COLUMNS and not c.startswith("embedding_") and pd.api.types.is_numeric_dtype(aligned[c])
    ]
    token_ids = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int)
    n_tokens = int(token_ids.max()) + 1

    if design == "pca_embedding_gb":
        cols = embedding_cols + ["token_distance"]
        matrix = aligned[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    elif design == "statistical_feature_gb":
        cols = stat_cols
        matrix = aligned[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    elif design == "token_state_gb":
        token = pd.to_numeric(aligned["token_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        one_hot = np.zeros((len(aligned), n_tokens), dtype=float)
        mask = (token >= 0) & (token < n_tokens)
        one_hot[np.arange(len(aligned))[mask], token[mask]] = 1.0
        distance = pd.to_numeric(aligned["token_distance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
        matrix = np.hstack([one_hot, distance])
        cols = [f"token_{idx}" for idx in range(n_tokens)] + ["token_distance"]
    else:
        raise ValueError(f"Unknown state-cache design: {design}")

    # Replace any remaining missing values with column medians.
    if np.isnan(matrix).any():
        med = np.nanmedian(matrix, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(med, inds[1])

    return FeatureCache(design, pd.DatetimeIndex(aligned["origin_time"]), matrix, list(cols))


def build_feature_caches(df: pd.DataFrame, states: pd.DataFrame) -> dict[str, FeatureCache]:
    origins = build_origin_times(df)
    return {
        "raw_window_gb": build_raw_cache(df, origins),
        "statistical_feature_gb": build_state_cache(states, origins, "statistical_feature_gb"),
        "pca_embedding_gb": build_state_cache(states, origins, "pca_embedding_gb"),
        "token_state_gb": build_state_cache(states, origins, "token_state_gb"),
    }


def lead_features(origin_times: pd.DatetimeIndex, lead_steps: np.ndarray, horizon_steps: int, time_step: pd.Timedelta) -> np.ndarray:
    future_times = origin_times + (lead_steps * time_step)
    seconds = (
        future_times.hour.to_numpy() * 3600.0
        + future_times.minute.to_numpy() * 60.0
        + future_times.second.to_numpy()
    )
    dow = future_times.dayofweek.to_numpy()
    lead_hours = lead_steps.astype(float) * (time_step / pd.Timedelta(hours=1))
    return np.column_stack(
        [
            lead_steps.astype(float) / float(horizon_steps),
            lead_hours,
            np.sin(2 * np.pi * seconds / 86400.0),
            np.cos(2 * np.pi * seconds / 86400.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ]
    )


def build_training_matrix(
    cache: FeatureCache,
    target: pd.Series,
    cutoff: pd.Timestamp,
    horizon_steps: int,
    train_days: int,
    time_step: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    start = cutoff - pd.Timedelta(days=train_days)
    origin_mask = (cache.origin_times >= start) & (cache.origin_times <= cutoff)
    origin_pos = np.flatnonzero(origin_mask)
    if len(origin_pos) == 0:
        raise ValueError(f"No origins for {cache.design} before cutoff {cutoff}")

    leads = np.arange(1, horizon_steps + 1, dtype=int)
    repeated_origin_pos = np.repeat(origin_pos, len(leads))
    repeated_leads = np.tile(leads, len(origin_pos))
    repeated_origins = cache.origin_times[repeated_origin_pos]
    target_times = repeated_origins + (repeated_leads * time_step)
    known_mask = target_times <= cutoff
    repeated_origin_pos = repeated_origin_pos[known_mask]
    repeated_leads = repeated_leads[known_mask]
    repeated_origins = repeated_origins[known_mask]
    target_times = target_times[known_mask]

    y = target.reindex(target_times).to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        raise ValueError(f"No finite training targets for {cache.design} cutoff {cutoff}")
    base = cache.matrix[repeated_origin_pos[ok]]
    extra = lead_features(pd.DatetimeIndex(repeated_origins[ok]), repeated_leads[ok], horizon_steps, time_step)
    return np.hstack([base, extra]), y[ok]


def build_forecast_matrix(
    cache: FeatureCache,
    cutoff: pd.Timestamp,
    horizon_steps: int,
    time_step: pd.Timedelta,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    pos = cache.origin_times.searchsorted(cutoff, side="right") - 1
    if pos < 0:
        raise ValueError(f"No {cache.design} feature available before cutoff {cutoff}")
    origin = pd.DatetimeIndex([cache.origin_times[pos]] * horizon_steps)
    leads = np.arange(1, horizon_steps + 1, dtype=int)
    future_idx = pd.date_range(cutoff + time_step, periods=horizon_steps, freq=time_step)
    base = np.repeat(cache.matrix[pos : pos + 1], horizon_steps, axis=0)
    extra = lead_features(origin, leads, horizon_steps, time_step)
    return np.hstack([base, extra]), future_idx


def configs() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "hgb_fast_shallow",
            "max_iter": 100,
            "learning_rate": 0.10,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 0.0,
        },
        {
            "config_id": "hgb_balanced",
            "max_iter": 160,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 0.01,
        },
        {
            "config_id": "hgb_regularized",
            "max_iter": 220,
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 80,
            "l2_regularization": 0.10,
        },
        {
            "config_id": "hgb_deeper",
            "max_iter": 180,
            "learning_rate": 0.05,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 40,
            "l2_regularization": 0.03,
        },
    ]


def fit_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    params = {k: v for k, v in config.items() if k != "config_id"}
    return HistGradientBoostingRegressor(
        **params,
        loss="squared_error",
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=int(seed),
    )


def load_transferred_configs() -> dict[tuple[str, int], dict[str, Any]]:
    """Load the KBOS-selected design/horizon settings for transfer to DDC."""

    frame = pd.read_csv(KBOS_BEST_CONFIGS)
    required = {"design", "horizon_hours", "config_json"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {KBOS_BEST_CONFIGS}: {sorted(required - set(frame.columns))}")
    transferred: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        transferred[(str(row["design"]), int(row["horizon_hours"]))] = json.loads(row["config_json"])
    return transferred


def evaluate_forecast(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_steps: int,
    pred: np.ndarray,
    future_idx: pd.DatetimeIndex,
) -> dict[str, float]:
    data_by_time = df.set_index("datetime").sort_index()
    truth = data_by_time.reindex(future_idx)["wind_speed"].to_numpy(dtype=float)
    m = metrics(truth, np.clip(pred.astype(float), 0.0, None))
    train_df = df[(df["datetime"] >= cutoff - pd.Timedelta(days=TRAIN_DAYS)) & (df["datetime"] <= cutoff)].copy()
    base_pred = seasonal_persistence(train_df, horizon_steps).set_index("datetime").reindex(future_idx)["wind_speed"].to_numpy(dtype=float)
    base_m = metrics(truth, base_pred)
    m["baseline_mae"] = base_m["mae"]
    m["skill_vs_persistence"] = 1.0 - (m["mae"] / (base_m["mae"] + 1e-9))
    return m


def tune_design_horizon(
    df: pd.DataFrame,
    target: pd.Series,
    cache: FeatureCache,
    horizon_hours: int,
    horizon_steps: int,
    tuning_cutoffs: pd.DatetimeIndex,
    time_step: pd.Timedelta,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs():
        split_mae: list[float] = []
        split_rmse: list[float] = []
        start = time.time()
        for cutoff in tuning_cutoffs:
            X_train, y_train = build_training_matrix(cache, target, cutoff, horizon_steps, TRAIN_DAYS, time_step)
            X_forecast, future_idx = build_forecast_matrix(cache, cutoff, horizon_steps, time_step)
            model = fit_model(config, seed=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_forecast)
            m = evaluate_forecast(df, cutoff, horizon_steps, pred, future_idx)
            split_mae.append(float(m["mae"]))
            split_rmse.append(float(m["rmse"]))
        rows.append(
            {
                "design": cache.design,
                "horizon_hours": horizon_hours,
                "horizon_steps": horizon_steps,
                "config_id": config["config_id"],
                "mae": float(np.mean(split_mae)),
                "rmse": float(np.mean(split_rmse)),
                "n_tuning_splits": int(len(tuning_cutoffs)),
                "config_json": json.dumps(config, sort_keys=True),
                "elapsed_sec": round(time.time() - start, 2),
            }
        )
        print(
            f"[tune] {cache.design} h={horizon_hours} {config['config_id']} "
            f"mae={rows[-1]['mae']:.4f}",
            flush=True,
        )
    table = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return table, json.loads(table.iloc[0]["config_json"])


def final_eval(
    df: pd.DataFrame,
    target: pd.Series,
    cache: FeatureCache,
    horizon_hours: int,
    horizon_steps: int,
    config: dict[str, Any],
    final_cutoffs: pd.DatetimeIndex,
    time_step: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_reproducible(seed)
        seed_metrics: list[dict[str, float]] = []
        for split_id, cutoff in enumerate(final_cutoffs, 1):
            X_train, y_train = build_training_matrix(cache, target, cutoff, horizon_steps, TRAIN_DAYS, time_step)
            X_forecast, future_idx = build_forecast_matrix(cache, cutoff, horizon_steps, time_step)
            model = fit_model(config, seed=seed)
            start = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_forecast)
            m = evaluate_forecast(df, cutoff, horizon_steps, pred, future_idx)
            seed_metrics.append(m)
            split_rows.append(
                {
                    "seed": seed,
                    "design": cache.design,
                    "horizon_hours": horizon_hours,
                    "horizon_steps": horizon_steps,
                    "config_id": config["config_id"],
                    "split_id": split_id,
                    "cutoff": cutoff,
                    "elapsed_sec": round(time.time() - start, 2),
                    **m,
                }
            )
        seed_summary = pd.DataFrame(seed_metrics)[["mae", "rmse", "smape", "nmae", "nrmse", "skill_vs_persistence"]].mean().to_dict()
        final_rows.append(
            {
                "seed": seed,
                "design": cache.design,
                "horizon_hours": horizon_hours,
                "horizon_steps": horizon_steps,
                "config_id": config["config_id"],
                "valid_splits": len(seed_metrics),
                **seed_summary,
            }
        )
        print(
            f"[final] seed={seed} {cache.design} h={horizon_hours} "
            f"{config['config_id']} mae={final_rows[-1]['mae']:.4f}",
            flush=True,
        )
    return pd.DataFrame(final_rows), pd.DataFrame(split_rows)


def aggregate(final: pd.DataFrame) -> pd.DataFrame:
    summary = (
        final.groupby(["design", "horizon_hours", "horizon_steps"], as_index=False)
        [["mae", "rmse", "smape", "skill_vs_persistence"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return summary


def write_lstm_comparison(gb_summary: pd.DataFrame) -> None:
    lstm_path = LSTM_RESULTS / "numeric_5min_two_seed_summary.csv"
    if not lstm_path.exists():
        return
    lstm = pd.read_csv(lstm_path)
    rows: list[dict[str, Any]] = []
    for _, row in gb_summary.iterrows():
        rows.append(
            {
                "family": "gradient_boosting",
                "design": row["design"],
                "horizon_hours": row["horizon_hours"],
                "mae_mean": row["mae_mean"],
                "rmse_mean": row["rmse_mean"],
                "skill_vs_persistence_mean": row["skill_vs_persistence_mean"],
            }
        )
    for _, row in lstm.iterrows():
        rows.append(
            {
                "family": "lstm_existing",
                "design": row["design"],
                "horizon_hours": row["horizon_hours"],
                "mae_mean": row["mae_mean"],
                "rmse_mean": row["rmse_mean"],
                "skill_vs_persistence_mean": row["skill_vs_persistence_mean"],
            }
        )
    pd.DataFrame(rows).sort_values(["horizon_hours", "mae_mean"]).to_csv(
        OUT / "gb_vs_existing_lstm_numeric_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the transferred-config DDC gradient-boosting evaluation.")
    parser.add_argument("--resume", action="store_true", help="Resume from the *_live.csv checkpoints.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    df = load_raw_frame()
    states = load_state_frame()
    time_step = infer_time_step(df)
    target = df.set_index("datetime").sort_index()["wind_speed"]
    caches = build_feature_caches(df, states)
    transferred_configs = load_transferred_configs()

    best_rows: list[dict[str, Any]] = []
    final_rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    if args.resume:
        best_live = OUT / "gb_best_configs_live.csv"
        final_live = OUT / "gb_final_results_live.csv"
        split_live = OUT / "gb_split_metrics_live.csv"
        if best_live.exists():
            best_rows = pd.read_csv(best_live).to_dict(orient="records")
        if final_live.exists():
            existing_final = pd.read_csv(final_live)
            final_rows.append(existing_final)
            counts = existing_final.groupby(["design", "horizon_hours"])["seed"].nunique()
            completed = {
                (str(design), int(horizon))
                for (design, horizon), count in counts.items()
                if int(count) >= len(SEEDS)
            }
        if split_live.exists():
            split_rows.append(pd.read_csv(split_live))
        print(f"[resume] completed design/horizon pairs: {len(completed)}", flush=True)

    for horizon_hours in HORIZON_HOURS:
        horizon_steps = HORIZON_STEPS[horizon_hours]
        cutoffs = make_cutoffs(df, horizon_steps)
        final_cutoffs = select_even(cutoffs, FINAL_SPLITS)
        print(
            f"[horizon] {horizon_hours}h/{horizon_steps}steps "
            f"transferred_config=KBOS final={list(final_cutoffs)}",
            flush=True,
        )
        for design, cache in caches.items():
            if (design, int(horizon_hours)) in completed:
                print(f"[resume] skip {design} h={horizon_hours}", flush=True)
                continue
            config_key = (design, int(horizon_hours))
            if config_key not in transferred_configs:
                raise KeyError(f"No transferred KBOS configuration for {config_key}")
            best_config = transferred_configs[config_key]
            best_rows.append(
                {
                    "design": design,
                    "horizon_hours": horizon_hours,
                    "horizon_steps": horizon_steps,
                    "config_id": best_config["config_id"],
                    "config_json": json.dumps(best_config, sort_keys=True),
                    "config_source": "KBOS validation selection",
                }
            )
            final, splits = final_eval(
                df=df,
                target=target,
                cache=cache,
                horizon_hours=horizon_hours,
                horizon_steps=horizon_steps,
                config=best_config,
                final_cutoffs=final_cutoffs,
                time_step=time_step,
            )
            final_rows.append(final)
            split_rows.append(splits)

            pd.DataFrame(best_rows).to_csv(OUT / "gb_best_configs_live.csv", index=False)
            pd.concat(final_rows, ignore_index=True).to_csv(OUT / "gb_final_results_live.csv", index=False)
            pd.concat(split_rows, ignore_index=True).to_csv(OUT / "gb_split_metrics_live.csv", index=False)

    best = pd.DataFrame(best_rows)
    final = pd.concat(final_rows, ignore_index=True)
    splits = pd.concat(split_rows, ignore_index=True)
    summary = aggregate(final)

    best.to_csv(OUT / "gb_best_configs.csv", index=False)
    final.to_csv(OUT / "gb_final_results.csv", index=False)
    splits.to_csv(OUT / "gb_split_metrics.csv", index=False)
    summary.to_csv(OUT / "gb_summary.csv", index=False)
    write_lstm_comparison(summary)

    manifest = {
        "algorithm": "HistGradientBoostingRegressor",
        "data_path": DATA_PATH.relative_to(ROOT).as_posix(),
        "semantic_states_path": STATES_PATH.relative_to(ROOT).as_posix(),
        "output_dir": OUT.relative_to(ROOT).as_posix(),
        "seeds": SEEDS,
        "horizon_hours": HORIZON_HOURS,
        "horizon_steps": HORIZON_STEPS,
        "train_days": TRAIN_DAYS,
        "step_hours": STEP_HOURS,
        "final_splits": FINAL_SPLITS,
        "tuning_splits": 0,
        "configuration_protocol": "KBOS-selected configurations transferred unchanged to DDC",
        "configuration_source": KBOS_BEST_CONFIGS.relative_to(ROOT).as_posix(),
        "lookback_steps_for_raw_window": LOOKBACK_STEPS,
        "origin_frequency": ORIGIN_FREQ,
        "designs": {name: cache.columns for name, cache in caches.items()},
        "config_grid": configs(),
        "notes": [
            "Gradient boosting is evaluated as a tabular learner for raw-window and compressed semantic representations.",
            "The model uses direct horizon conditioning: lead step and future calendar features are added to each origin feature vector.",
            "DDC final evaluation transfers the design/horizon configurations selected on KBOS validation folds.",
            "LLM-refined regime names are not used as model inputs; token IDs/embeddings/statistical features are non-LLM outputs.",
        ],
        "execution_mode": "resume" if args.resume else "full",
        "last_invocation_elapsed_sec": round(time.time() - start, 2),
    }
    (OUT / "gb_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    print(f"Saved gradient-boosting outputs to {OUT}")
    print(summary.to_string(index=False))
    print(f"Elapsed seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
