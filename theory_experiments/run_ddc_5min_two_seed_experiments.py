from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_wind import load_features, rolling_backtest, set_reproducible  # noqa: E402


SEEDS = [42, 123]
HORIZON_HOURS = [1, 3, 6, 12]
STEPS_PER_HOUR = 12
HORIZON_STEPS = {hour: hour * STEPS_PER_HOUR for hour in HORIZON_HOURS}
NUMERIC_FINAL_SPLITS = 2
NUMERIC_TRAIN_DAYS = 30
NUMERIC_STEP_HOURS = 168

DATA_PATH = ROOT / "data" / "noaa_5min" / "DDC_2024_5min.parquet"
STATES_PATH = ROOT / "data" / "semantic" / "ddc_5min_phase_semantic_states.csv"
OUT = ROOT / "results" / "ddc_5min_two_seed_experiments"
PREVIOUS_RESULTS = ROOT / "results" / "wind_compression_experiments"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _present_exog(df: pd.DataFrame) -> list[str]:
    candidates = [
        "wind_gusts_10m_ms",
        "wind_gust_10m_ms",
        "temperature_2m_c",
        "relative_humidity_2m",
        "wind_dir_sin",
        "wind_dir_cos",
        "u100",
        "v100",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    return [col for col in candidates if col in df.columns]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _best_numeric_configs() -> pd.DataFrame:
    path = PREVIOUS_RESULTS / "best_tuning_configs.csv"
    best = pd.read_csv(path)
    if "config_json" not in best.columns:
        raise ValueError(f"Expected config_json in {path}")
    return best


def run_numeric_5min() -> pd.DataFrame:
    df = load_features(str(DATA_PATH))
    exog_cols = _present_exog(df)
    best = _best_numeric_configs()
    live_path = OUT / "numeric_5min_two_seed_final_live.csv"
    resumed = live_path.exists()
    if live_path.exists():
        existing = pd.read_csv(live_path)
        rows: list[dict[str, Any]] = existing.to_dict("records")
        completed = {
            (int(row["seed"]), str(row["design"]), int(row["horizon_hours"]))
            for row in rows
            if str(row.get("status", "")).lower() == "ok"
        }
        print(f"[numeric:resume] loaded {len(rows)} existing rows from {live_path}", flush=True)
    else:
        rows = []
        completed = set()
    split_tables: list[pd.DataFrame] = []

    for seed in SEEDS:
        set_reproducible(seed)
        for _, best_row in best.iterrows():
            config = json.loads(str(best_row["config_json"]))
            design = str(best_row["design"])
            horizon_hours = int(best_row["horizon"])
            horizon_steps = int(HORIZON_STEPS[horizon_hours])
            if config.get("semantic_lstm_cfg"):
                config["semantic_lstm_cfg"]["random_state"] = int(seed)
            start = time.time()
            row: dict[str, Any] = {
                "seed": int(seed),
                "design": design,
                "horizon_hours": horizon_hours,
                "horizon_steps": horizon_steps,
                "config_id": str(best_row["config_id"]),
                "status": "ok",
                "train_days": int(NUMERIC_TRAIN_DAYS),
                "max_splits": int(NUMERIC_FINAL_SPLITS),
                "data_interval": "5min",
            }
            if (int(seed), design, horizon_hours) in completed:
                print(
                    f"[numeric:skip] seed={seed} {design} h={horizon_hours}h/{horizon_steps}steps",
                    flush=True,
                )
                continue
            try:
                result = rolling_backtest(
                    df=df,
                    model=str(config["model"]),
                    horizon=horizon_steps,
                    train_days=NUMERIC_TRAIN_DAYS,
                    step_hours=NUMERIC_STEP_HOURS,
                    use_future_exog="persistence",
                    exog_cols=exog_cols,
                    max_splits=NUMERIC_FINAL_SPLITS,
                    lstm_cfg=config.get("lstm_cfg"),
                    semantic_lstm_cfg=config.get("semantic_lstm_cfg"),
                )
                row.update(result["summary"])
                splits = result["splits"].copy()
                splits.insert(0, "config_id", row["config_id"])
                splits.insert(0, "horizon_steps", horizon_steps)
                splits.insert(0, "horizon_hours", horizon_hours)
                splits.insert(0, "design", design)
                splits.insert(0, "seed", seed)
                split_tables.append(splits)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            row["elapsed_sec"] = round(time.time() - start, 2)
            rows.append(row)
            pd.DataFrame(rows).drop_duplicates(
                subset=["seed", "design", "horizon_hours"],
                keep="last",
            ).to_csv(live_path, index=False)
            print(
                f"[numeric:{row['status']}] seed={seed} {design} "
                f"h={horizon_hours}h/{horizon_steps}steps mae={row.get('mae')} "
                f"rmse={row.get('rmse')} elapsed={row['elapsed_sec']}s",
                flush=True,
            )

    final = pd.DataFrame(rows).drop_duplicates(
        subset=["seed", "design", "horizon_hours"],
        keep="last",
    )
    final.to_csv(OUT / "numeric_5min_two_seed_final_results.csv", index=False)
    if split_tables:
        # Split-level tables are complete only for a fresh run. Resume mode skips
        # completed configs, so we avoid writing a misleading partial split file.
        if not resumed:
            pd.concat(split_tables, ignore_index=True).to_csv(
                OUT / "numeric_5min_two_seed_split_metrics.csv",
                index=False,
            )
    return final


def run_phase_transition_5min() -> pd.DataFrame:
    phase = _load_module(ROOT / "theory_experiments" / "run_phase_transition_experiments.py", "phase_transition_runner")
    states = pd.read_csv(STATES_PATH)
    if "window_start" in states.columns:
        states["window_start"] = pd.to_datetime(states["window_start"], errors="coerce")
        states = states.sort_values("window_start").reset_index(drop=True)
    tokens = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int).to_numpy()
    n = len(tokens)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_val_tokens = tokens[:val_end]
    test_tokens = tokens[val_end:]
    best = pd.read_csv(PREVIOUS_RESULTS / "phase_transition_best_configs.csv")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for _, row in best.iterrows():
            result = phase.evaluate(
                train_val_tokens,
                test_tokens,
                horizon=int(row["horizon"]),
                history_length=int(row["history_length"]),
                min_support=int(row["min_support"]),
            )
            result["seed"] = int(seed)
            result["split"] = "test"
            result["data_interval"] = "5min_semantic_windows"
            rows.append(result)
    final = pd.DataFrame(rows)
    final.to_csv(OUT / "phase_transition_5min_two_seed_final_results.csv", index=False)
    print("[phase-transition] saved", len(final), "rows", flush=True)
    return final


def run_gru_phase_5min() -> pd.DataFrame:
    gru = _load_module(ROOT / "theory_experiments" / "gru_phase_roi.py", "gru_phase_runner")
    gru.STATES_PATH = STATES_PATH
    gru.OUTPUT_DIR = OUT
    states, tokens, features, _feature_columns = gru.load_state_frame()
    n_tokens = int(tokens.max()) + 1
    n = len(tokens)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    scaler = gru.StandardScaler().fit(features[:train_end])
    scaled_features = scaler.transform(features)
    train_val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_val_mask[:val_end] = True
    test_mask[val_end:] = True
    config_map = {cfg["config_id"]: cfg for cfg in gru.configs()}
    best = pd.read_csv(PREVIOUS_RESULTS / "gru_phase_roi_best_configs.csv")
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for _, best_row in best.iterrows():
            config = dict(config_map[str(best_row["config_id"])])
            config["seed"] = int(seed)
            gru.set_reproducible(seed)
            row = gru.final_train_eval(
                tokens=tokens,
                scaled_features=scaled_features,
                train_val_mask=train_val_mask,
                test_mask=test_mask,
                horizon=int(best_row["horizon"]),
                config=config,
                n_tokens=n_tokens,
            )
            row["seed"] = int(seed)
            row["split"] = "test"
            row["data_interval"] = "5min_semantic_windows"
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "gru_phase_5min_two_seed_final_live.csv", index=False)
            print(
                f"[gru] seed={seed} h={row['horizon']} {row['config_id']} "
                f"top1={row['top1_accuracy']:.4f} top3={row['top3_accuracy']:.4f}",
                flush=True,
            )

    final = pd.DataFrame(rows)
    final.to_csv(OUT / "gru_phase_5min_two_seed_final_results.csv", index=False)
    return final


def aggregate_results(
    numeric: pd.DataFrame,
    transition: pd.DataFrame,
    gru: pd.DataFrame,
) -> None:
    numeric_ok = numeric[numeric["status"] == "ok"].copy()
    numeric_summary = (
        numeric_ok.groupby(["design", "horizon_hours", "horizon_steps"], as_index=False)
        [["mae", "rmse", "smape", "skill_vs_persistence"]]
        .agg(["mean", "std"])
    )
    numeric_summary.columns = ["_".join(col).strip("_") for col in numeric_summary.columns.values]
    numeric_summary.to_csv(OUT / "numeric_5min_two_seed_summary.csv", index=False)

    transition_summary = (
        transition.groupby(["horizon"], as_index=False)[["top1_accuracy", "top3_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    transition_summary.columns = ["_".join(col).strip("_") for col in transition_summary.columns.values]
    transition_summary.to_csv(OUT / "phase_transition_5min_two_seed_summary.csv", index=False)

    gru_summary = (
        gru.groupby(["horizon"], as_index=False)[["top1_accuracy", "top3_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    gru_summary.columns = ["_".join(col).strip("_") for col in gru_summary.columns.values]
    gru_summary.to_csv(OUT / "gru_phase_5min_two_seed_summary.csv", index=False)

    manifest = {
        "data_path": DATA_PATH.relative_to(ROOT).as_posix(),
        "semantic_states_path": STATES_PATH.relative_to(ROOT).as_posix(),
        "seeds": SEEDS,
        "horizon_hours": HORIZON_HOURS,
        "horizon_steps": HORIZON_STEPS,
        "numeric_train_days": NUMERIC_TRAIN_DAYS,
        "numeric_final_splits": NUMERIC_FINAL_SPLITS,
        "numeric_step_hours": NUMERIC_STEP_HOURS,
        "notes": [
            "Numeric horizons are converted from hours to 5-minute steps.",
            "Numeric forecasting transfers the prior KBOS-selected configs and reruns final evaluation only.",
            "Phase transition and GRU use the 5-minute-derived semantic state sequence.",
        ],
    }
    (OUT / "5min_two_seed_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    numeric = run_numeric_5min()
    transition = run_phase_transition_5min()
    gru = run_gru_phase_5min()
    aggregate_results(numeric, transition, gru)
    print(f"Saved all 5-min two-seed outputs to {OUT}")
    print(f"Elapsed seconds: {time.time() - start:.2f}")


if __name__ == "__main__":
    main()

