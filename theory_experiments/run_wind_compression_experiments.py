from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Admin\Desktop\wind_forecasting_app")
OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\nf1\wind_compression_experiments")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_ollama import load_features, rolling_backtest, set_reproducible  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "wind_data.csv"
HORIZONS = [1, 3, 6, 12]
SEED = 42
TRAIN_DAYS_TUNE = 45
TRAIN_DAYS_FINAL = 60
STEP_HOURS = 168
TUNE_SPLITS = 2
FINAL_SPLITS = 4


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
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


def raw_lstm_configs() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "raw_direct_lb24_u32_huber",
            "model": "lstm",
            "lstm_cfg": {
                "lookback": 24,
                "units": 32,
                "epochs": 8,
                "batch_size": 32,
                "dropout": 0.10,
                "loss": "huber",
                "direct": True,
            },
        },
        {
            "config_id": "raw_direct_lb48_u64_huber",
            "model": "lstm",
            "lstm_cfg": {
                "lookback": 48,
                "units": 64,
                "epochs": 10,
                "batch_size": 32,
                "dropout": 0.20,
                "loss": "huber",
                "direct": True,
            },
        },
        {
            "config_id": "raw_recursive_lb48_u64_mse",
            "model": "lstm",
            "lstm_cfg": {
                "lookback": 48,
                "units": 64,
                "epochs": 10,
                "batch_size": 32,
                "dropout": 0.20,
                "loss": "mse",
                "direct": False,
            },
        },
    ]


def semantic_lstm_configs(encoder_type: str) -> list[dict[str, Any]]:
    prefix = "pca" if encoder_type in {"statistical", "pca"} else "lstmenc"
    base = [
        {
            "config_id": f"{prefix}_c8_k12_seq8_u32_huber",
            "model": "semantic_lstm",
            "semantic_lstm_cfg": {
                "window_size": 24,
                "step_size": 1,
                "n_components": 8,
                "n_clusters": 12,
                "encoder_type": encoder_type,
                "encoder_units": 24,
                "encoder_epochs": 5,
                "encoder_batch_size": 32,
                "encoder_dropout": 0.0,
                "sequence_length": 8,
                "units": 32,
                "epochs": 8,
                "batch_size": 16,
                "dropout": 0.15,
                "loss": "huber",
                "random_state": SEED,
            },
        },
        {
            "config_id": f"{prefix}_c12_k16_seq12_u48_huber",
            "model": "semantic_lstm",
            "semantic_lstm_cfg": {
                "window_size": 48,
                "step_size": 1,
                "n_components": 12,
                "n_clusters": 16,
                "encoder_type": encoder_type,
                "encoder_units": 32,
                "encoder_epochs": 6,
                "encoder_batch_size": 32,
                "encoder_dropout": 0.0,
                "sequence_length": 12,
                "units": 48,
                "epochs": 8,
                "batch_size": 16,
                "dropout": 0.20,
                "loss": "huber",
                "random_state": SEED,
            },
        },
    ]
    return base


def all_designs() -> dict[str, list[dict[str, Any]]]:
    return {
        "raw_lstm": raw_lstm_configs(),
        "pca_compressed_lstm": semantic_lstm_configs("statistical"),
        "lstm_compressed_lstm": semantic_lstm_configs("lstm"),
    }


def run_one(
    df: pd.DataFrame,
    design: str,
    horizon: int,
    config: dict[str, Any],
    exog_cols: list[str],
    train_days: int,
    max_splits: int,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    start = time.time()
    row: dict[str, Any] = {
        "design": design,
        "horizon": int(horizon),
        "config_id": config["config_id"],
        "status": "ok",
        "train_days": int(train_days),
        "max_splits": int(max_splits),
    }
    try:
        result = rolling_backtest(
            df=df,
            model=config["model"],
            horizon=int(horizon),
            train_days=int(train_days),
            step_hours=STEP_HOURS,
            use_future_exog="persistence",
            exog_cols=exog_cols,
            max_splits=int(max_splits),
            lstm_cfg=config.get("lstm_cfg"),
            semantic_lstm_cfg=config.get("semantic_lstm_cfg"),
        )
        row.update(result["summary"])
        splits = result["splits"].copy()
        splits.insert(0, "config_id", config["config_id"])
        splits.insert(0, "horizon", int(horizon))
        splits.insert(0, "design", design)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        splits = None
    row["elapsed_sec"] = round(time.time() - start, 2)
    row["config_json"] = json.dumps(_json_safe(config), sort_keys=True)
    print(
        f"[{row['status']}] {design} h={horizon} {config['config_id']} "
        f"mae={row.get('mae')} rmse={row.get('rmse')} elapsed={row['elapsed_sec']}s",
        flush=True,
    )
    return row, splits


def run_tuning(df: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for design, configs in all_designs().items():
            for config in configs:
                row, _ = run_one(
                    df=df,
                    design=design,
                    horizon=horizon,
                    config=config,
                    exog_cols=exog_cols,
                    train_days=TRAIN_DAYS_TUNE,
                    max_splits=TUNE_SPLITS,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tuning_results_live.csv", index=False)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    return tuning


def select_best(tuning: pd.DataFrame) -> pd.DataFrame:
    ok = tuning[tuning["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful tuning runs.")
    ok["rank_metric"] = pd.to_numeric(ok["mae"], errors="coerce")
    best = (
        ok.sort_values(["design", "horizon", "rank_metric", "rmse"], ascending=[True, True, True, True])
        .groupby(["design", "horizon"], as_index=False)
        .first()
    )
    best.to_csv(OUTPUT_DIR / "best_tuning_configs.csv", index=False)
    return best


def run_final(df: pd.DataFrame, exog_cols: list[str], best: pd.DataFrame) -> pd.DataFrame:
    configs_by_id: dict[str, dict[str, Any]] = {}
    for configs in all_designs().values():
        for config in configs:
            configs_by_id[config["config_id"]] = config

    rows: list[dict[str, Any]] = []
    split_tables: list[pd.DataFrame] = []
    for _, best_row in best.iterrows():
        config = configs_by_id[str(best_row["config_id"])]
        row, splits = run_one(
            df=df,
            design=str(best_row["design"]),
            horizon=int(best_row["horizon"]),
            config=config,
            exog_cols=exog_cols,
            train_days=TRAIN_DAYS_FINAL,
            max_splits=FINAL_SPLITS,
        )
        rows.append(row)
        if splits is not None:
            split_tables.append(splits)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "final_results_live.csv", index=False)

    final = pd.DataFrame(rows)
    final.to_csv(OUTPUT_DIR / "final_results.csv", index=False)
    if split_tables:
        pd.concat(split_tables, ignore_index=True).to_csv(OUTPUT_DIR / "final_split_metrics.csv", index=False)
    return final


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducible(SEED)
    df = load_features(str(DATA_PATH))
    exog_cols = _present_exog(df)

    manifest = {
        "data_path": str(DATA_PATH),
        "rows_after_loading": int(len(df)),
        "datetime_min": str(df["datetime"].min()),
        "datetime_max": str(df["datetime"].max()),
        "horizons": HORIZONS,
        "seed": SEED,
        "train_days_tune": TRAIN_DAYS_TUNE,
        "train_days_final": TRAIN_DAYS_FINAL,
        "tune_splits": TUNE_SPLITS,
        "final_splits": FINAL_SPLITS,
        "step_hours": STEP_HOURS,
        "exog_cols": exog_cols,
        "designs": _json_safe(all_designs()),
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tuning = run_tuning(df, exog_cols)
    best = select_best(tuning)
    final = run_final(df, exog_cols, best)

    summary = {
        "best_configs": json.loads(best.to_json(orient="records")),
        "final_results": json.loads(final.to_json(orient="records")),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
