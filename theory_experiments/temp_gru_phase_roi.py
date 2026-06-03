from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, GRU, Input
    from tensorflow.keras.optimizers import Adam
except Exception as exc:  # pragma: no cover
    raise RuntimeError("TensorFlow is required for this temporary GRU ROI experiment.") from exc


PROJECT_ROOT = Path(r"C:\Users\Admin\Desktop\wind_forecasting_app")
OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\nf1\wind_compression_experiments")
STATES_PATH = PROJECT_ROOT / "data" / "semantic" / "kbos_5min_phase_semantic_states.csv"
TRANSITION_FINAL_PATH = OUTPUT_DIR / "phase_transition_final_results.csv"
SEED = 42
HORIZONS = [1, 3, 6, 12]
TOP_K = 3


BASE_FEATURE_COLUMNS = [
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


def set_reproducible(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_state_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    states = pd.read_csv(STATES_PATH)
    if "window_start" in states.columns:
        states["window_start"] = pd.to_datetime(states["window_start"], errors="coerce")
        states = states.sort_values("window_start").reset_index(drop=True)

    states["token_id"] = pd.to_numeric(states["token_id"], errors="coerce")
    states = states.dropna(subset=["token_id"]).reset_index(drop=True)
    tokens = states["token_id"].astype(int).to_numpy()

    embedding_columns = [col for col in states.columns if col.startswith("embedding_")]
    feature_columns = [col for col in BASE_FEATURE_COLUMNS if col in states.columns] + embedding_columns
    feature_frame = states[feature_columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    features = feature_frame.to_numpy(dtype=float)
    return states, tokens, features, feature_columns


def make_sequences(
    tokens: np.ndarray,
    features: np.ndarray,
    history_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tokens: list[np.ndarray] = []
    x_features: list[np.ndarray] = []
    targets: list[int] = []
    target_indices: list[int] = []
    for end_idx in range(history_length - 1, len(tokens) - horizon):
        start_idx = end_idx - history_length + 1
        target_idx = end_idx + horizon
        x_tokens.append(tokens[start_idx : end_idx + 1])
        x_features.append(features[start_idx : end_idx + 1])
        targets.append(int(tokens[target_idx]))
        target_indices.append(int(target_idx))
    return (
        np.asarray(x_tokens, dtype=int),
        np.asarray(x_features, dtype=float),
        np.asarray(targets, dtype=int),
        np.asarray(target_indices, dtype=int),
    )


def build_model(history_length: int, n_features: int, n_tokens: int, config: dict[str, Any]) -> Model:
    token_input = Input(shape=(history_length,), name="token_sequence")
    feature_input = Input(shape=(history_length, n_features), name="feature_sequence")
    token_embedding = Embedding(
        input_dim=n_tokens,
        output_dim=int(config["embedding_dim"]),
        name="token_embedding",
    )(token_input)
    merged = Concatenate(axis=-1)([token_embedding, feature_input])
    encoded = GRU(int(config["units"]), name="gru_encoder")(merged)
    dropout = float(config.get("dropout", 0.0))
    if dropout > 0:
        encoded = Dropout(dropout)(encoded)
    output = Dense(n_tokens, activation="softmax", name="phase_distribution")(encoded)
    model = Model(inputs=[token_input, feature_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=float(config.get("learning_rate", 0.001))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    ranked = np.argsort(-probabilities, axis=1)
    top1 = ranked[:, 0]
    topk = ranked[:, :TOP_K]
    labels = sorted(set(y_true.tolist()) | set(top1.tolist()))
    return {
        "top1_accuracy": float(np.mean(top1 == y_true)),
        "top3_accuracy": float(np.mean([actual in row for actual, row in zip(y_true, topk)])),
        "macro_f1": float(f1_score(y_true, top1, labels=labels, average="macro", zero_division=0)),
    }


def train_eval_one(
    tokens: np.ndarray,
    scaled_features: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    horizon: int,
    config: dict[str, Any],
    n_tokens: int,
) -> tuple[dict[str, Any], Model]:
    history_length = int(config["history_length"])
    x_tok, x_feat, y, target_idx = make_sequences(tokens, scaled_features, history_length, horizon)
    tr = train_mask[target_idx]
    va = val_mask[target_idx]
    if tr.sum() < 30 or va.sum() < 10:
        raise ValueError("Not enough train/validation sequences for this config.")

    model = build_model(history_length, x_feat.shape[-1], n_tokens, config)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(config.get("patience", 3)),
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        [x_tok[tr], x_feat[tr]],
        y[tr],
        validation_data=([x_tok[va], x_feat[va]], y[va]),
        epochs=int(config.get("epochs", 20)),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    probs = model.predict([x_tok[va], x_feat[va]], verbose=0)
    row = metrics(y[va], probs)
    row.update(
        {
            "horizon": int(horizon),
            "history_length": history_length,
            "config_id": str(config["config_id"]),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "n_train": int(tr.sum()),
            "n_eval": int(va.sum()),
        }
    )
    return row, model


def final_train_eval(
    tokens: np.ndarray,
    scaled_features: np.ndarray,
    train_val_mask: np.ndarray,
    test_mask: np.ndarray,
    horizon: int,
    config: dict[str, Any],
    n_tokens: int,
) -> dict[str, Any]:
    history_length = int(config["history_length"])
    x_tok, x_feat, y, target_idx = make_sequences(tokens, scaled_features, history_length, horizon)
    tr = train_val_mask[target_idx]
    te = test_mask[target_idx]
    if tr.sum() < 30 or te.sum() < 10:
        raise ValueError("Not enough train/test sequences for this config.")

    model = build_model(history_length, x_feat.shape[-1], n_tokens, config)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(config.get("patience", 3)),
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    model.fit(
        [x_tok[tr], x_feat[tr]],
        y[tr],
        validation_split=0.12,
        epochs=int(config.get("epochs", 20)),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    probs = model.predict([x_tok[te], x_feat[te]], verbose=0)
    row = metrics(y[te], probs)
    row.update(
        {
            "horizon": int(horizon),
            "history_length": history_length,
            "config_id": str(config["config_id"]),
            "n_train": int(tr.sum()),
            "n_eval": int(te.sum()),
        }
    )
    return row


def configs() -> list[dict[str, Any]]:
    # Single-seed tuning grid: broad enough for thesis evidence, but still small
    # enough to run quickly on CPU without turning this into a deep-learning study.
    base = {
        "batch_size": 32,
        "epochs": 28,
        "learning_rate": 0.001,
        "patience": 4,
    }
    grid = [
        ("gru_h3_e8_u24_d010_lr1e3", 3, 8, 24, 0.10, 0.001),
        ("gru_h3_e12_u32_d015_lr1e3", 3, 12, 32, 0.15, 0.001),
        ("gru_h3_e16_u48_d020_lr5e4", 3, 16, 48, 0.20, 0.0005),
        ("gru_h4_e8_u32_d010_lr1e3", 4, 8, 32, 0.10, 0.001),
        ("gru_h4_e12_u48_d015_lr1e3", 4, 12, 48, 0.15, 0.001),
        ("gru_h4_e16_u64_d020_lr5e4", 4, 16, 64, 0.20, 0.0005),
        ("gru_h6_e8_u32_d015_lr1e3", 6, 8, 32, 0.15, 0.001),
        ("gru_h6_e12_u48_d020_lr1e3", 6, 12, 48, 0.20, 0.001),
        ("gru_h6_e16_u64_d025_lr5e4", 6, 16, 64, 0.25, 0.0005),
        ("gru_h9_e12_u48_d020_lr1e3", 9, 12, 48, 0.20, 0.001),
        ("gru_h9_e16_u64_d025_lr5e4", 9, 16, 64, 0.25, 0.0005),
    ]
    return [
        {
            **base,
            "config_id": config_id,
            "history_length": history_length,
            "embedding_dim": embedding_dim,
            "units": units,
            "dropout": dropout,
            "learning_rate": learning_rate,
        }
        for config_id, history_length, embedding_dim, units, dropout, learning_rate in grid
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducible(SEED)
    start = time.time()

    states, tokens, features, feature_columns = load_state_frame()
    n_tokens = int(tokens.max()) + 1
    n = len(tokens)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    scaler = StandardScaler().fit(features[:train_end])
    scaled_features = scaler.transform(features)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    train_val_mask[:val_end] = True
    test_mask[val_end:] = True

    manifest = {
        "states_path": str(STATES_PATH),
        "rows": int(len(states)),
        "n_tokens": int(n_tokens),
        "feature_columns": feature_columns,
        "horizons": HORIZONS,
        "configs": configs(),
        "seed": SEED,
        "split": {"train_end": train_end, "val_end": val_end, "test_end": n},
    }
    (OUTPUT_DIR / "temp_gru_phase_roi_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2),
        encoding="utf-8",
    )

    live_path = OUTPUT_DIR / "temp_gru_phase_roi_tuning_live.csv"
    if live_path.exists():
        tuning_rows = pd.read_csv(live_path).to_dict("records")
        print(f"Resuming from {len(tuning_rows)} completed validation rows.")
    else:
        tuning_rows: list[dict[str, Any]] = []
    best_configs: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for config in configs():
            already_done = any(
                int(row["horizon"]) == int(horizon) and str(row["config_id"]) == str(config["config_id"])
                for row in tuning_rows
            )
            if already_done:
                print(f"[skip] h={horizon} {config['config_id']}")
                continue
            set_reproducible(SEED)
            row, _ = train_eval_one(
                tokens=tokens,
                scaled_features=scaled_features,
                train_mask=train_mask,
                val_mask=val_mask,
                horizon=horizon,
                config=config,
                n_tokens=n_tokens,
            )
            row["split"] = "val"
            tuning_rows.append(row)
            pd.DataFrame(tuning_rows).to_csv(live_path, index=False)
            print(f"[val] h={horizon} {config['config_id']} top1={row['top1_accuracy']:.4f} top3={row['top3_accuracy']:.4f}")

        best = sorted(
            [row for row in tuning_rows if row["horizon"] == horizon],
            key=lambda row: (row["top1_accuracy"], row["top3_accuracy"], row["macro_f1"]),
            reverse=True,
        )[0]
        best_configs.append(best)

    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(OUTPUT_DIR / "temp_gru_phase_roi_tuning_results.csv", index=False)
    best_df = pd.DataFrame(best_configs)
    best_df.to_csv(OUTPUT_DIR / "temp_gru_phase_roi_best_configs.csv", index=False)

    final_rows: list[dict[str, Any]] = []
    config_map = {cfg["config_id"]: cfg for cfg in configs()}
    for _, best in best_df.iterrows():
        set_reproducible(SEED)
        config = config_map[str(best["config_id"])]
        row = final_train_eval(
            tokens=tokens,
            scaled_features=scaled_features,
            train_val_mask=train_val_mask,
            test_mask=test_mask,
            horizon=int(best["horizon"]),
            config=config,
            n_tokens=n_tokens,
        )
        row["split"] = "test"
        final_rows.append(row)
        print(f"[test] h={row['horizon']} {row['config_id']} top1={row['top1_accuracy']:.4f} top3={row['top3_accuracy']:.4f}")
    final = pd.DataFrame(final_rows)
    final.to_csv(OUTPUT_DIR / "temp_gru_phase_roi_final_results.csv", index=False)

    comparison = None
    if TRANSITION_FINAL_PATH.exists():
        transition = pd.read_csv(TRANSITION_FINAL_PATH)
        comparison = transition.merge(final, on="horizon", suffixes=("_transition", "_gru"))
        comparison["top1_delta_gru_minus_transition"] = comparison["top1_accuracy_gru"] - comparison["top1_accuracy_transition"]
        comparison["top3_delta_gru_minus_transition"] = comparison["top3_accuracy_gru"] - comparison["top3_accuracy_transition"]
        comparison["macro_f1_delta_gru_minus_transition"] = comparison["macro_f1_gru"] - comparison["macro_f1_transition"]
        comparison.to_csv(OUTPUT_DIR / "temp_gru_phase_roi_comparison.csv", index=False)

    report_lines = [
        "# Temporary GRU Phase ROI Report",
        "",
        "This file is an isolated ROI check. It does not modify the active HITL phase predictor.",
        "",
        "## GRU Final Test Results",
        final.to_string(index=False),
    ]
    if comparison is not None:
        keep = [
            "horizon",
            "top1_accuracy_transition",
            "top1_accuracy_gru",
            "top1_delta_gru_minus_transition",
            "top3_accuracy_transition",
            "top3_accuracy_gru",
            "top3_delta_gru_minus_transition",
            "macro_f1_transition",
            "macro_f1_gru",
            "macro_f1_delta_gru_minus_transition",
        ]
        report_lines.extend(["", "## GRU vs Transition Comparison", comparison[keep].to_string(index=False)])
    report_lines.extend(["", f"Elapsed seconds: {time.time() - start:.2f}"])
    (OUTPUT_DIR / "temp_gru_phase_roi_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved GRU ROI files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
