from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "metadata" / "ddc_5min_phase_semantic_build.json"
RAW_PATH = ROOT / "data" / "noaa_5min" / "DDC_2024_5min.parquet"
ENCODER_PATH = ROOT / "artifacts" / "semantic_encoder" / "ddc_5min_phase_encoder.joblib"
OUT = ROOT / "results" / "ddc_5min_two_seed_experiments"


def reduction_percent(before: float, after: float) -> float:
    return 100.0 * (1.0 - after / before)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raw = pd.read_parquet(RAW_PATH)
    encoder = joblib.load(ENCODER_PATH)
    reducer = encoder["reducer"]

    window_size = int(metadata["window_size"])
    step_size = int(metadata["step_size"])
    n_windows = int(metadata["n_windows"])
    feature_dim = len(metadata["feature_columns"])
    embedding_dim = int(metadata["embedding_dim"])
    n_tokens = int(metadata["n_tokens"])

    # Only observed meteorological channels are counted as the raw per-window signal.
    raw_channels = [c for c in ["wind_speed", "wind_gust_10m_ms", "wind_direction"] if c in raw.columns]
    raw_window_dim = window_size * len(raw_channels)

    token_bits = (n_tokens - 1).bit_length()
    raw_float32_bytes = raw_window_dim * 4
    feature_float32_bytes = feature_dim * 4
    embedding_float32_bytes = embedding_dim * 4

    rows = [
        {
            "stage": "raw_window_to_engineered_features",
            "before_dim": raw_window_dim,
            "after_dim": feature_dim,
            "compression_ratio": raw_window_dim / feature_dim,
            "reduction_percent": reduction_percent(raw_window_dim, feature_dim),
            "interpretation": "Raw 5-minute samples are summarized into physical window descriptors.",
        },
        {
            "stage": "engineered_features_to_pca_embedding",
            "before_dim": feature_dim,
            "after_dim": embedding_dim,
            "compression_ratio": feature_dim / embedding_dim,
            "reduction_percent": reduction_percent(feature_dim, embedding_dim),
            "interpretation": "Physical descriptors are compressed into the PCA semantic embedding.",
        },
        {
            "stage": "raw_window_to_pca_embedding",
            "before_dim": raw_window_dim,
            "after_dim": embedding_dim,
            "compression_ratio": raw_window_dim / embedding_dim,
            "reduction_percent": reduction_percent(raw_window_dim, embedding_dim),
            "interpretation": "End-to-end numeric dimensional compression before clustering.",
        },
        {
            "stage": "raw_window_to_token_id",
            "before_dim": raw_window_dim,
            "after_dim": 1,
            "compression_ratio": raw_window_dim,
            "reduction_percent": reduction_percent(raw_window_dim, 1),
            "interpretation": "Each four-hour raw window is represented by one discrete regime token.",
        },
        {
            "stage": "raw_window_to_token_one_hot",
            "before_dim": raw_window_dim,
            "after_dim": n_tokens,
            "compression_ratio": raw_window_dim / n_tokens,
            "reduction_percent": reduction_percent(raw_window_dim, n_tokens),
            "interpretation": "Alternative fair comparison if a token is represented as an 8-way one-hot vector.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "compression_metrics_5min.csv", index=False)

    summary = {
        "data_interval": "5min",
        "n_raw_rows": int(len(raw)),
        "n_semantic_windows": n_windows,
        "window_size_samples": window_size,
        "window_step_samples": step_size,
        "raw_channels_counted": raw_channels,
        "raw_window_dim": raw_window_dim,
        "engineered_feature_dim": feature_dim,
        "embedding_dim": embedding_dim,
        "n_tokens": n_tokens,
        "token_bits_minimum": token_bits,
        "pca_explained_variance_ratio": [float(x) for x in reducer.explained_variance_ratio_],
        "pca_total_explained_variance": float(sum(reducer.explained_variance_ratio_)),
        "estimated_float32_bytes_per_window": {
            "raw_window": raw_float32_bytes,
            "engineered_features": feature_float32_bytes,
            "pca_embedding": embedding_float32_bytes,
            "token_id_minimum_bits": token_bits,
        },
        "estimated_float32_storage_ratios": {
            "raw_to_features": raw_float32_bytes / feature_float32_bytes,
            "raw_to_embedding": raw_float32_bytes / embedding_float32_bytes,
        },
        "temporal_compression": {
            "samples_per_token_window": window_size,
            "sample_cadence_minutes": 5,
            "token_step_samples": step_size,
            "token_step_minutes": step_size * 5,
        },
    }
    (OUT / "compression_metrics_5min_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved metrics to {OUT}")


if __name__ == "__main__":
    main()

