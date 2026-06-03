from __future__ import annotations

from typing import Any

import pandas as pd

from app.semantic.labeling.llm_regime_explainer import LLMRegimeExplainer


PROFILE_AGG_COLUMNS = {
    "count": ("token_id", "size"),
    "avg_distance": ("token_distance", "mean"),
    "max_distance": ("token_distance", "max"),
    "wind_speed_mean": ("wind_speed_mean", "mean"),
    "wind_speed_std": ("wind_speed_std", "mean"),
    "ramp_abs_mean": ("ramp_abs_mean", "mean"),
    "ramp_abs_max": ("ramp_abs_max", "mean"),
    "gust_factor": ("gust_factor", "mean"),
    "direction_abs_change_mean_deg": ("direction_abs_change_mean_deg", "mean"),
    "direction_net_turn_deg": ("direction_net_turn_deg", "mean"),
    "calm_fraction": ("calm_fraction", "mean"),
    "strong_fraction": ("strong_fraction", "mean"),
    "vector_resultant_strength": ("vector_resultant_strength", "mean"),
}


def build_cluster_profiles(
    feature_frame: pd.DataFrame,
    embeddings_frame: pd.DataFrame,
    representative_examples: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    embedding_columns = [
        column for column in embeddings_frame.columns if column.startswith("embedding_")
    ]
    embedding_subset = embeddings_frame[
        ["window_id", *embedding_columns, "token_id", "token_distance"]
    ]
    merged = feature_frame.merge(
        embedding_subset,
        on="window_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Cannot build cluster profiles from an empty semantic state frame.")

    available_aggs = {
        out_col: spec for out_col, spec in PROFILE_AGG_COLUMNS.items() if spec[0] in merged.columns
    }
    profile_frame = (
        merged.groupby("token_id")
        .agg(**available_aggs)
        .reset_index()
        .sort_values("token_id")
        .reset_index(drop=True)
    )

    examples: list[dict[str, Any]] = []
    for token_id, group in merged.groupby("token_id"):
        rows = (
            group.sort_values("token_distance")
            .head(representative_examples)
            .copy()
        )
        for _, row in rows.iterrows():
            examples.append(
                {
                    "token_id": int(token_id),
                    "window_id": row.get("window_id"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "token_distance": float(row.get("token_distance", 0.0)),
                    "wind_speed_mean": float(row.get("wind_speed_mean", 0.0)),
                    "wind_speed_std": float(row.get("wind_speed_std", 0.0)),
                    "ramp_abs_max": float(row.get("ramp_abs_max", 0.0)),
                    "gust_factor": float(row.get("gust_factor", 0.0)),
                    "direction_abs_change_mean_deg": float(
                        row.get("direction_abs_change_mean_deg", 0.0)
                    ),
                }
            )

    examples_frame = pd.DataFrame(examples)
    return merged, profile_frame, examples_frame


def label_cluster_profiles(
    profile_frame: pd.DataFrame,
    example_frame: pd.DataFrame,
    enable_llm: bool = False,
) -> pd.DataFrame:
    explainer = LLMRegimeExplainer(enable_llm=enable_llm)
    labeled_rows: list[dict[str, Any]] = []

    for _, row in profile_frame.iterrows():
        token_id = int(row["token_id"])
        profile = row.to_dict()
        examples = (
            example_frame[example_frame["token_id"] == token_id]
            .drop(columns=["token_id"], errors="ignore")
            .to_dict(orient="records")
        )
        label_payload = explainer.label_cluster(profile, examples)
        profile.update(label_payload)
        labeled_rows.append(profile)

    return pd.DataFrame(labeled_rows).sort_values("token_id").reset_index(drop=True)


def attach_cluster_labels(
    semantic_state_frame: pd.DataFrame,
    labeled_profile_frame: pd.DataFrame,
) -> pd.DataFrame:
    label_cols = [
        "token_id",
        "regime_name",
        "short_explanation",
        "meteorological_interpretation",
        "label_source",
    ]
    available = [col for col in label_cols if col in labeled_profile_frame.columns]
    return semantic_state_frame.merge(
        labeled_profile_frame[available],
        on="token_id",
        how="left",
    )
