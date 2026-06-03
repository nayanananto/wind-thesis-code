from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.feature_engineering import prepare_semantic_frame
from app.semantic.encoders.statistical_encoder import StatisticalWindowEncoder
from app.semantic.summarizers.physics_summary import summarize_window
from app.semantic.tokenization.cluster_tokenizer import ClusterTokenizer
from app.semantic.windowing.window_builder import SemanticWindow


META_COLUMNS = {
    "window_id",
    "window_start",
    "window_end",
    "window_rows",
    "time_span_hours",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw_path: str | None, fallback_dirs: list[Path]) -> Path:
    if not raw_path:
        raise ValueError("Missing artifact path in semantic metadata.")
    path = Path(raw_path)
    if path.exists():
        return path

    basename = path.name
    for directory in fallback_dirs:
        candidate = PROJECT_ROOT / directory / basename
        if candidate.exists():
            return candidate

    matches = list(PROJECT_ROOT.glob(f"**/{basename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve artifact path: {raw_path}")


def _read_live_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No live observations found in {path}")

    rename_map = {
        "observation_time_utc": "datetime",
        "wind_gust": "wind_gust_10m_ms",
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    if "datetime" not in frame.columns:
        raise ValueError("Live CSV must contain observation_time_utc or datetime.")
    if "wind_speed" not in frame.columns:
        raise ValueError("Live CSV must contain wind_speed in m/s.")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(
        subset=["datetime", "station_id"],
        keep="last",
    )
    if frame.empty:
        raise ValueError("No valid live rows remained after datetime parsing.")
    return frame.reset_index(drop=True)


def _build_feature_row(frame: pd.DataFrame, station: str, live_window_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    window_frame = frame.tail(live_window_rows).copy()
    prepared = prepare_semantic_frame(window_frame)
    start = pd.Timestamp(prepared["datetime"].iloc[0])
    end = pd.Timestamp(prepared["datetime"].iloc[-1])
    window_id = f"live_{station}_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}"
    window = SemanticWindow(
        window_id=window_id,
        start_time=start,
        end_time=end,
        row_start=max(0, len(frame) - len(prepared)),
        row_end=len(frame) - 1,
        frame=prepared,
    )
    row = summarize_window(window)
    diagnostics = {
        "live_window_rows_requested": int(live_window_rows),
        "live_window_rows_used": int(len(prepared)),
        "live_window_start": start.isoformat(),
        "live_window_end": end.isoformat(),
        "live_window_span_hours": float((end - start).total_seconds() / 3600.0),
    }
    return pd.DataFrame([row]), diagnostics


def _build_feature_history(frame: pd.DataFrame, station: str, live_window_rows: int) -> pd.DataFrame:
    if len(frame) < live_window_rows:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for end_pos in range(live_window_rows, len(frame) + 1):
        window_frame = frame.iloc[end_pos - live_window_rows : end_pos].copy()
        prepared = prepare_semantic_frame(window_frame)
        if prepared.empty:
            continue
        start = pd.Timestamp(prepared["datetime"].iloc[0])
        end = pd.Timestamp(prepared["datetime"].iloc[-1])
        window = SemanticWindow(
            window_id=f"live_{station}_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}",
            start_time=start,
            end_time=end,
            row_start=end_pos - live_window_rows,
            row_end=end_pos - 1,
            frame=prepared,
        )
        rows.append(summarize_window(window))
    return pd.DataFrame(rows)


def _load_cluster_profiles(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = _resolve_path(
        metadata["output_paths"].get("cluster_profiles"),
        [Path("data/semantic")],
    )
    frame = pd.read_csv(path)
    if frame.empty or "token_id" not in frame.columns:
        return {}
    return {
        int(row["token_id"]): row.to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row.get("token_id"))
    }


def _load_states(metadata: dict[str, Any]) -> pd.DataFrame:
    path = _resolve_path(
        metadata["output_paths"].get("semantic_states"),
        [Path("data/semantic")],
    )
    frame = pd.read_csv(path)
    if "window_start" in frame.columns:
        frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
        frame = frame.sort_values("window_start").reset_index(drop=True)
    return frame


def _similar_windows(live_embedding: np.ndarray, states: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    embedding_columns = [column for column in states.columns if column.startswith("embedding_")]
    if not embedding_columns:
        return []
    matrix = states[embedding_columns].to_numpy(dtype=float)
    query = np.asarray(live_embedding, dtype=float).reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
    distances = 1.0 - ((matrix @ query.ravel()) / np.maximum(norms, 1e-12))
    order = np.argsort(distances)[: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    keep = [
        "window_id",
        "window_start",
        "window_end",
        "token_id",
        "regime_name",
        "wind_speed_mean",
        "wind_speed_std",
        "wind_speed_min",
        "wind_speed_max",
        "gust_factor",
        "direction_abs_change_mean_deg",
    ]
    for rank, idx in enumerate(order, start=1):
        row = states.iloc[int(idx)]
        payload = {key: row.get(key) for key in keep if key in row}
        payload["rank"] = rank
        payload["retrieval_distance"] = float(distances[int(idx)])
        rows.append(payload)
    return rows


def _phase_candidates_from_token(
    token_id: int,
    states: pd.DataFrame,
    profiles: dict[int, dict[str, Any]],
    horizon_steps: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    tokens = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int).to_numpy()
    horizon = max(1, int(horizon_steps))
    counts: Counter[int] = Counter()
    for idx in range(0, len(tokens) - horizon):
        if int(tokens[idx]) == int(token_id):
            counts[int(tokens[idx + horizon])] += 1
    support = sum(counts.values())
    if not support:
        return [], 0

    rows: list[dict[str, Any]] = []
    for rank, (candidate, count) in enumerate(counts.most_common(max(1, int(top_k))), start=1):
        profile = profiles.get(int(candidate), {})
        rows.append(
            {
                "rank": rank,
                "token_id": int(candidate),
                "regime_name": str(profile.get("regime_name") or f"token {candidate}"),
                "probability": round(float(count / support), 4),
                "count": int(count),
                "support": int(support),
                "transition_source": "live_token_markov",
            }
        )
    return rows, support


def build_live_semantic_state(
    live_csv: Path,
    metadata_path: Path,
    station: str,
    live_window_rows: int,
    top_k: int,
    horizon_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    feature_columns = [column for column in metadata.get("feature_columns", []) if column not in META_COLUMNS]
    if not feature_columns:
        raise ValueError("Semantic metadata does not contain feature_columns.")

    encoder_path = _resolve_path(metadata["output_paths"].get("encoder"), [Path("artifacts/semantic_encoder")])
    tokenizer_path = _resolve_path(metadata["output_paths"].get("tokenizer"), [Path("artifacts/tokenizer")])
    encoder = StatisticalWindowEncoder.load(str(encoder_path))
    tokenizer = ClusterTokenizer.load(str(tokenizer_path))

    live_rows = _read_live_rows(live_csv)
    feature_row, diagnostics = _build_feature_row(live_rows, station=station, live_window_rows=live_window_rows)
    feature_history = _build_feature_history(live_rows, station=station, live_window_rows=live_window_rows)
    matrix = feature_row.drop(columns=list(META_COLUMNS), errors="ignore").reindex(columns=feature_columns)
    embedding = encoder.transform(matrix)
    token_frame = tokenizer.transform(embedding)
    token_id = int(token_frame.iloc[0]["token_id"])
    token_distance = float(token_frame.iloc[0]["token_distance"])

    profiles = _load_cluster_profiles(metadata)
    states = _load_states(metadata)
    profile = profiles.get(token_id, {})
    similar = _similar_windows(embedding[0], states=states, top_k=top_k)
    phase_candidates, phase_support = _phase_candidates_from_token(
        token_id=token_id,
        states=states,
        profiles=profiles,
        horizon_steps=horizon_steps,
        top_k=top_k,
    )

    embedding_payload = {f"embedding_{idx:02d}": float(value) for idx, value in enumerate(embedding[0])}
    feature_payload = feature_row.iloc[0].to_dict()
    missing_features = [column for column in feature_columns if column not in feature_row.columns]

    result = {
        "station": station,
        "source_csv": str(live_csv),
        "metadata_path": str(metadata_path),
        "semantic_run_name": metadata.get("run_name"),
        "live_window": diagnostics,
        "adapter_warning": (
            "AviationWeather METAR is usually hourly, while the historical semantic build may be based on "
            f"{metadata.get('window_size')} higher-frequency rows. Schema is matched, but sampling frequency can differ."
        ),
        "live_state": {
            "window_id": feature_payload.get("window_id"),
            "window_start": feature_payload.get("window_start"),
            "window_end": feature_payload.get("window_end"),
            "window_rows": feature_payload.get("window_rows"),
            "time_span_hours": feature_payload.get("time_span_hours"),
            "token_id": token_id,
            "token_distance": token_distance,
            "regime_name": profile.get("regime_name") or f"token {token_id}",
            "short_explanation": profile.get("short_explanation", ""),
            "meteorological_interpretation": profile.get("meteorological_interpretation", ""),
        },
        "features": {column: feature_payload.get(column) for column in feature_columns},
        "missing_features_filled_by_encoder": missing_features,
        "embedding": embedding_payload,
        "similar_historical_windows": similar,
        "phase_prediction": {
            "horizon_steps": int(horizon_steps),
            "candidate_next_phases": phase_candidates,
            "support": int(phase_support),
            "transition_source": "live_token_markov",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{station}_live_semantic_state.json"
    csv_path = output_dir / f"{station}_live_semantic_state.csv"
    history_path = output_dir / f"{station}_live_semantic_history.csv"
    json_path.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    flat_row = {
        **{key: result["live_state"].get(key) for key in result["live_state"]},
        **result["features"],
        **result["embedding"],
        "phase_support": phase_support,
        "top_phase_token": phase_candidates[0]["token_id"] if phase_candidates else None,
        "top_phase_probability": phase_candidates[0]["probability"] if phase_candidates else None,
    }
    pd.DataFrame([flat_row]).to_csv(csv_path, index=False)

    if not feature_history.empty:
        history_matrix = feature_history.drop(columns=list(META_COLUMNS), errors="ignore").reindex(columns=feature_columns)
        history_embedding = encoder.transform(history_matrix)
        history_token_frame = tokenizer.transform(history_embedding).reset_index(drop=True)
        history_embedding_frame = pd.DataFrame(
            history_embedding,
            columns=[f"embedding_{idx:02d}" for idx in range(history_embedding.shape[1])],
        )
        history_state = pd.concat(
            [
                feature_history.reset_index(drop=True),
                history_embedding_frame,
                history_token_frame,
            ],
            axis=1,
        )
        history_state["regime_name"] = history_state["token_id"].map(
            lambda value: profiles.get(int(value), {}).get("regime_name", f"token {int(value)}")
        )
        history_state.to_csv(history_path, index=False)
    else:
        pd.DataFrame().to_csv(history_path, index=False)

    result["output_paths"] = {"json": str(json_path), "csv": str(csv_path), "history_csv": str(history_path)}
    return _json_safe(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build a live semantic wind state from AviationWeather METAR rows.")
    parser.add_argument("--live_csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--station", default="KBOS")
    parser.add_argument("--live_window_rows", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--phase_horizon_steps", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/live/semantic_states"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_live_semantic_state(
        live_csv=args.live_csv,
        metadata_path=args.metadata,
        station=args.station.upper(),
        live_window_rows=args.live_window_rows,
        top_k=args.top_k,
        horizon_steps=args.phase_horizon_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
