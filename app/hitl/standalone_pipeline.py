from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.hitl.feedback_store import FeedbackRecord, FeedbackStore
from app.hitl.hitl_model_artifacts import predict_numeric_lstm, predict_phase_gru
from app.hitl.semantic_review import load_semantic_metadata
from app.llm.semantic_reasoner import SemanticReasoner
from app.phase_forecasting.semantic_phase_forecaster import PhaseForecastConfig, SemanticPhaseForecaster
from app.semantic.retrieval.similar_regime_search import SimilarRegimeSearcher


SUPPORTED_ACTIONS = {"accept", "flag", "note"}


def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


def _clean_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


@dataclass
class FilterDecision:
    intent: str
    confidence: float
    allowed: bool
    reason: str
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HITLQuestionFilter:
    """A deterministic safety gate so unrelated questions never reach the LLM."""

    domain_terms = {
        "wind",
        "forecast",
        "prediction",
        "numeric",
        "numerical",
        "value",
        "values",
        "raw",
        "metar",
        "speed",
        "gust",
        "direction",
        "regime",
        "state",
        "token",
        "embedding",
        "compressed",
        "compression",
        "similar",
        "history",
        "window",
        "model",
        "lstm",
        "persistence",
        "phase",
        "sequence",
        "subsequence",
        "probability",
        "distribution",
        "flag",
        "accept",
        "approve",
        "reject",
        "note",
    }
    intent_terms = {
        "explain_state": {
            "regime",
            "state",
            "token",
            "embedding",
            "compressed",
            "compression",
            "window",
            "explain",
        },
        "retrieve_similar": {
            "similar",
            "history",
            "historical",
            "past",
            "retrieve",
            "nearest",
            "match",
            "seen",
            "before",
        },
        "phase_prediction": {
            "phase",
            "sequence",
            "subsequence",
            "transition",
            "probability",
            "distribution",
            "token",
            "next",
            "predict",
            "forecast",
        },
        "feedback": {
            "accept",
            "approve",
            "flag",
            "wrong",
            "incorrect",
            "reject",
            "correct",
            "note",
        },
    }
    irrelevant_terms = {
        "recipe",
        "movie",
        "song",
        "lyrics",
        "capital",
        "president",
        "football",
        "stock",
        "crypto",
        "medical",
        "doctor",
        "legal",
        "lawyer",
        "homework",
    }

    def __init__(self, threshold: float = 0.45):
        self.threshold = float(threshold)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def classify(self, question: str, forced_action: str | None = None) -> FilterDecision:
        terms = self._terms(question)
        if forced_action:
            terms.add(str(forced_action).lower())

        domain_matches = sorted(terms & self.domain_terms)
        irrelevant_matches = sorted(terms & self.irrelevant_terms)

        intent_scores: dict[str, int] = {
            intent: len(terms & keywords)
            for intent, keywords in self.intent_terms.items()
        }
        feedback_matches = sorted(terms & self.intent_terms["feedback"])
        phase_matches = sorted(terms & self.intent_terms["phase_prediction"])
        retrieval_matches = sorted(terms & self.intent_terms["retrieve_similar"])
        if feedback_matches:
            intent = "feedback"
        elif phase_matches:
            intent = "phase_prediction"
        elif retrieval_matches and {"similar", "history", "historical", "past", "retrieve", "nearest", "match", "seen", "before"} & terms:
            intent = "retrieve_similar"
        else:
            intent = max(intent_scores, key=intent_scores.get)
        intent_matches = sorted(terms & self.intent_terms[intent])

        if forced_action:
            intent = "feedback"
            intent_matches = sorted(set(intent_matches + [forced_action]))

        if irrelevant_matches and not domain_matches:
            return FilterDecision(
                intent="out_of_scope",
                confidence=0.05,
                allowed=False,
                reason="Question matched unrelated-topic terms and no wind-forecasting context.",
                matched_terms=irrelevant_matches,
            )

        domain_score = min(len(domain_matches) / 3.0, 1.0) * 0.45
        intent_score = min(len(intent_matches) / 2.0, 1.0) * 0.45
        generic_explain_bonus = 0.10 if "explain" in terms and domain_matches else 0.0
        confidence = min(domain_score + intent_score + generic_explain_bonus, 1.0)
        if intent == "feedback" and feedback_matches:
            confidence = max(confidence, 0.65)

        if not domain_matches and intent not in {"feedback"}:
            confidence = min(confidence, 0.25)

        allowed = confidence >= self.threshold
        reason = "Question is relevant to wind forecast review." if allowed else (
            "Question confidence is below the HITL relevance threshold."
        )
        return FilterDecision(
            intent=intent if allowed else "out_of_scope",
            confidence=round(float(confidence), 3),
            allowed=allowed,
            reason=reason,
            matched_terms=sorted(set(domain_matches + intent_matches)),
        )


@dataclass
class SemanticHITLContext:
    metadata_path: Path
    window_id: str | None = None
    latest: bool = False
    top_k: int = 5

    def __post_init__(self) -> None:
        self.metadata = load_semantic_metadata(self.metadata_path)
        self.semantic_states = pd.read_csv(self.metadata["output_paths"]["semantic_states"])
        self.cluster_profiles = pd.read_csv(self.metadata["output_paths"]["cluster_profiles"])
        self.searcher = SimilarRegimeSearcher.load(self.metadata["output_paths"]["search_index"])
        if self.searcher.state_frame is not None:
            label_columns = [
                "window_id",
                "regime_name",
                "short_explanation",
                "meteorological_interpretation",
            ]
            available = [column for column in label_columns if column in self.semantic_states.columns]
            if "window_id" in available:
                labels = self.semantic_states[available].drop_duplicates("window_id")
                base = self.searcher.state_frame.drop(
                    columns=[column for column in available if column != "window_id"],
                    errors="ignore",
                )
                self.searcher.state_frame = base.merge(labels, on="window_id", how="left")

        if self.semantic_states.empty:
            raise ValueError("Semantic state file is empty.")

        if self.window_id:
            resolved = self.window_id
        elif self.latest:
            resolved = str(self.semantic_states.iloc[-1]["window_id"])
        else:
            raise ValueError("Provide a window_id or set latest=True.")

        state_rows = self.semantic_states[self.semantic_states["window_id"].astype(str) == str(resolved)]
        if state_rows.empty:
            raise ValueError(f"Window id '{resolved}' was not found.")

        self.resolved_window_id = str(resolved)
        self.query_state = state_rows.iloc[0].to_dict()
        self.token_id = int(self.query_state.get("token_id", -1))
        profile_rows = self.cluster_profiles[self.cluster_profiles["token_id"].astype(int) == self.token_id]
        self.cluster_profile = profile_rows.iloc[0].to_dict() if not profile_rows.empty else {}

    def retrieve_neighbors(self) -> list[dict[str, Any]]:
        neighbors = self.searcher.query_by_window_id(
            self.resolved_window_id,
            top_k=self.top_k,
        )
        return neighbors.to_dict(orient="records")

    def compact_state(self) -> dict[str, Any]:
        keys = [
            "window_id",
            "window_start",
            "window_end",
            "token_id",
            "token_distance",
            "regime_name",
            "short_explanation",
            "meteorological_interpretation",
            "wind_speed_mean",
            "wind_speed_std",
            "wind_speed_min",
            "wind_speed_max",
            "ramp_abs_mean",
            "ramp_abs_max",
            "gust_factor",
            "direction_abs_change_mean_deg",
            "direction_net_turn_deg",
        ]
        return {key: self.query_state.get(key) for key in keys if key in self.query_state}


def _compact_neighbor(row: dict[str, Any], idx: int) -> dict[str, Any]:
    keys = [
        "window_id",
        "window_start",
        "window_end",
        "retrieval_distance",
        "token_id",
        "regime_name",
        "wind_speed_mean",
        "wind_speed_std",
        "wind_speed_min",
        "wind_speed_max",
        "ramp_abs_max",
        "gust_factor",
        "direction_abs_change_mean_deg",
    ]
    payload = {key: row.get(key) for key in keys if key in row}
    payload["evidence_id"] = f"similar_{idx}"
    return payload


def _compact_feedback(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "evidence_id": f"feedback_{idx}",
        "created_at": row.get("created_at"),
        "reviewer": row.get("reviewer"),
        "action": row.get("action"),
        "label": row.get("label"),
        "note": row.get("note"),
    }


def load_forecast_frame(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("forecast", payload.get("rows", payload))
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(path)

    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"])
    if "wind_speed" not in frame.columns:
        raise ValueError("Forecast file must include a 'wind_speed' column.")
    frame["wind_speed"] = pd.to_numeric(frame["wind_speed"], errors="coerce")
    return frame.dropna(subset=["wind_speed"]).reset_index(drop=True)


def summarize_forecast(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "available": False,
            "message": "No forecast artifact was supplied to the HITL pipeline.",
        }

    speeds = frame["wind_speed"].to_numpy(dtype=float)
    start_speed = _clean_float(speeds[0])
    end_speed = _clean_float(speeds[-1])
    delta = end_speed - start_speed
    if abs(delta) < 0.25:
        trend = "mostly stable"
    elif delta > 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    peak_idx = int(np.nanargmax(speeds))
    low_idx = int(np.nanargmin(speeds))
    start_time = str(frame["datetime"].iloc[0]) if "datetime" in frame.columns else ""
    end_time = str(frame["datetime"].iloc[-1]) if "datetime" in frame.columns else ""
    peak_time = str(frame["datetime"].iloc[peak_idx]) if "datetime" in frame.columns else ""
    low_time = str(frame["datetime"].iloc[low_idx]) if "datetime" in frame.columns else ""

    return {
        "available": True,
        "steps": int(len(frame)),
        "start_time": start_time,
        "end_time": end_time,
        "start_wind_speed": round(start_speed, 3),
        "end_wind_speed": round(end_speed, 3),
        "mean_wind_speed": round(float(np.nanmean(speeds)), 3),
        "min_wind_speed": round(float(np.nanmin(speeds)), 3),
        "max_wind_speed": round(float(np.nanmax(speeds)), 3),
        "delta": round(float(delta), 3),
        "trend": trend,
        "peak_time": peak_time,
        "low_time": low_time,
    }


def _state_dict_explanation_text(
    state: dict[str, Any],
    default_window_id: str,
    subject: str = "current compressed wind state",
) -> str:
    token = state.get("token_id")
    name = state.get("regime_name") or (f"token {token}" if token is not None else "the current regime")
    window_id = state.get("window_id") or default_window_id
    mean_speed = _clean_float(state.get("wind_speed_mean"))
    spread = _clean_float(state.get("wind_speed_std"))
    ramp = _clean_float(state.get("ramp_abs_max"))
    gust = _clean_float(state.get("gust_factor"), default=1.0)
    direction_change = _clean_float(state.get("direction_abs_change_mean_deg"))
    token_text = f" (token {int(token)})" if token is not None and str(token) != "" else ""

    return (
        f"The {subject} is {name}{token_text}. "
        f"It represents window {window_id}, with mean wind speed around "
        f"{mean_speed:.2f} m/s, standard deviation {spread:.2f}, maximum ramp {ramp:.2f}, "
        f"gust factor {gust:.2f}, and average directional change {direction_change:.1f} degrees. "
        "This explanation is based on the numeric semantic state, not a direct LLM forecast."
    )


def _state_explanation_text(context: SemanticHITLContext) -> str:
    return _state_dict_explanation_text(
        context.compact_state(),
        default_window_id=context.resolved_window_id,
        subject="current compressed wind state",
    )


def _retrieval_text(neighbors: list[dict[str, Any]]) -> str:
    if not neighbors:
        return "No similar historical wind states were retrieved."

    first = neighbors[0]
    regime = first.get("regime_name") or f"token {first.get('token_id')}"
    distance = _clean_float(first.get("retrieval_distance"))
    return (
        f"The nearest historical match is {first.get('window_id')} "
        f"({regime}) with retrieval distance {distance:.3f}. "
        f"{len(neighbors)} similar windows were returned for human inspection."
    )


def _rank_phase_counts(
    counts: Counter[int],
    support: int,
    source: str,
    cluster_profiles: pd.DataFrame,
    top_k: int,
) -> list[dict[str, Any]]:
    if not support:
        return []

    rows: list[dict[str, Any]] = []
    for rank, (token_id, count) in enumerate(counts.most_common(max(1, int(top_k))), start=1):
        profile_rows = cluster_profiles[
            pd.to_numeric(cluster_profiles["token_id"], errors="coerce") == int(token_id)
        ] if "token_id" in cluster_profiles.columns else pd.DataFrame()
        profile = profile_rows.iloc[0].to_dict() if not profile_rows.empty else {}
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "regime_name": str(profile.get("regime_name") or f"token {token_id}"),
                "probability": round(float(count / support), 4),
                "count": int(count),
                "support": int(support),
                "transition_source": source,
            }
        )
    return rows


def _historical_transition_counts(
    historical_tokens: np.ndarray,
    sequence: list[int],
    horizon_steps: int,
    min_support: int,
) -> tuple[str, Counter[int], int]:
    horizon = max(1, int(horizon_steps))
    threshold = max(1, int(min_support))
    clean_sequence = [int(token) for token in sequence]
    fallback: tuple[str, Counter[int], int] | None = None

    for length in range(len(clean_sequence), 1, -1):
        suffix = clean_sequence[-length:]
        counts: Counter[int] = Counter()
        for end_idx in range(length - 1, len(historical_tokens) - horizon):
            past = [int(x) for x in historical_tokens[end_idx - length + 1 : end_idx + 1]]
            if past == suffix:
                counts[int(historical_tokens[end_idx + horizon])] += 1
        support = sum(counts.values())
        if support:
            source = "live_exact_sequence" if length == len(clean_sequence) else f"live_suffix_{length}_tokens"
            if fallback is None:
                fallback = (f"{source}_low_support", counts, support)
            if support >= threshold:
                return source, counts, support

    if clean_sequence:
        last_token = int(clean_sequence[-1])
        counts = Counter()
        for idx in range(0, len(historical_tokens) - horizon):
            if int(historical_tokens[idx]) == last_token:
                counts[int(historical_tokens[idx + horizon])] += 1
        support = sum(counts.values())
        if support >= threshold:
            return "live_last_token_markov", counts, support
        if fallback is not None:
            return fallback
        return "live_last_token_markov_low_support", counts, support

    return "no_live_sequence", Counter(), 0


def _parse_forecast_steps(question: str, default: int = 2, maximum: int = 12) -> int:
    lowered = question.lower()
    match = re.search(r"(?:next|forecast|predict)\s+(\d+)", lowered)
    if not match:
        match = re.search(r"(\d+)\s*(?:step|steps|window|windows|hour|hours)", lowered)
    if not match:
        return default
    try:
        value = int(match.group(1))
    except Exception:
        return default
    return max(1, min(value, maximum))


def _load_live_raw_wind(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Live raw wind CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Live raw wind CSV is empty: {path}")
    time_column = "observation_time_utc" if "observation_time_utc" in frame.columns else "datetime"
    if time_column not in frame.columns:
        raise ValueError("Live raw wind CSV needs observation_time_utc or datetime.")
    if "wind_speed" not in frame.columns:
        raise ValueError("Live raw wind CSV needs wind_speed in m/s.")
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    frame["wind_speed"] = pd.to_numeric(frame["wind_speed"], errors="coerce")
    if "wind_direction" in frame.columns:
        frame["wind_direction"] = pd.to_numeric(frame["wind_direction"], errors="coerce")
    if "wind_gust_10m_ms" in frame.columns:
        frame["wind_gust_10m_ms"] = pd.to_numeric(frame["wind_gust_10m_ms"], errors="coerce")
    frame = frame.dropna(subset=[time_column, "wind_speed"]).sort_values(time_column)
    frame = frame.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)
    return frame.rename(columns={time_column: "datetime"})


def _raw_numeric_forecast(frame: pd.DataFrame, steps: int) -> dict[str, Any]:
    speeds = frame["wind_speed"].astype(float).to_numpy()
    times = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    recent_n = int(min(12, len(speeds)))
    recent = speeds[-recent_n:]
    last_speed = float(recent[-1])

    if len(times) >= 2:
        deltas_seconds = times.diff().dropna().dt.total_seconds()
        positive = deltas_seconds[deltas_seconds > 0]
        step_seconds = float(positive.median()) if not positive.empty else 3600.0
    else:
        step_seconds = 3600.0

    if len(recent) >= 3:
        diffs = np.diff(recent)
        weights = np.linspace(0.3, 1.0, len(diffs))
        trend = float(np.average(diffs, weights=weights))
        residuals = speeds[1:] - speeds[:-1]
        recent_residuals = residuals[-min(24, len(residuals)):] if len(residuals) else np.array([0.0])
        residual_std = float(np.std(recent_residuals)) if len(recent_residuals) else 0.0
        clip = max(0.5, residual_std * 1.5)
        trend = float(np.clip(trend, -clip, clip))
        method = "raw_recent_damped_trend"
    else:
        trend = 0.0
        residual_std = 0.0
        method = "raw_persistence"

    predictions: list[dict[str, Any]] = []
    running = last_speed
    last_time = times.iloc[-1]
    for step in range(1, steps + 1):
        running = max(0.0, running + trend * (0.70 ** (step - 1)))
        timestamp = last_time + pd.to_timedelta(step_seconds * step, unit="s")
        uncertainty = max(0.35, residual_std * (step ** 0.5))
        predictions.append(
            {
                "step": step,
                "datetime": timestamp.isoformat(),
                "wind_speed": round(float(running), 3),
                "lower": round(max(0.0, float(running - uncertainty)), 3),
                "upper": round(float(running + uncertainty), 3),
                "uncertainty": round(float(uncertainty), 3),
            }
        )

    latest = frame.iloc[-1].to_dict()
    context_columns = [
        column
        for column in ["datetime", "wind_speed", "wind_direction", "wind_gust_10m_ms"]
        if column in frame.columns
    ]
    recent_context = frame.tail(min(8, len(frame)))[context_columns].copy()
    recent_context["datetime"] = recent_context["datetime"].astype(str)
    sketch_values = np.concatenate(
        [
            speeds[-min(6, len(speeds)) :],
            np.array([row["wind_speed"] for row in predictions], dtype=float),
        ]
    )
    sketch_diffs = np.diff(sketch_values) if len(sketch_values) > 1 else np.array([0.0])
    predicted_window_sketch = {
        "evidence_id": "predicted_numeric_window",
        "construction": "recent_observed_raw_values_plus_numeric_forecast",
        "observed_points_used": int(min(6, len(speeds))),
        "forecast_points_used": int(len(predictions)),
        "window_rows": int(len(sketch_values)),
        "wind_speed_mean": round(float(np.mean(sketch_values)), 6),
        "wind_speed_std": round(float(np.std(sketch_values)), 6),
        "wind_speed_min": round(float(np.min(sketch_values)), 6),
        "wind_speed_max": round(float(np.max(sketch_values)), 6),
        "wind_speed_p10": round(float(np.quantile(sketch_values, 0.10)), 6),
        "wind_speed_p90": round(float(np.quantile(sketch_values, 0.90)), 6),
        "wind_speed_range": round(float(np.max(sketch_values) - np.min(sketch_values)), 6),
        "ramp_abs_mean": round(float(np.mean(np.abs(sketch_diffs))), 6),
        "ramp_abs_max": round(float(np.max(np.abs(sketch_diffs))), 6),
    }
    return {
        "method": method,
        "steps": int(steps),
        "source_rows": int(len(frame)),
        "recent_rows_used": int(recent_n),
        "last_observation": {
            "datetime": str(latest.get("datetime")),
            "wind_speed": round(_clean_float(latest.get("wind_speed")), 3),
            "wind_direction": None if pd.isna(latest.get("wind_direction")) else int(latest.get("wind_direction")),
            "wind_gust_10m_ms": None
            if pd.isna(latest.get("wind_gust_10m_ms"))
            else round(_clean_float(latest.get("wind_gust_10m_ms")), 3),
        },
        "trend_per_step": round(float(trend), 3),
        "predictions": predictions,
        "predicted_window_sketch": predicted_window_sketch,
        "recent_context": recent_context.to_dict(orient="records"),
        "grounding_rules": [
            "This numeric forecast uses raw live wind-speed observations, not semantic tokens or compressed windows.",
            "It is a short-horizon deterministic recent-trend baseline for HITL review.",
            "Prediction intervals are heuristic uncertainty bands from recent one-step variation.",
        ],
    }


class StandaloneHITLPipeline:
    """Standalone HITL layer, deliberately separate from the old app agents."""

    def __init__(
        self,
        metadata_path: str | Path,
        window_id: str | None = None,
        latest: bool = False,
        forecast_path: str | Path | None = None,
        top_k: int = 5,
        phase_tokens: list[int] | None = None,
        phase_history_length: int = 2,
        phase_horizon_steps: int = 1,
        phase_top_k: int = 3,
        phase_analog_k: int = 5,
        phase_min_support: int = 5,
        live_phase_history_path: str | Path | None = None,
        live_phase_state_path: str | Path | None = None,
        live_raw_path: str | Path | None = None,
        numeric_forecast_steps: int = 6,
        numeric_forecast_mode: str = "auto",
        numeric_model_path: str | Path | None = None,
        phase_model_mode: str = "auto",
        phase_model_path: str | Path | None = None,
        prefer_live_phase: bool = True,
        enable_rag: bool = True,
        enable_llm: bool = False,
        enable_agent_graph: bool = False,
        confidence_threshold: float = 0.45,
        feedback_path: str | Path | None = None,
    ):
        self.context = SemanticHITLContext(
            metadata_path=Path(metadata_path),
            window_id=window_id,
            latest=latest,
            top_k=top_k,
        )
        self.forecast_frame = load_forecast_frame(forecast_path)
        self.forecast_summary = summarize_forecast(self.forecast_frame)
        self.phase_tokens = [int(token) for token in phase_tokens] if phase_tokens else None
        self.phase_config = PhaseForecastConfig(
            history_length=phase_history_length,
            horizon_steps=phase_horizon_steps,
            top_k=phase_top_k,
            analog_k=phase_analog_k,
            min_support=phase_min_support,
            enable_llm=enable_llm,
        )
        project_root = self.context.metadata_path.parent.parent.parent
        self.project_root = project_root
        self.live_phase_history_path = Path(live_phase_history_path) if live_phase_history_path else (
            project_root / "data" / "live" / "semantic_states" / "KBOS_live_semantic_history.csv"
        )
        self.live_phase_state_path = Path(live_phase_state_path) if live_phase_state_path else (
            project_root / "data" / "live" / "semantic_states" / "KBOS_live_semantic_state.json"
        )
        default_local_live_raw = project_root / "data" / "live" / "aviationweather_metar" / "KBOS" / "metar_live.csv"
        self.live_raw_path = Path(live_raw_path) if live_raw_path else (
            default_local_live_raw
        )
        self.numeric_forecast_steps = max(1, min(int(numeric_forecast_steps), 12))
        self.numeric_forecast_mode = str(numeric_forecast_mode or "auto").lower()
        if self.numeric_forecast_mode not in {"disabled", "auto", "lstm", "trend"}:
            self.numeric_forecast_mode = "disabled"
        self.numeric_model_path = Path(numeric_model_path) if numeric_model_path else (
            project_root / "artifacts" / "hitl_numeric_lstm" / "kbos_hourly_v1"
        )
        self.phase_model_mode = str(phase_model_mode or "auto").lower()
        if self.phase_model_mode not in {"auto", "gru", "transition"}:
            self.phase_model_mode = "auto"
        self.phase_model_path = Path(phase_model_path) if phase_model_path else (
            project_root / "artifacts" / "hitl_phase_gru" / "kbos_hourly_v1"
        )
        self.prefer_live_phase = bool(prefer_live_phase)
        self.filter = HITLQuestionFilter(threshold=confidence_threshold)
        self.feedback_store = FeedbackStore(feedback_path)
        self.enable_rag = bool(enable_rag)
        self.enable_llm = bool(enable_llm)
        self.enable_agent_graph = bool(enable_agent_graph)
        self.reasoner = SemanticReasoner() if enable_llm else None
        self.live_phase_refresh_warning: str | None = None

    @property
    def llm_available(self) -> bool:
        return bool(self.reasoner and self.reasoner.available)

    def build_evidence_pack(self, include_neighbors: bool = True) -> dict[str, Any]:
        current_state = self._current_live_compact_state() if self.prefer_live_phase else {}
        if not current_state:
            current_state = self.context.compact_state()
        current_state["evidence_id"] = "current_state"
        current_token = _clean_int(current_state.get("token_id"), default=self.context.token_id)
        profile_rows = self.context.cluster_profiles[
            pd.to_numeric(self.context.cluster_profiles["token_id"], errors="coerce") == current_token
        ] if "token_id" in self.context.cluster_profiles.columns else pd.DataFrame()
        cluster_profile = (
            profile_rows.iloc[0].to_dict()
            if not profile_rows.empty
            else dict(self.context.cluster_profile)
        )
        cluster_profile["evidence_id"] = "cluster_profile"
        forecast_summary = dict(self.forecast_summary)
        forecast_summary["evidence_id"] = "forecast_summary"

        neighbors: list[dict[str, Any]] = []
        if self.enable_rag and include_neighbors:
            live_neighbors = self._current_live_neighbors() if self.prefer_live_phase else []
            source_neighbors = live_neighbors or self.context.retrieve_neighbors()
            neighbors = [
                _compact_neighbor(row, idx + 1)
                for idx, row in enumerate(source_neighbors)
            ]

        feedback_window_id = str(current_state.get("window_id") or self.context.resolved_window_id)
        feedback_rows = self.feedback_store.filter(
            artifact_type="semantic_window",
            artifact_id=feedback_window_id,
        )
        feedback = [
            _compact_feedback(row, idx + 1)
            for idx, row in enumerate(feedback_rows[-5:])
        ]

        return {
            "current_state": current_state,
            "cluster_profile": cluster_profile,
            "forecast_summary": forecast_summary,
            "similar_windows": neighbors,
            "human_feedback": feedback,
            "grounding_rules": [
                "Use only these evidence records.",
                "Do not infer exact meteorology that is not supported by fields in the evidence.",
                "If evidence is missing, state that the evidence is insufficient.",
                "Mention evidence IDs used in the answer.",
            ],
        }

    def _llm_json(self, task: str, question: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.llm_available:
            return None

        prompt = (
            "You are the standalone HITL explanation layer for a wind forecasting system.\n"
            "Only answer using the provided structured wind-forecasting context.\n"
            "Do not answer unrelated general questions.\n"
            "If evidence is insufficient, say so rather than guessing.\n"
            "Cite the evidence_id values you used inside the evidence field.\n"
            "Return only valid JSON with keys: answer, evidence, human_review_prompt.\n"
            "For live_phase_prediction or phase_prediction, write 4-6 clear sentences explaining: "
            "the live/current token sequence, the predicted next phase, probability/support, alternative candidates, "
            "and what the prediction does not mean. Do not invent numeric wind-speed forecasts.\n\n"
            f"Task: {task}\n"
            f"User question: {question}\n"
            f"Context:\n{json.dumps(_json_safe(payload), ensure_ascii=False)}\n"
        )
        return self.reasoner.invoke_json(prompt)

    def explain_forecast(self, question: str) -> dict[str, Any]:
        payload = self.build_evidence_pack(include_neighbors=True)
        llm_payload = self._llm_json("explain_forecast", question, payload)
        if llm_payload:
            return {"mode": "llm_rag", **llm_payload, "evidence_pack": payload}

        if not self.forecast_summary.get("available"):
            answer = (
                "No forecast artifact was provided, so I can only explain the current compressed wind state. "
                + _state_explanation_text(self.context)
            )
        else:
            fs = self.forecast_summary
            answer = (
                f"The supplied forecast is {fs['trend']} over {fs['steps']} steps, "
                f"moving from {fs['start_wind_speed']} m/s to {fs['end_wind_speed']} m/s. "
                f"The mean forecast speed is {fs['mean_wind_speed']} m/s, with a peak of "
                f"{fs['max_wind_speed']} m/s near {fs['peak_time']}. "
                + _state_explanation_text(self.context)
            )

        return {
            "mode": "heuristic",
            "answer": answer,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": "Check whether the trend and semantic regime agree with your domain expectation.",
        }

    def explain_state(self, question: str, window_context: str = "current_window") -> dict[str, Any]:
        if window_context != "current_window":
            return {
                "mode": "explain_window",
                "explanation_context": window_context,
                "answer": f"Window explanation context '{window_context}' must be resolved by the agent graph.",
                "evidence": {},
                "evidence_pack": {},
                "human_review_prompt": "Use the agent graph so session memory can resolve non-current windows.",
            }
        payload = self.build_evidence_pack(include_neighbors=False)
        llm_payload = self._llm_json("explain_state", question, payload)
        if llm_payload:
            return {
                "mode": "explain_window",
                "explanation_context": "current_window",
                "explanation_mode": "llm_rag",
                **llm_payload,
                "evidence_pack": payload,
            }
        current_state = payload.get("current_state", {}) if isinstance(payload, dict) else {}
        subject = (
            "current live semantic wind state"
            if isinstance(current_state, dict) and current_state.get("semantic_state_source") == "live_phase_state"
            else "current compressed wind state"
        )
        answer = _state_dict_explanation_text(
            current_state if isinstance(current_state, dict) else self.context.compact_state(),
            default_window_id=self.context.resolved_window_id,
            subject=subject,
        )
        return {
            "mode": "explain_window",
            "explanation_context": "current_window",
            "explanation_mode": "heuristic",
            "answer": answer,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": "Accept, flag, or note this regime if the description is not useful.",
        }

    def retrieve_similar(self, question: str, window_context: str = "current_window") -> dict[str, Any]:
        if window_context != "current_window":
            return {
                "mode": "retrieve_similar",
                "retrieval_context": window_context,
                "explanation_mode": "missing_context",
                "answer": f"Retrieval context '{window_context}' must be resolved by the agent graph.",
                "similar_windows": [],
                "evidence": {},
                "evidence_pack": {},
                "human_review_prompt": "Use the agent graph so session memory can resolve non-current windows.",
            }
        live_neighbors = self._current_live_neighbors() if self.prefer_live_phase else []
        neighbors = live_neighbors or self.context.retrieve_neighbors()
        payload = self.build_evidence_pack(include_neighbors=True)
        llm_payload = self._llm_json("retrieve_similar", question, payload)
        if llm_payload:
            return {
                "mode": "retrieve_similar",
                "retrieval_context": "current_window",
                "explanation_mode": "llm_rag",
                **llm_payload,
                "similar_windows": _json_safe(neighbors),
                "evidence_pack": payload,
            }
        return {
            "mode": "retrieve_similar",
            "retrieval_context": "current_window",
            "explanation_mode": "heuristic",
            "answer": _retrieval_text(neighbors),
            "evidence": payload,
            "similar_windows": _json_safe(neighbors),
            "evidence_pack": payload,
            "human_review_prompt": "Inspect the nearest windows and flag any retrieved match that is not physically similar.",
        }

    def predict_numeric_wind(self, question: str) -> dict[str, Any]:
        steps = _parse_forecast_steps(question, default=self.numeric_forecast_steps, maximum=12)
        try:
            frame = _load_live_raw_wind(self.live_raw_path)
            model_warning = None
            if self.numeric_forecast_mode in {"auto", "lstm"}:
                try:
                    forecast = predict_numeric_lstm(
                        frame=frame,
                        artifact_dir=self.numeric_model_path,
                        steps=steps,
                    )
                except Exception as exc:
                    if self.numeric_forecast_mode == "lstm":
                        raise
                    model_warning = f"Numeric LSTM unavailable; used recent-trend fallback: {exc}"
                    forecast = _raw_numeric_forecast(frame, steps=steps)
                    forecast["forecast_model_source"] = "recent_trend_fallback"
                    forecast["model_artifact_path"] = str(self.numeric_model_path)
                    forecast["model_warning"] = model_warning
            else:
                forecast = _raw_numeric_forecast(frame, steps=steps)
                forecast["forecast_model_source"] = "recent_trend_fallback"
                forecast["model_artifact_path"] = str(self.numeric_model_path)
                forecast["model_warning"] = "Numeric forecast mode is set to trend."
        except Exception as exc:
            payload = self.build_evidence_pack(include_neighbors=False)
            payload["numeric_forecast_error"] = {
                "live_raw_path": str(self.live_raw_path),
                "numeric_forecast_mode": self.numeric_forecast_mode,
                "model_artifact_path": str(self.numeric_model_path),
                "error": str(exc),
                "evidence_id": "numeric_forecast_error",
            }
            return {
                "mode": "numeric_forecast_error",
                "answer": f"Could not run raw numeric wind forecast: {exc}",
                "numeric_forecast": payload["numeric_forecast_error"],
                "evidence": payload,
                "evidence_pack": payload,
                "human_review_prompt": "Check whether the live raw METAR CSV path exists and contains wind_speed.",
            }

        payload = self.build_evidence_pack(include_neighbors=False)
        forecast["live_raw_path"] = str(self.live_raw_path)
        forecast["evidence_id"] = "raw_numeric_forecast"
        payload["numeric_forecast"] = forecast

        forecast_text = "; ".join(
            f"step {row['step']} at {row['datetime']}: {row['wind_speed']} m/s "
            f"(band {row['lower']}-{row['upper']})"
            for row in forecast["predictions"]
        )
        answer = (
            "Raw numeric wind-speed forecast from latest live METAR observations: "
            f"{forecast_text}. Last observed speed was {forecast['last_observation']['wind_speed']} m/s at "
            f"{forecast['last_observation']['datetime']}. Method: {forecast['method']}; "
            f"source: {forecast.get('forecast_model_source', forecast['method'])}; this is not a semantic "
            "phase forecast."
        )
        if forecast.get("model_warning"):
            answer = f"{answer} Warning: {forecast['model_warning']}"
        llm_payload = self._llm_json("numeric_forecast", question, payload)
        if llm_payload:
            return {
                "mode": "llm_rag_numeric_forecast",
                **llm_payload,
                "numeric_forecast": forecast,
                "forecast_model_source": forecast.get("forecast_model_source"),
                "model_artifact_path": forecast.get("model_artifact_path"),
                "model_warning": forecast.get("model_warning"),
                "evidence_pack": payload,
            }
        return {
            "mode": "raw_numeric_forecast",
            "answer": answer,
            "numeric_forecast": forecast,
            "forecast_model_source": forecast.get("forecast_model_source"),
            "model_artifact_path": forecast.get("model_artifact_path"),
            "model_warning": forecast.get("model_warning"),
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": "Review whether the short-term numeric forecast is consistent with current METAR trend.",
        }

    @staticmethod
    def _parse_token_sequence(question: str) -> list[int] | None:
        bracket_match = re.search(r"\[([0-9,\s]+)\]", question)
        if bracket_match:
            source = bracket_match.group(1)
        else:
            label_match = re.search(
                r"(?:tokens?|sequence|subsequence|pattern)\s*(?:=|:|is|are)?\s*([0-9][0-9,\s]+)",
                question,
                flags=re.IGNORECASE,
            )
            if not label_match:
                return None
            source = label_match.group(1)
        numbers = re.findall(r"\b\d+\b", source)
        if len(numbers) < 1:
            return None
        return [int(number) for number in numbers]

    def _load_live_phase_state(self) -> dict[str, Any]:
        if not self.live_phase_state_path.exists():
            return {}
        try:
            payload = json.loads(self.live_phase_state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _current_live_compact_state(self) -> dict[str, Any]:
        """Return the same live semantic state used by phase prediction."""

        self._refresh_live_phase_artifacts()
        payload = self._load_live_phase_state()
        live_state = payload.get("live_state", {}) if isinstance(payload, dict) else {}
        if not isinstance(live_state, dict) or not live_state:
            if not self.live_phase_history_path.exists():
                return {}
            try:
                history = pd.read_csv(self.live_phase_history_path)
                if history.empty:
                    return {}
                if "window_end" in history.columns:
                    history["window_end"] = pd.to_datetime(history["window_end"], errors="coerce", utc=True)
                    history = history.sort_values("window_end").reset_index(drop=True)
                live_state = history.iloc[-1].to_dict()
            except Exception:
                return {}

        features = payload.get("features", {}) if isinstance(payload, dict) else {}
        state = dict(live_state)
        if isinstance(features, dict):
            for key in [
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
            ]:
                if key not in state and key in features:
                    state[key] = features[key]

        live_history = payload.get("live_history", {}) if isinstance(payload, dict) else {}
        if isinstance(live_history, dict):
            state["live_token_sequence"] = live_history.get("live_token_sequence")
        state["semantic_state_source"] = "live_phase_state"
        state["live_phase_state_path"] = str(self.live_phase_state_path)
        state["live_phase_history_path"] = str(self.live_phase_history_path)
        return state

    def _current_live_neighbors(self) -> list[dict[str, Any]]:
        """Return live-state analog windows generated by the live semantic builder."""

        self._refresh_live_phase_artifacts()
        payload = self._load_live_phase_state()
        neighbors = payload.get("similar_historical_windows", []) if isinstance(payload, dict) else []
        if not isinstance(neighbors, list):
            return []
        return [row for row in neighbors if isinstance(row, dict)]

    def _live_phase_artifacts_need_refresh(self) -> bool:
        if not self.live_raw_path.exists():
            return False
        if not self.live_phase_history_path.exists() or not self.live_phase_state_path.exists():
            return True

        state = self._load_live_phase_state()
        state_metadata = Path(str(state.get("metadata_path", ""))) if state.get("metadata_path") else None
        state_run_name = str(state.get("semantic_run_name") or "")
        active_run_name = str(self.context.metadata.get("run_name") or "")
        try:
            metadata_matches = state_metadata is not None and state_metadata.resolve() == self.context.metadata_path.resolve()
            package_metadata_matches = bool(active_run_name and state_run_name == active_run_name)
            if not metadata_matches and not package_metadata_matches:
                return True
        except Exception:
            if not (active_run_name and state_run_name == active_run_name):
                return True

        try:
            raw_mtime = self.live_raw_path.stat().st_mtime
            history_mtime = self.live_phase_history_path.stat().st_mtime
            state_mtime = self.live_phase_state_path.stat().st_mtime
            return raw_mtime > min(history_mtime, state_mtime)
        except Exception:
            return True

    def _refresh_live_phase_artifacts(self) -> None:
        """Rebuild live semantic files with the active HITL metadata/encoder."""

        self.live_phase_refresh_warning = None
        if not self._live_phase_artifacts_need_refresh():
            return

        try:
            from scripts.build_live_semantic_state import build_live_semantic_state

            # Live METAR rows are hourly; keep the live adapter on the same 6-row window
            # used by the synced GitHub live package instead of the historical 5-min window size.
            window_size = int(self.context.metadata.get("live_window_rows") or 6)
            build_live_semantic_state(
                live_csv=self.live_raw_path,
                metadata_path=self.context.metadata_path,
                station="KBOS",
                live_window_rows=max(1, window_size),
                top_k=max(1, int(self.context.top_k)),
                horizon_steps=max(1, int(self.phase_config.horizon_steps)),
                output_dir=self.live_phase_history_path.parent,
            )
        except Exception as exc:
            self.live_phase_refresh_warning = f"Live phase refresh failed; using existing files if available: {exc}"

    def _predict_live_phase(self, question: str) -> dict[str, Any] | None:
        if not self.prefer_live_phase:
            return None

        self._refresh_live_phase_artifacts()
        if not self.live_phase_history_path.exists():
            return None

        live_history = pd.read_csv(self.live_phase_history_path)
        if live_history.empty or "token_id" not in live_history.columns:
            return None
        if "window_end" in live_history.columns:
            live_history["window_end"] = pd.to_datetime(live_history["window_end"], errors="coerce", utc=True)
            live_history = live_history.sort_values("window_end").reset_index(drop=True)
        live_history["token_id"] = pd.to_numeric(live_history["token_id"], errors="coerce")
        live_history = live_history.dropna(subset=["token_id"]).reset_index(drop=True)
        if live_history.empty:
            return None

        phase_model_warning = None
        if self.phase_model_mode in {"auto", "gru"}:
            try:
                evidence = predict_phase_gru(
                    live_history=live_history,
                    artifact_dir=self.phase_model_path,
                    top_k=self.phase_config.top_k,
                )
                candidates = evidence.get("candidate_next_phases", [])
                top = candidates[0] if candidates else {}
                phase_name = str(top.get("regime_name") or "insufficient evidence")
                current_live_state = evidence.get("current_live_state", {})
                current_name = str(current_live_state.get("regime_name") or "the current live regime")
                current_window_id = str(current_live_state.get("window_id") or "the latest live window")
                live_sequence = evidence.get("live_token_sequence", [])
                alternatives = candidates[1:3]
                alt_text = ""
                if alternatives:
                    alt_text = " Other candidates are " + ", ".join(
                        f"{row['regime_name']} ({float(row['probability']):.2f})" for row in alternatives
                    ) + "."
                explanation = (
                    f"The current live window is {current_window_id}, labeled {current_name}. "
                    f"The HITL phase GRU uses the latest live semantic token sequence {live_sequence} and compact "
                    f"state features to estimate the next phase. The top candidate is {phase_name} "
                    f"(token {top.get('token_id')}, score {float(top.get('probability') or 0.0):.2f}).{alt_text} "
                    "These scores are model probabilities over learned semantic tokens, not exact wind-speed values."
                )
                phase_result = {
                    "mode": "hitl_phase_gru",
                    "llm_requested": bool(self.enable_llm),
                    "llm_available": self.llm_available,
                    "phase_forecast": {
                        "phase_forecast": phase_name,
                        "explanation": explanation,
                        "evidence": ["live_token_sequence", "candidate_next_phases", "current_live_state"],
                        "human_review_prompt": "Review whether the GRU phase probabilities match the live semantic state.",
                    },
                    "evidence": evidence,
                }
                if self.live_phase_refresh_warning:
                    evidence["live_refresh_warning"] = self.live_phase_refresh_warning
                payload = self.build_evidence_pack(include_neighbors=True)
                payload["live_phase_prediction"] = evidence
                llm_payload = self._llm_json("live_phase_prediction", question, payload)
                if llm_payload:
                    return {
                        "mode": "llm_rag_live_phase_prediction",
                        **llm_payload,
                        "phase_prediction": phase_result,
                        "forecast_model_source": evidence.get("forecast_model_source"),
                        "model_artifact_path": evidence.get("model_artifact_path"),
                        "evidence_pack": payload,
                    }
                return {
                    "mode": "hitl_gru_live_phase_prediction",
                    "answer": f"Predicted next phase from the latest real-time semantic window sequence. {explanation}",
                    "phase_prediction": phase_result,
                    "forecast_model_source": evidence.get("forecast_model_source"),
                    "model_artifact_path": evidence.get("model_artifact_path"),
                    "evidence": payload,
                    "evidence_pack": payload,
                    "human_review_prompt": phase_result["phase_forecast"]["human_review_prompt"],
                }
            except Exception as exc:
                if self.phase_model_mode == "gru":
                    payload = self.build_evidence_pack(include_neighbors=False)
                    payload["phase_forecast_error"] = {
                        "phase_model_mode": self.phase_model_mode,
                        "model_artifact_path": str(self.phase_model_path),
                        "error": str(exc),
                        "evidence_id": "phase_forecast_error",
                    }
                    return {
                        "mode": "phase_forecast_error",
                        "answer": f"Could not run HITL phase GRU: {exc}",
                        "phase_prediction": {"evidence": payload["phase_forecast_error"]},
                        "forecast_model_source": "hitl_phase_gru_error",
                        "model_artifact_path": str(self.phase_model_path),
                        "model_warning": str(exc),
                        "evidence": payload,
                        "evidence_pack": payload,
                        "human_review_prompt": "Check whether the phase GRU artifact exists and live semantic history is compatible.",
                    }
                phase_model_warning = f"Phase GRU unavailable; used transition fallback: {exc}"

        history_length = max(1, int(self.phase_config.history_length))
        latest_history = live_history.tail(history_length)
        live_sequence = [int(token) for token in latest_history["token_id"].tolist()]
        historical_tokens = pd.to_numeric(
            self.context.semantic_states["token_id"],
            errors="coerce",
        ).dropna().astype(int).to_numpy()
        source, counts, support = _historical_transition_counts(
            historical_tokens=historical_tokens,
            sequence=live_sequence,
            horizon_steps=self.phase_config.horizon_steps,
            min_support=self.phase_config.min_support,
        )
        candidates = _rank_phase_counts(
            counts=counts,
            support=support,
            source=source,
            cluster_profiles=self.context.cluster_profiles,
            top_k=self.phase_config.top_k,
        )
        live_state = self._load_live_phase_state()
        current_live_state = live_state.get("live_state", {}) if isinstance(live_state, dict) else {}
        latest_window = live_history.iloc[-1].to_dict()
        support_is_sufficient = bool(support >= max(1, int(self.phase_config.min_support)))
        transition_model_source = "transition_fallback" if phase_model_warning else "transition_count"
        transition_artifact_path = str(self.phase_model_path) if phase_model_warning else None
        evidence = {
            "run_name": self.context.metadata.get("run_name"),
            "metadata_path": str(self.context.metadata_path),
            "live_history_path": str(self.live_phase_history_path),
            "live_state_path": str(self.live_phase_state_path),
            "history_length": int(history_length),
            "horizon_steps": int(self.phase_config.horizon_steps),
            "minimum_support": int(self.phase_config.min_support),
            "live_token_sequence": live_sequence,
            "current_live_state": current_live_state or latest_window,
            "candidate_next_phases": candidates,
            "support": int(support),
            "support_is_sufficient": support_is_sufficient,
            "similar_transition_analogs": [],
            "forecast_model_source": transition_model_source,
            "model_artifact_path": transition_artifact_path,
            "model_warning": phase_model_warning,
            "grounding_rules": [
                "Live phase prediction uses the latest accumulated live semantic-window token sequence.",
                "Candidate phases are estimated from historical NOAA semantic token transitions.",
                "The LLM may only explain the deterministic candidates and must not invent probabilities.",
            ],
        }
        if self.live_phase_refresh_warning:
            evidence["live_refresh_warning"] = self.live_phase_refresh_warning

        current_name = str(current_live_state.get("regime_name") or latest_window.get("regime_name") or "the current live regime")
        current_window_id = str(current_live_state.get("window_id") or latest_window.get("window_id") or "the latest live window")
        if not candidates:
            explanation = (
                "The live semantic history is available, but there is not enough historical transition evidence "
                "to predict a future phase."
            )
            phase_name = "insufficient evidence"
        elif not support_is_sufficient:
            top = candidates[0]
            phase_name = "insufficient evidence"
            explanation = (
                f"The latest live token sequence {live_sequence} points to {top['regime_name']} "
                f"(token {top['token_id']}), but support is only {support}, below the threshold "
                f"{int(self.phase_config.min_support)}. Treat this as weak evidence."
            )
        else:
            top = candidates[0]
            phase_name = top["regime_name"]
            alternatives = candidates[1:3]
            alt_text = ""
            if alternatives:
                alt_text = " Other candidates are " + ", ".join(
                    f"{row['regime_name']} ({row['probability']:.2f})" for row in alternatives
                ) + "."
            explanation = (
                f"The current live window is {current_window_id}, labeled {current_name}. "
                f"The phase forecaster uses the latest live semantic token sequence {live_sequence}, not a raw "
                f"wind-speed forecast. In the historical KBOS semantic database, this sequence maps most often to "
                f"{top['regime_name']} (token {top['token_id']}) as the next phase. The estimated transition "
                f"probability is {top['probability']:.2f}, based on {support} historical transition(s) from "
                f"{source}. This means the system expects the live wind state to most likely continue or move into "
                f"that semantic regime, rather than predicting an exact numeric wind speed.{alt_text} "
                "The probability should be interpreted as transition evidence from similar token histories, not as "
                "a meteorological certainty."
            )

        phase_result = {
            "mode": "deterministic_live_phase_prediction",
            "llm_requested": bool(self.enable_llm),
            "llm_available": self.llm_available,
            "phase_forecast": {
                "phase_forecast": phase_name,
                "explanation": explanation,
                "evidence": ["live_token_sequence", "candidate_next_phases", "current_live_state"],
                "human_review_prompt": (
                    "Review whether the live semantic window sequence and retrieved historical support look credible."
                ),
            },
            "evidence": evidence,
        }

        payload = self.build_evidence_pack(include_neighbors=True)
        payload["live_phase_prediction"] = evidence
        llm_payload = self._llm_json("live_phase_prediction", question, payload) if support_is_sufficient else None
        if llm_payload:
            return {
                "mode": "llm_rag_live_phase_prediction",
                **llm_payload,
                "phase_prediction": phase_result,
                "forecast_model_source": transition_model_source,
                "model_artifact_path": transition_artifact_path,
                "model_warning": phase_model_warning,
                "evidence_pack": payload,
            }
        answer = f"Predicted next phase from the latest real-time semantic window sequence. {explanation}"
        if phase_model_warning:
            answer = f"{answer} Warning: {phase_model_warning}"
        return {
            "mode": "deterministic_live_phase_prediction",
            "answer": answer,
            "phase_prediction": phase_result,
            "forecast_model_source": transition_model_source,
            "model_artifact_path": transition_artifact_path,
            "model_warning": phase_model_warning,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": phase_result["phase_forecast"]["human_review_prompt"],
        }

    def predict_phase(self, question: str) -> dict[str, Any]:
        live_result = self._predict_live_phase(question)
        if live_result:
            return live_result

        forecaster = SemanticPhaseForecaster(
            metadata_path=self.context.metadata_path,
            config=self.phase_config,
        )
        token_sequence = self.phase_tokens

        if token_sequence:
            phase_result = forecaster.forecast_from_tokens(
                token_sequence=token_sequence,
                question=question,
            )
            answer_prefix = (
                f"Predicted next phase distribution from selected token sequence {token_sequence}."
            )
        else:
            phase_result = forecaster.forecast(
                window_id=self.context.resolved_window_id,
                latest=False,
                question=question,
            )
            answer_prefix = "Predicted next phase from the selected semantic window's recent token history."

        payload = self.build_evidence_pack(include_neighbors=True)
        payload["phase_prediction"] = phase_result.get("evidence", {})
        support_is_sufficient = bool(payload["phase_prediction"].get("support_is_sufficient", True))
        llm_payload = self._llm_json("phase_prediction", question, payload) if support_is_sufficient else None
        if llm_payload:
            return {
                "mode": "llm_rag",
                **llm_payload,
                "phase_prediction": phase_result,
                "evidence_pack": payload,
            }

        phase_forecast = phase_result.get("phase_forecast", {})
        return {
            "mode": "deterministic_phase_prediction",
            "answer": f"{answer_prefix} {phase_forecast.get('explanation', '')}",
            "phase_prediction": phase_result,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": phase_forecast.get(
                "human_review_prompt",
                "Review the candidate next phases and retrieved transition analogs.",
            ),
        }

    @staticmethod
    def _infer_feedback_from_question(question: str) -> tuple[str, str, str]:
        text = question.strip()
        lowered = text.lower()

        if any(term in lowered for term in ("accept", "approve", "looks good", "correct")):
            return "accept", "", text
        if any(term in lowered for term in ("flag", "wrong", "incorrect", "suspicious", "reject")):
            return "flag", "", text

        if any(term in lowered for term in ("relabel", "rename", "call it", "label it")):
            return "note", "", text

        return "note", "", text

    def record_feedback(
        self,
        question: str,
        action: str | None = None,
        label: str = "",
        note: str = "",
        reviewer: str = "human",
        filter_decision: FilterDecision | None = None,
    ) -> dict[str, Any]:
        inferred_action, inferred_label, inferred_note = self._infer_feedback_from_question(question)
        action = (action or inferred_action).strip().lower()
        if action == "reject":
            action = "flag"
        elif action == "relabel":
            action = "note"
        if action not in SUPPORTED_ACTIONS:
            action = "note"
        label = label or inferred_label
        note = note or inferred_note

        current_state = self._current_live_compact_state() if self.prefer_live_phase else {}
        if not current_state:
            current_state = self.context.compact_state()
        artifact_id = str(current_state.get("window_id") or self.context.resolved_window_id)

        record = FeedbackRecord(
            artifact_type="semantic_window",
            artifact_id=artifact_id,
            reviewer=reviewer,
            action=action,
            label=label,
            note=note,
            payload={
                "question": question,
                "filter": filter_decision.to_dict() if filter_decision else None,
                "current_semantic_state": current_state,
                "forecast_summary": self.forecast_summary,
            },
        )
        path = self.feedback_store.append(record)
        return {
            "mode": "feedback",
            "answer": f"Recorded human feedback action '{action}'.",
            "feedback_path": str(path),
            "feedback_record": _json_safe(asdict(record)),
        }

    def process(
        self,
        question: str,
        feedback_action: str | None = None,
        feedback_label: str = "",
        feedback_note: str = "",
        reviewer: str = "human",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        if self.enable_agent_graph:
            from app.hitl.controlled_agent_graph import ControlledHITLAgentGraph

            result = ControlledHITLAgentGraph(self).invoke(
                question=question,
                feedback_action=feedback_action,
                feedback_label=feedback_label,
                feedback_note=feedback_note,
                reviewer=reviewer,
                thread_id=thread_id,
            )
            return _json_safe(result)

        decision = self.filter.classify(question, forced_action=feedback_action)
        base = {
            "filter": decision.to_dict(),
            "run_name": self.context.metadata.get("run_name"),
            "window_id": self.context.resolved_window_id,
            "rag_enabled": self.enable_rag,
            "llm_requested": self.enable_llm,
            "llm_available": self.llm_available,
        }

        if not decision.allowed:
            return {
                **base,
                "answer": (
                    "I cannot handle that request in this HITL pipeline. "
                    "Ask about semantic phase predictions, compressed wind states, similar historical regimes, "
                    "or provide human feedback such as accept, flag, or note."
                ),
            }

        if decision.intent == "feedback" or feedback_action:
            result = self.record_feedback(
                question=question,
                action=feedback_action,
                label=feedback_label,
                note=feedback_note,
                reviewer=reviewer,
                filter_decision=decision,
            )
        elif decision.intent == "retrieve_similar":
            result = self.retrieve_similar(question)
        elif decision.intent == "numeric_forecast":
            result = self.predict_phase(question)
        elif decision.intent == "phase_prediction":
            result = self.predict_phase(question)
        elif decision.intent == "explain_forecast":
            result = self.explain_state(question)
        else:
            result = self.explain_state(question)

        final_result = {**base, **result}
        phase_payload = final_result.get("phase_prediction", {})
        phase_evidence = phase_payload.get("evidence", {}) if isinstance(phase_payload, dict) else {}
        evidence_pack = final_result.get("evidence_pack", {})
        current_live_state = phase_evidence.get("current_live_state", {}) if isinstance(phase_evidence, dict) else {}
        if not current_live_state and isinstance(evidence_pack, dict):
            current_live_state = evidence_pack.get("current_state", {})
        live_window_id = current_live_state.get("window_id") if isinstance(current_live_state, dict) else None
        if live_window_id and str(live_window_id).startswith("live_"):
            final_result["window_id"] = str(live_window_id)
        return _json_safe(final_result)
