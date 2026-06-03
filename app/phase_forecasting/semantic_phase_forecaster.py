from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.hitl.semantic_review import load_semantic_metadata
from app.llm.semantic_reasoner import SemanticReasoner


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


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


@dataclass
class PhaseForecastConfig:
    history_length: int = 4
    horizon_steps: int = 1
    top_k: int = 3
    analog_k: int = 5
    min_support: int = 5
    enable_llm: bool = False


class SemanticPhaseForecaster:
    """Predict ranked next semantic wind regimes from token transitions.

    The predictor is deterministic. The optional LLM only explains the evidence.
    """

    def __init__(self, metadata_path: str | Path, config: PhaseForecastConfig | None = None):
        self.metadata_path = Path(metadata_path)
        self.config = config or PhaseForecastConfig()
        self.metadata = load_semantic_metadata(self.metadata_path)
        self.states = pd.read_csv(self.metadata["output_paths"]["semantic_states"])
        if self.states.empty:
            raise ValueError("Semantic states are empty.")

        if "window_start" in self.states.columns:
            self.states["window_start"] = pd.to_datetime(self.states["window_start"], errors="coerce")
            self.states = self.states.sort_values("window_start").reset_index(drop=True)
        else:
            self.states = self.states.reset_index(drop=True)

        self.states["token_id"] = pd.to_numeric(self.states["token_id"], errors="coerce").astype("Int64")
        self.states = self.states.dropna(subset=["token_id"]).reset_index(drop=True)
        self.tokens = self.states["token_id"].astype(int).to_numpy()
        self.embedding_columns = [c for c in self.states.columns if c.startswith("embedding_")]
        self.cluster_profiles = self._load_cluster_profiles()
        self.centers = self._load_token_centers()
        self.reasoner = SemanticReasoner() if self.config.enable_llm else None

    @property
    def llm_available(self) -> bool:
        return bool(self.reasoner and self.reasoner.available)

    def _load_cluster_profiles(self) -> dict[int, dict[str, Any]]:
        path = self.metadata["output_paths"].get("cluster_profiles")
        if not path:
            return {}
        profiles = pd.read_csv(path)
        if profiles.empty or "token_id" not in profiles.columns:
            return {}
        return {
            int(row["token_id"]): row.to_dict()
            for _, row in profiles.iterrows()
            if pd.notna(row.get("token_id"))
        }

    def _load_token_centers(self) -> np.ndarray | None:
        path = self.metadata["output_paths"].get("tokenizer")
        if not path:
            return None
        try:
            payload = joblib.load(path)
            model = payload.get("model") if isinstance(payload, dict) else getattr(payload, "model", None)
            centers = getattr(model, "cluster_centers_", None)
            if centers is None:
                return None
            return np.asarray(centers, dtype=float)
        except Exception:
            return None

    def _resolve_index(self, window_id: str | None = None, latest: bool = False) -> int:
        if window_id:
            matches = self.states.index[self.states["window_id"].astype(str) == str(window_id)].tolist()
            if not matches:
                raise ValueError(f"Window id '{window_id}' was not found.")
            return int(matches[0])
        if latest:
            return int(len(self.states) - 1)
        raise ValueError("Provide a window_id or set latest=True.")

    def _token_label(self, token_id: int) -> str:
        profile = self.cluster_profiles.get(int(token_id), {})
        return str(profile.get("regime_name") or f"token {token_id}")

    def _compact_state(self, idx: int) -> dict[str, Any]:
        row = self.states.iloc[idx].to_dict()
        keys = [
            "window_id",
            "window_start",
            "window_end",
            "token_id",
            "token_distance",
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
        payload["sequence_index"] = int(idx)
        payload["evidence_id"] = "current_state"
        return payload

    def _recent_sequence(self, idx: int) -> list[int]:
        history = max(1, int(self.config.history_length))
        start = max(0, idx - history + 1)
        return [int(x) for x in self.tokens[start : idx + 1]]

    def _candidate_end_indices(self, query_idx: int) -> list[int]:
        history = max(1, int(self.config.history_length))
        horizon = max(1, int(self.config.horizon_steps))
        last_end = query_idx - horizon
        if last_end < history - 1:
            return []
        return list(range(history - 1, last_end + 1))

    def _transition_counts(self, query_idx: int) -> tuple[str, Counter[int], int]:
        query_seq = self._recent_sequence(query_idx)
        history = len(query_seq)
        horizon = max(1, int(self.config.horizon_steps))
        min_support = max(1, int(self.config.min_support))
        candidates = self._candidate_end_indices(query_idx)

        global_counts: Counter[int] = Counter()
        current_token = query_seq[-1]
        suffix_counts_by_length: list[tuple[int, Counter[int], int]] = []
        markov_counts: Counter[int] = Counter()

        for end_idx in candidates:
            next_idx = end_idx + horizon
            if next_idx >= len(self.tokens):
                continue
            next_token = int(self.tokens[next_idx])
            global_counts[next_token] += 1
            if int(self.tokens[end_idx]) == current_token:
                markov_counts[next_token] += 1

        for length in range(history, 1, -1):
            counts: Counter[int] = Counter()
            query_suffix = query_seq[-length:]
            for end_idx in candidates:
                next_idx = end_idx + horizon
                start_idx = end_idx - length + 1
                if start_idx < 0 or next_idx >= len(self.tokens):
                    continue
                past_seq = [int(x) for x in self.tokens[start_idx : end_idx + 1]]
                if past_seq == query_suffix:
                    counts[int(self.tokens[next_idx])] += 1
            support = sum(counts.values())
            if support:
                suffix_counts_by_length.append((length, counts, support))
            if support >= min_support:
                source = "exact_sequence" if length == history else f"suffix_{length}_tokens"
                return source, counts, support

        markov_support = sum(markov_counts.values())
        if markov_support >= min_support:
            return "last_token_markov", markov_counts, markov_support

        if suffix_counts_by_length:
            length, counts, support = suffix_counts_by_length[0]
            source = "exact_sequence_low_support" if length == history else f"suffix_{length}_tokens_low_support"
            return source, counts, support
        if markov_counts:
            return "last_token_markov_low_support", markov_counts, markov_support
        return "global_prior", global_counts, sum(global_counts.values())

    def _rank_candidates(self, query_idx: int) -> list[dict[str, Any]]:
        source, counts, support = self._transition_counts(query_idx)
        if not counts:
            return []

        total = float(sum(counts.values()))
        rows: list[dict[str, Any]] = []
        actual_idx = query_idx + max(1, int(self.config.horizon_steps))
        actual_token = int(self.tokens[actual_idx]) if actual_idx < len(self.tokens) else None

        for rank, (token_id, count) in enumerate(counts.most_common(max(1, int(self.config.top_k))), start=1):
            token_id = int(token_id)
            probability = float(count / total) if total else 0.0
            rows.append(
                {
                    "rank": rank,
                    "token_id": token_id,
                    "regime_name": self._token_label(token_id),
                    "probability": round(probability, 4),
                    "count": int(count),
                    "support": int(support),
                    "transition_source": source,
                    "centroid_distance_to_actual": self._centroid_distance(token_id, actual_token)
                    if actual_token is not None
                    else None,
                }
            )
        return rows

    def _transition_counts_for_tokens(self, token_sequence: list[int]) -> tuple[str, Counter[int], int]:
        sequence = [int(token) for token in token_sequence]
        if not sequence:
            return "selected_token_sequence", Counter(), 0

        horizon = max(1, int(self.config.horizon_steps))
        history = len(sequence)
        counts: Counter[int] = Counter()
        last_end = len(self.tokens) - horizon - 1

        for end_idx in range(history - 1, last_end + 1):
            start_idx = end_idx - history + 1
            past_seq = [int(x) for x in self.tokens[start_idx : end_idx + 1]]
            if past_seq == sequence:
                counts[int(self.tokens[end_idx + horizon])] += 1

        return "selected_token_sequence", counts, sum(counts.values())

    def _rank_counts(
        self,
        counts: Counter[int],
        source: str,
        support: int,
        actual_token: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not counts:
            return [], []

        total = float(sum(counts.values()))
        all_token_ids = sorted({int(token) for token in self.tokens})
        distribution: list[dict[str, Any]] = []
        for token_id in all_token_ids:
            count = int(counts.get(token_id, 0))
            distribution.append(
                {
                    "token_id": token_id,
                    "regime_name": self._token_label(token_id),
                    "probability": round(float(count / total), 4) if total else 0.0,
                    "count": count,
                    "support": int(support),
                    "transition_source": source,
                }
            )

        ranked = sorted(
            distribution,
            key=lambda row: (row["probability"], row["count"]),
            reverse=True,
        )
        candidates: list[dict[str, Any]] = []
        positive_ranked = [row for row in ranked if int(row["count"]) > 0]
        for rank, row in enumerate(positive_ranked[: max(1, int(self.config.top_k))], start=1):
            candidate = dict(row)
            candidate["rank"] = rank
            candidate["centroid_distance_to_actual"] = self._centroid_distance(
                int(candidate["token_id"]),
                actual_token,
            ) if actual_token is not None else None
            candidates.append(candidate)
        return candidates, ranked

    def _retrieve_analogs_for_tokens(self, token_sequence: list[int]) -> list[dict[str, Any]]:
        sequence = [int(token) for token in token_sequence]
        if not sequence:
            return []

        horizon = max(1, int(self.config.horizon_steps))
        history = len(sequence)
        rows: list[dict[str, Any]] = []
        last_end = len(self.tokens) - horizon - 1

        for end_idx in range(history - 1, last_end + 1):
            start_idx = end_idx - history + 1
            past_seq = [int(x) for x in self.tokens[start_idx : end_idx + 1]]
            if past_seq != sequence:
                continue

            next_idx = end_idx + horizon
            next_token = int(self.tokens[next_idx])
            rows.append(
                {
                    "evidence_id": f"pattern_match_{len(rows) + 1}",
                    "source_window_id": str(self.states.iloc[end_idx].get("window_id")),
                    "future_window_id": str(self.states.iloc[next_idx].get("window_id")),
                    "source_index": int(end_idx),
                    "future_index": int(next_idx),
                    "history_tokens": past_seq,
                    "next_token_id": next_token,
                    "next_regime_name": self._token_label(next_token),
                    "source_window_start": str(self.states.iloc[end_idx].get("window_start", "")),
                    "future_window_start": str(self.states.iloc[next_idx].get("window_start", "")),
                }
            )

        return rows[: max(1, int(self.config.analog_k))]

    def _centroid_distance(self, pred_token: int, actual_token: int | None) -> float | None:
        if actual_token is None or self.centers is None:
            return None
        if pred_token >= len(self.centers) or actual_token >= len(self.centers):
            return None
        distance = np.linalg.norm(self.centers[int(pred_token)] - self.centers[int(actual_token)])
        return round(float(distance), 4)

    def _embedding_distance(self, query_idx: int, past_idx: int) -> float:
        if not self.embedding_columns:
            return 0.0
        q = self.states.iloc[query_idx][self.embedding_columns].to_numpy(dtype=float)
        p = self.states.iloc[past_idx][self.embedding_columns].to_numpy(dtype=float)
        return float(np.linalg.norm(q - p))

    def _retrieve_analogs(self, query_idx: int) -> list[dict[str, Any]]:
        query_seq = self._recent_sequence(query_idx)
        history = len(query_seq)
        horizon = max(1, int(self.config.horizon_steps))
        rows: list[dict[str, Any]] = []

        for end_idx in self._candidate_end_indices(query_idx):
            next_idx = end_idx + horizon
            if next_idx >= len(self.tokens) or end_idx - history + 1 < 0:
                continue
            past_seq = [int(x) for x in self.tokens[end_idx - history + 1 : end_idx + 1]]
            token_mismatch = float(np.mean(np.asarray(query_seq) != np.asarray(past_seq)))
            embedding_distance = self._embedding_distance(query_idx, end_idx)
            score = token_mismatch + (0.02 * embedding_distance)
            next_token = int(self.tokens[next_idx])
            rows.append(
                {
                    "evidence_id": f"analog_{len(rows) + 1}",
                    "source_window_id": str(self.states.iloc[end_idx].get("window_id")),
                    "future_window_id": str(self.states.iloc[next_idx].get("window_id")),
                    "source_index": int(end_idx),
                    "future_index": int(next_idx),
                    "history_tokens": past_seq,
                    "next_token_id": next_token,
                    "next_regime_name": self._token_label(next_token),
                    "token_mismatch_rate": round(token_mismatch, 4),
                    "embedding_distance": round(float(embedding_distance), 4),
                    "similarity_score": round(float(score), 4),
                    "source_window_start": str(self.states.iloc[end_idx].get("window_start", "")),
                    "future_window_start": str(self.states.iloc[next_idx].get("window_start", "")),
                }
            )

        return sorted(rows, key=lambda row: row["similarity_score"])[: max(1, int(self.config.analog_k))]

    def _heuristic_explanation(self, evidence: dict[str, Any]) -> dict[str, Any]:
        candidates = evidence.get("candidate_next_phases", [])
        support_is_sufficient = bool(evidence.get("support_is_sufficient", True))
        if not candidates:
            return {
                "phase_forecast": "insufficient evidence",
                "explanation": "There were not enough matching semantic transitions to forecast the next phase.",
                "evidence": ["selected_token_sequence", "candidate_next_phases", "similar_transition_analogs"],
                "human_review_prompt": "Try a shorter token sequence or a different horizon to get more historical support.",
            }

        top = candidates[0]
        if not support_is_sufficient:
            return {
                "phase_forecast": "insufficient evidence",
                "explanation": (
                    f"The selected window's recent token history matched only {top['support']} "
                    f"historical transition(s), below the minimum support threshold of "
                    f"{int(self.config.min_support)}. A reliable next-phase probability is not reported."
                ),
                "evidence": ["current_state", "candidate_next_phases", "similar_transition_analogs"],
                "human_review_prompt": "Review another selected window or lower the horizon if you need stronger transition support.",
            }

        alternatives = candidates[1:3]
        alt_text = ""
        if alternatives:
            alt_text = " Other plausible candidates are " + ", ".join(
                f"{row['regime_name']} ({row['probability']:.2f})" for row in alternatives
            ) + "."

        return {
            "phase_forecast": top["regime_name"],
            "explanation": (
                f"The most likely next semantic phase is {top['regime_name']} "
                f"(token {top['token_id']}, probability {top['probability']:.2f}). "
                f"This is based on {top['transition_source']} with {top['support']} historical transitions."
                f"{alt_text}"
            ),
            "evidence": ["candidate_next_phases", "similar_transition_analogs"],
            "human_review_prompt": (
                "Review whether the retrieved analog transitions are physically similar before trusting the phase label."
            ),
        }

    def _llm_explanation(self, question: str, evidence: dict[str, Any]) -> dict[str, Any] | None:
        if not self.llm_available:
            return None
        prompt = (
            "You are an LLM-assisted semantic phase forecaster for wind regimes.\n"
            "The deterministic phase model has already produced candidate next tokens.\n"
            "Your job is only to explain the ranked candidates from the evidence.\n"
            "Do not invent wind speeds, probabilities, or meteorological causes not present in the evidence.\n"
            "If evidence is weak, say it is weak.\n"
            "Return only valid JSON with keys: phase_forecast, explanation, evidence, human_review_prompt.\n\n"
            f"User question: {question}\n"
            f"Evidence:\n{json.dumps(_json_safe(evidence), ensure_ascii=False)}\n"
        )
        payload = self.reasoner.invoke_json(prompt) if self.reasoner else None
        if not payload:
            return None
        required = {"phase_forecast", "explanation", "evidence", "human_review_prompt"}
        if not required.issubset(payload.keys()):
            return None
        return payload

    def forecast(
        self,
        window_id: str | None = None,
        latest: bool = False,
        question: str = "Forecast the next semantic wind phase.",
    ) -> dict[str, Any]:
        query_idx = self._resolve_index(window_id=window_id, latest=latest)
        recent_tokens = self._recent_sequence(query_idx)
        candidates = self._rank_candidates(query_idx)
        analogs = self._retrieve_analogs(query_idx)

        evidence = {
            "run_name": self.metadata.get("run_name"),
            "metadata_path": str(self.metadata_path),
            "query_index": int(query_idx),
            "history_length": int(self.config.history_length),
            "horizon_steps": int(self.config.horizon_steps),
            "minimum_support": int(self.config.min_support),
            "recent_tokens": recent_tokens,
            "current_state": self._compact_state(query_idx),
            "candidate_next_phases": candidates,
            "support_is_sufficient": bool(candidates and candidates[0].get("support", 0) >= int(self.config.min_support)),
            "similar_transition_analogs": analogs,
            "grounding_rules": [
                "Candidate next phases come from token transition counts, not LLM guessing.",
                "Do not report a confident probability when support_is_sufficient is false.",
                "Similar transition analogs are retrieved from prior semantic token histories.",
                "The LLM may only explain these candidates and should not invent numeric forecasts.",
            ],
        }
        explanation = self._llm_explanation(question, evidence) or self._heuristic_explanation(evidence)
        mode = "llm_grounded" if self.llm_available else "deterministic"
        return _json_safe(
            {
                "mode": mode,
                "llm_requested": bool(self.config.enable_llm),
                "llm_available": self.llm_available,
                "phase_forecast": explanation,
                "evidence": evidence,
            }
        )

    def forecast_from_tokens(
        self,
        token_sequence: list[int],
        question: str = "Forecast the next semantic wind phase from this token sequence.",
    ) -> dict[str, Any]:
        sequence = [int(token) for token in token_sequence]
        if not sequence:
            raise ValueError("Provide at least one token for selected-token phase forecasting.")

        source, counts, support = self._transition_counts_for_tokens(sequence)
        candidates, distribution = self._rank_counts(
            counts=counts,
            source=source,
            support=support,
        )
        analogs = self._retrieve_analogs_for_tokens(sequence)

        evidence = {
            "run_name": self.metadata.get("run_name"),
            "metadata_path": str(self.metadata_path),
            "history_length": len(sequence),
            "horizon_steps": int(self.config.horizon_steps),
            "minimum_support": int(self.config.min_support),
            "selected_token_sequence": sequence,
            "candidate_next_phases": candidates,
            "support_is_sufficient": bool(candidates and candidates[0].get("support", 0) >= int(self.config.min_support)),
            "next_phase_distribution": distribution,
            "similar_transition_analogs": analogs,
            "grounding_rules": [
                "Candidate next phases come from historical occurrences of the selected token sequence.",
                "Do not report a confident probability when support_is_sufficient is false.",
                "Probabilities are estimated over existing semantic tokens only.",
                "The LLM may only explain these candidates and should not invent numeric forecasts.",
            ],
        }
        explanation = self._llm_explanation(question, evidence) or self._heuristic_explanation(evidence)
        mode = "llm_grounded" if self.llm_available else "deterministic"
        return _json_safe(
            {
                "mode": mode,
                "llm_requested": bool(self.config.enable_llm),
                "llm_available": self.llm_available,
                "phase_forecast": explanation,
                "evidence": evidence,
            }
        )

    def evaluate(self, max_cases: int | None = None, min_index: int | None = None) -> dict[str, Any]:
        history = max(1, int(self.config.history_length))
        horizon = max(1, int(self.config.horizon_steps))
        start = max(history, int(min_index or history))
        stop = len(self.states) - horizon
        if stop <= start:
            raise ValueError("Not enough semantic states for phase-forecast evaluation.")

        indices = list(range(start, stop))
        if max_cases and len(indices) > max_cases:
            selected = np.linspace(0, len(indices) - 1, int(max_cases), dtype=int)
            indices = [indices[i] for i in selected]

        rows: list[dict[str, Any]] = []
        for idx in indices:
            candidates = self._rank_candidates(idx)
            if not candidates:
                continue
            actual_token = int(self.tokens[idx + horizon])
            ranked_tokens = [int(row["token_id"]) for row in candidates]
            top1 = ranked_tokens[0] if ranked_tokens else None
            try:
                rank = ranked_tokens.index(actual_token) + 1
                reciprocal_rank = 1.0 / rank
            except ValueError:
                rank = None
                reciprocal_rank = 0.0
            rows.append(
                {
                    "window_id": str(self.states.iloc[idx].get("window_id")),
                    "actual_token": actual_token,
                    "actual_regime_name": self._token_label(actual_token),
                    "top1_token": top1,
                    "top1_regime_name": self._token_label(top1) if top1 is not None else "",
                    "top1_correct": bool(top1 == actual_token),
                    "topk_correct": bool(actual_token in ranked_tokens),
                    "actual_rank": rank,
                    "reciprocal_rank": reciprocal_rank,
                    "top1_centroid_distance": self._centroid_distance(top1, actual_token)
                    if top1 is not None
                    else None,
                    "candidate_tokens": ranked_tokens,
                }
            )

        if not rows:
            raise ValueError("No phase evaluation rows could be scored.")

        frame = pd.DataFrame(rows)
        distances = pd.to_numeric(frame["top1_centroid_distance"], errors="coerce")
        summary = {
            "cases": int(len(frame)),
            "history_length": int(history),
            "horizon_steps": int(horizon),
            "top1_accuracy": round(float(frame["top1_correct"].mean()), 4),
            "topk_accuracy": round(float(frame["topk_correct"].mean()), 4),
            "mean_reciprocal_rank": round(float(frame["reciprocal_rank"].mean()), 4),
            "mean_top1_centroid_distance": round(float(distances.mean()), 4)
            if distances.notna().any()
            else None,
        }
        return _json_safe(
            {
                "summary": summary,
                "rows": frame.to_dict(orient="records"),
            }
        )
