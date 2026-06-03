from __future__ import annotations

import json
from typing import Any

from app.llm.semantic_reasoner import SemanticReasoner


REGIME_FIELDS = [
    "wind_speed_mean",
    "wind_speed_std",
    "ramp_abs_max",
    "gust_factor",
    "direction_abs_change_mean_deg",
    "direction_net_turn_deg",
    "calm_fraction",
    "strong_fraction",
    "vector_resultant_strength",
]


def _first_present(profile: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in profile and profile[key] is not None:
            try:
                return float(profile[key])
            except Exception:
                continue
    return default


def heuristic_regime_label(profile: dict[str, Any]) -> dict[str, str]:
    speed_mean = _first_present(profile, ["wind_speed_mean"])
    speed_std = _first_present(profile, ["wind_speed_std"])
    ramp_max = _first_present(profile, ["ramp_abs_max"])
    gust_factor = _first_present(profile, ["gust_factor"], default=1.0)
    direction_change = _first_present(profile, ["direction_abs_change_mean_deg"])
    direction_turn = abs(_first_present(profile, ["direction_net_turn_deg"]))
    calm_fraction = _first_present(profile, ["calm_fraction"])
    strong_fraction = _first_present(profile, ["strong_fraction"])

    if speed_mean < 3.0 or calm_fraction > 0.45:
        intensity = "calm"
    elif speed_mean >= 9.0 or strong_fraction > 0.45:
        intensity = "strong-flow"
    else:
        intensity = "moderate-flow"

    descriptors: list[str] = []
    is_stable = speed_std < 0.8 and ramp_max < 1.2 and direction_change < 10.0
    is_gusty = (
        speed_std >= 1.6
        or ramp_max >= 2.5
        or (gust_factor >= 1.45 and speed_mean >= 3.0)
    )
    is_turning = direction_change >= 20.0 or direction_turn >= 60.0

    if is_stable:
        descriptors.append("stable")
    if is_gusty:
        descriptors.append("gusty")
    if is_turning:
        descriptors.append("turning")

    if not descriptors:
        if speed_std >= 1.0 or ramp_max >= 1.5:
            descriptors.append("variable")
        else:
            descriptors.append("persistent")

    label = f"{intensity} {' '.join(descriptors[:2])} regime"

    if "gusty" in descriptors:
        explanation = "The cluster shows stronger short-term variability, suggesting bursts or uneven flow."
    elif "variable" in descriptors:
        explanation = "The cluster has moderate internal variation without the sharper bursts seen in gustier states."
    elif "turning" in descriptors:
        explanation = "The cluster is characterized by a noticeable directional shift across the window."
    elif "stable" in descriptors:
        explanation = "The cluster remains relatively steady in speed and direction across the window."
    else:
        explanation = "The cluster reflects a recurring wind state without large internal swings."

    if intensity == "calm":
        interpretation = "Likely a weak-flow period with limited production potential and low turbulence."
    elif intensity == "strong-flow":
        interpretation = "Likely a sustained energetic period that may be tied to stronger synoptic forcing."
    else:
        interpretation = "Likely a common operational regime with moderate winds and manageable variability."

    return {
        "regime_name": label,
        "short_explanation": explanation,
        "meteorological_interpretation": interpretation,
        "label_source": "heuristic",
    }


class LLMRegimeExplainer:
    def __init__(self, enable_llm: bool = False):
        self.enable_llm = bool(enable_llm)
        self.reasoner = SemanticReasoner() if enable_llm else None

    def _build_prompt(
        self,
        profile: dict[str, Any],
        representative_examples: list[dict[str, Any]],
    ) -> str:
        compact_profile = {key: profile.get(key) for key in REGIME_FIELDS if key in profile}
        compact_profile["token_id"] = profile.get("token_id")
        compact_profile["count"] = profile.get("count")
        compact_profile["avg_distance"] = profile.get("avg_distance")

        return (
            "You are labeling a learned wind-state cluster from a semantic embedding system.\n"
            "Return only valid JSON with keys: regime_name, short_explanation, meteorological_interpretation.\n"
            "The name must be short and human-readable.\n\n"
            f"Cluster profile:\n{json.dumps(compact_profile, ensure_ascii=False)}\n\n"
            f"Representative windows:\n{json.dumps(representative_examples, ensure_ascii=False)}\n"
        )

    def label_cluster(
        self,
        profile: dict[str, Any],
        representative_examples: list[dict[str, Any]],
    ) -> dict[str, str]:
        heuristic = heuristic_regime_label(profile)
        if not self.enable_llm or not self.reasoner or not self.reasoner.available:
            return heuristic

        payload = self.reasoner.invoke_json(
            self._build_prompt(profile, representative_examples)
        )
        if not payload:
            return heuristic

        name = str(payload.get("regime_name", "")).strip()
        explanation = str(payload.get("short_explanation", "")).strip()
        interpretation = str(payload.get("meteorological_interpretation", "")).strip()

        if not name or not explanation or not interpretation:
            return heuristic

        return {
            "regime_name": name,
            "short_explanation": explanation,
            "meteorological_interpretation": interpretation,
            "label_source": "llm",
        }

    def explain_retrieval(
        self,
        query_window: dict[str, Any],
        neighbor_windows: list[dict[str, Any]],
    ) -> dict[str, str]:
        base = {
            "summary": (
                f"The query state is closest to {len(neighbor_windows)} prior windows with similar "
                "wind-speed level, variability, and directional behavior."
            ),
            "differences": "Use the retrieved rows to compare the regime names, distances, and key summary fields.",
            "source": "heuristic",
        }

        if not self.enable_llm or not self.reasoner or not self.reasoner.available:
            return base

        prompt = (
            "You are explaining a semantic retrieval result for wind-state embeddings.\n"
            "Return only valid JSON with keys: summary, differences.\n\n"
            f"Query window:\n{json.dumps(query_window, ensure_ascii=False)}\n\n"
            f"Retrieved neighbors:\n{json.dumps(neighbor_windows, ensure_ascii=False)}\n"
        )
        payload = self.reasoner.invoke_json(prompt)
        if not payload:
            return base

        summary = str(payload.get("summary", "")).strip()
        differences = str(payload.get("differences", "")).strip()
        if not summary or not differences:
            return base

        return {
            "summary": summary,
            "differences": differences,
            "source": "llm",
        }
