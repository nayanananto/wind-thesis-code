from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.paths import resolve_metadata_output_paths
from app.semantic.labeling.llm_regime_explainer import LLMRegimeExplainer
from app.semantic.retrieval.similar_regime_search import SimilarRegimeSearcher


def load_semantic_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return resolve_metadata_output_paths(json.load(handle))


def build_semantic_review_packet(
    metadata_path: str | Path,
    window_id: str | None = None,
    latest: bool = False,
    top_k: int = 5,
    enable_llm_explanation: bool = False,
) -> dict[str, Any]:
    metadata = load_semantic_metadata(metadata_path)
    semantic_states = pd.read_csv(metadata["output_paths"]["semantic_states"])
    cluster_profiles = pd.read_csv(metadata["output_paths"]["cluster_profiles"])
    searcher = SimilarRegimeSearcher.load(metadata["output_paths"]["search_index"])

    if semantic_states.empty:
        raise ValueError("Semantic state file is empty.")

    if window_id:
        query_window_id = window_id
    elif latest:
        query_window_id = str(semantic_states.iloc[-1]["window_id"])
    else:
        raise ValueError("Provide a window_id or set latest=True.")

    query_rows = semantic_states[semantic_states["window_id"].astype(str) == str(query_window_id)]
    if query_rows.empty:
        raise ValueError(f"Window id '{query_window_id}' was not found in semantic states.")

    query_row = query_rows.iloc[0].to_dict()
    token_id = int(query_row["token_id"])
    profile_rows = cluster_profiles[cluster_profiles["token_id"].astype(int) == token_id]
    cluster_profile = profile_rows.iloc[0].to_dict() if not profile_rows.empty else {}

    neighbors = searcher.query_by_window_id(query_window_id, top_k=top_k)
    explainer = LLMRegimeExplainer(enable_llm=enable_llm_explanation)
    explanation = explainer.explain_retrieval(
        query_window=query_row,
        neighbor_windows=neighbors.to_dict(orient="records"),
    )

    return {
        "run_name": metadata["run_name"],
        "query_window_id": query_window_id,
        "query_state": query_row,
        "cluster_profile": cluster_profile,
        "similar_windows": neighbors.to_dict(orient="records"),
        "retrieval_explanation": explanation,
        "llm_explanation_enabled": bool(enable_llm_explanation),
    }
