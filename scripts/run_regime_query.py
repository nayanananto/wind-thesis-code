import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.semantic.labeling.llm_regime_explainer import LLMRegimeExplainer
from app.semantic.retrieval.similar_regime_search import SimilarRegimeSearcher
from app.hitl.semantic_review import load_semantic_metadata


def main() -> None:
    parser = argparse.ArgumentParser("Query similar historical semantic wind regimes.")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--window_id", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--enable_llm_explanation", action="store_true")
    args = parser.parse_args()

    metadata = load_semantic_metadata(Path(args.metadata))
    search_index_path = metadata["output_paths"]["search_index"]
    state_frame_path = metadata["output_paths"]["semantic_states"]

    searcher = SimilarRegimeSearcher.load(search_index_path)
    state_frame = searcher.state_frame
    if state_frame is None or state_frame.empty:
        raise ValueError("Semantic state frame is empty.")

    if args.window_id:
        window_id = args.window_id
    elif args.latest:
        window_id = str(state_frame.iloc[-1]["window_id"])
    else:
        raise ValueError("Provide --window_id or use --latest.")

    query_row = state_frame[state_frame["window_id"] == window_id]
    if query_row.empty:
        raise ValueError(f"Window id '{window_id}' not found.")

    neighbors = searcher.query_by_window_id(window_id=window_id, top_k=args.top_k)
    print(f"Query window: {window_id}")
    print(neighbors[
        [
            "window_id",
            "retrieval_distance",
            "token_id",
            "regime_name",
            "wind_speed_mean",
            "wind_speed_std",
            "ramp_abs_max",
        ]
    ].to_string(index=False))

    explainer = LLMRegimeExplainer(enable_llm=args.enable_llm_explanation)
    explanation = explainer.explain_retrieval(
        query_window=query_row.iloc[0].to_dict(),
        neighbor_windows=neighbors.to_dict(orient="records"),
    )
    print("\nExplanation:")
    print(explanation["summary"])
    print(explanation["differences"])


if __name__ == "__main__":
    main()
