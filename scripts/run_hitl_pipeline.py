import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.hitl.standalone_pipeline import StandaloneHITLPipeline


DEFAULT_HOURLY_METADATA = PROJECT_ROOT / "data" / "metadata" / "kbos_hourly_hitl_semantic_build.json"
DEFAULT_5MIN_METADATA = PROJECT_ROOT / "data" / "metadata" / "kbos_5min_phase_semantic_build.json"


def _default_metadata_path() -> Path:
    return DEFAULT_HOURLY_METADATA if DEFAULT_HOURLY_METADATA.exists() else DEFAULT_5MIN_METADATA


def _parse_tokens(value: str | None) -> list[int] | None:
    if not value:
        return None
    tokens = []
    for part in value.replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    return tokens or None


def main() -> None:
    parser = argparse.ArgumentParser("Run the standalone wind HITL review pipeline.")
    parser.add_argument("--metadata", type=str, default=str(_default_metadata_path()))
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--forecast_path", type=str, default=None)
    parser.add_argument("--window_id", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--phase_tokens", type=str, default=None)
    parser.add_argument("--phase_history_length", type=int, default=2)
    parser.add_argument("--phase_horizon_steps", type=int, default=1)
    parser.add_argument("--phase_top_k", type=int, default=3)
    parser.add_argument("--phase_analog_k", type=int, default=5)
    parser.add_argument("--phase_min_support", type=int, default=5)
    parser.add_argument("--live_phase_history_path", type=str, default=None)
    parser.add_argument("--live_phase_state_path", type=str, default=None)
    parser.add_argument("--live_raw_path", type=str, default=None)
    parser.add_argument("--numeric_forecast_steps", type=int, default=6)
    parser.add_argument("--numeric_forecast_mode", type=str, default="auto", choices=["auto", "lstm", "trend"])
    parser.add_argument("--numeric_model_path", type=str, default=None)
    parser.add_argument("--phase_model_mode", type=str, default="auto", choices=["auto", "gru", "transition"])
    parser.add_argument("--phase_model_path", type=str, default=None)
    parser.add_argument("--disable_live_phase", action="store_true")
    parser.add_argument("--disable_rag", action="store_true")
    parser.add_argument("--enable_llm", action="store_true")
    parser.add_argument("--agent_graph", action="store_true")
    parser.add_argument("--thread_id", type=str, default="hitl_cli_default")
    parser.add_argument("--confidence_threshold", type=float, default=0.45)
    parser.add_argument("--feedback_action", type=str, default=None, choices=["accept", "flag", "relabel", "note", "reject"])
    parser.add_argument("--feedback_label", type=str, default="")
    parser.add_argument("--feedback_note", type=str, default="")
    parser.add_argument("--reviewer", type=str, default="human")
    parser.add_argument("--feedback_path", type=str, default=None)
    args = parser.parse_args()

    pipeline = StandaloneHITLPipeline(
        metadata_path=args.metadata,
        window_id=args.window_id,
        latest=args.latest,
        forecast_path=args.forecast_path,
        top_k=args.top_k,
        phase_tokens=_parse_tokens(args.phase_tokens),
        phase_history_length=args.phase_history_length,
        phase_horizon_steps=args.phase_horizon_steps,
        phase_top_k=args.phase_top_k,
        phase_analog_k=args.phase_analog_k,
        phase_min_support=args.phase_min_support,
        live_phase_history_path=args.live_phase_history_path,
        live_phase_state_path=args.live_phase_state_path,
        live_raw_path=args.live_raw_path,
        numeric_forecast_steps=args.numeric_forecast_steps,
        numeric_forecast_mode=args.numeric_forecast_mode,
        numeric_model_path=args.numeric_model_path,
        phase_model_mode=args.phase_model_mode,
        phase_model_path=args.phase_model_path,
        prefer_live_phase=not args.disable_live_phase,
        enable_rag=not args.disable_rag,
        enable_llm=args.enable_llm,
        enable_agent_graph=args.agent_graph,
        confidence_threshold=args.confidence_threshold,
        feedback_path=args.feedback_path,
    )
    result = pipeline.process(
        question=args.question,
        feedback_action=args.feedback_action,
        feedback_label=args.feedback_label,
        feedback_note=args.feedback_note,
        reviewer=args.reviewer,
        thread_id=args.thread_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
