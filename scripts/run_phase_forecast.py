import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.phase_forecasting.semantic_phase_forecaster import (  # noqa: E402
    PhaseForecastConfig,
    SemanticPhaseForecaster,
)


def main() -> None:
    parser = argparse.ArgumentParser("Run semantic phase forecasting over tokenized wind states.")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--window_id", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--horizon_steps", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--analog_k", type=int, default=5)
    parser.add_argument("--question", type=str, default="Forecast the next semantic wind phase.")
    parser.add_argument("--enable_llm", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--max_cases", type=int, default=250)
    parser.add_argument("--min_index", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = PhaseForecastConfig(
        history_length=args.history_length,
        horizon_steps=args.horizon_steps,
        top_k=args.top_k,
        analog_k=args.analog_k,
        enable_llm=args.enable_llm,
    )
    forecaster = SemanticPhaseForecaster(
        metadata_path=args.metadata,
        config=config,
    )

    if args.evaluate:
        result = forecaster.evaluate(
            max_cases=args.max_cases,
            min_index=args.min_index,
        )
    else:
        result = forecaster.forecast(
            window_id=args.window_id,
            latest=args.latest,
            question=args.question,
        )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
