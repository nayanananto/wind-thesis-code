import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.wind_dataset import load_local_wind_data
from app.orchestration.semantic_workflow import run_semantic_build


def main() -> None:
    parser = argparse.ArgumentParser("Build semantic wind-state embeddings and tokens.")
    parser.add_argument("--data", type=str, default="data/wind_data.csv")
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--enable_llm_labels", action="store_true")
    args = parser.parse_args()

    frame = load_local_wind_data(args.data)
    metadata = run_semantic_build(
        raw_df=frame,
        window_size=args.window_size,
        step_size=args.step_size,
        n_components=args.components,
        n_clusters=args.clusters,
        enable_llm_labels=args.enable_llm_labels,
        run_name=args.run_name,
    )

    print("Semantic build complete.")
    for key, value in metadata["output_paths"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
