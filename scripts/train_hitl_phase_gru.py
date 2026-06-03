from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.hitl.hitl_model_artifacts import train_tuned_phase_gru


def main() -> None:
    parser = argparse.ArgumentParser("Tune and train the hourly HITL semantic phase GRU.")
    parser.add_argument("--metadata", type=str, default="data/metadata/kbos_hourly_hitl_semantic_build.json")
    parser.add_argument("--output_dir", type=str, default="artifacts/hitl_phase_gru/kbos_hourly_v1")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_tuned_phase_gru(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        horizon=args.horizon,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
