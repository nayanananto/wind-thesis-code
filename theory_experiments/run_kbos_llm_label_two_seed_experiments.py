from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "theory_experiments" / "run_5min_two_seed_final_experiments.py"
OUT = ROOT / "results" / "kbos_llm_label_5min_two_seed_experiments"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location("kbos_base_two_seed_runner", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base runner: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kbos_base_two_seed_runner"] = module
    spec.loader.exec_module(module)
    return module


def _label_sources() -> list[str]:
    states_path = ROOT / "data" / "semantic" / "kbos_5min_phase_semantic_states.csv"
    states = pd.read_csv(states_path, usecols=["label_source"])
    return sorted(str(x) for x in states["label_source"].dropna().unique())


def main() -> None:
    runner = _load_base_runner()
    runner.OUT = OUT
    runner.main()

    manifest_path = OUT / "5min_two_seed_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["label_sources"] = _label_sources()
        manifest["rerun_note"] = (
            "KBOS rerun after LLM-assisted regime-label refinement. "
            "Regime names/explanations changed; token IDs, embeddings, and quantitative setup are unchanged."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
