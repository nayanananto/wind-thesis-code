from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "theory_experiments" / "compute_compression_metrics.py"
OUT = ROOT / "results" / "kbos_llm_label_5min_two_seed_experiments"


def _load_base_metrics():
    spec = importlib.util.spec_from_file_location("kbos_base_compression_metrics", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base compression script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kbos_base_compression_metrics"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    metrics = _load_base_metrics()
    metrics.OUT = OUT
    metrics.main()

    summary_path = OUT / "compression_metrics_5min_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["rerun_note"] = (
            "Compression metrics for the KBOS rerun after LLM-assisted regime-label refinement. "
            "Compression dimensions are unchanged by label refinement."
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
