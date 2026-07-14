from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SEMANTIC_DATA_DIR = DATA_DIR / "semantic"
METADATA_DIR = DATA_DIR / "metadata"
FORECAST_DATA_DIR = DATA_DIR / "forecasts"
HITL_DATA_DIR = DATA_DIR / "hitl"

SEMANTIC_ENCODER_DIR = ARTIFACTS_DIR / "semantic_encoder"
TOKENIZER_ARTIFACT_DIR = ARTIFACTS_DIR / "tokenizer"
SEMANTIC_INDEX_DIR = ARTIFACTS_DIR / "semantic_index"
BASELINE_ARTIFACT_DIR = ARTIFACTS_DIR / "baseline_lstm"
SCALER_ARTIFACT_DIR = ARTIFACTS_DIR / "scalers"


def project_relative_path(path: str | Path) -> str:
    """Return a portable project-relative path when the file is in this repo."""

    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def resolve_project_path(path: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve relative or stale artifact paths against the current checkout.

    Older experiment metadata stored absolute paths from the machine that created
    it. The basename fallback keeps those artifacts usable after cloning while
    project-relative metadata remains the preferred format.
    """

    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.insert(0, PROJECT_ROOT / raw)

    for candidate in candidates:
        if candidate.exists() or not must_exist:
            return candidate.resolve()

    matches = list(PROJECT_ROOT.rglob(raw.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        suffix_parts = raw.parts[-3:]
        suffix_matches = [
            match for match in matches
            if tuple(match.parts[-len(suffix_parts):]) == tuple(suffix_parts)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0].resolve()

    if must_exist:
        raise FileNotFoundError(f"Could not resolve project artifact path: {path}")
    return (PROJECT_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()


def resolve_metadata_output_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve every metadata output path without modifying the JSON on disk."""

    resolved = dict(metadata)
    outputs = dict(resolved.get("output_paths") or {})
    for key, value in outputs.items():
        if value:
            outputs[key] = str(resolve_project_path(value))
    resolved["output_paths"] = outputs
    return resolved


def ensure_project_dirs() -> dict[str, Path]:
    paths = {
        "data": DATA_DIR,
        "artifacts": ARTIFACTS_DIR,
        "semantic_data": SEMANTIC_DATA_DIR,
        "metadata": METADATA_DIR,
        "forecasts": FORECAST_DATA_DIR,
        "hitl": HITL_DATA_DIR,
        "semantic_encoder": SEMANTIC_ENCODER_DIR,
        "tokenizer": TOKENIZER_ARTIFACT_DIR,
        "semantic_index": SEMANTIC_INDEX_DIR,
        "baseline_lstm": BASELINE_ARTIFACT_DIR,
        "scalers": SCALER_ARTIFACT_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
