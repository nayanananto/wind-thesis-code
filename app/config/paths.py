from pathlib import Path


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
