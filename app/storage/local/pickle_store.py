import pickle
from pathlib import Path
from typing import Any


def save_pickle(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)

