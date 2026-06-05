from pathlib import Path

import pandas as pd

from app.config.paths import DATA_DIR

try:
    from utils import fetch_wind_data, load_scada_data
except ModuleNotFoundError:
    fetch_wind_data = None
    load_scada_data = None


DEFAULT_REMOTE_WIND_URL = (
    "https://raw.githubusercontent.com/nayanananto/"
    "wind-data-pipeline/main/data/wind_data.csv"
)


def _normalize_datetime(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    frame = df.copy()
    if time_col not in frame.columns:
        raise ValueError(f"Expected a '{time_col}' column in the wind dataset.")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return frame


def load_runtime_wind_data(
    remote_url: str = DEFAULT_REMOTE_WIND_URL,
    local_path: str | Path | None = None,
    hist_path: str | Path | None = None,
) -> pd.DataFrame:
    if fetch_wind_data is None:
        raise ModuleNotFoundError(
            "fetch_wind_data is unavailable because utils.py is not included in this clean codebase. "
            "Use load_local_wind_data for local CSV/parquet experiments."
        )
    local_path = Path(local_path or DATA_DIR / "wind_data.csv")
    hist_path = Path(hist_path or DATA_DIR / "historical_wind.csv")
    return fetch_wind_data(
        remote_url,
        local_path=str(local_path),
        hist_path=str(hist_path),
    )


def load_local_wind_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or DATA_DIR / "wind_data.csv")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return _normalize_datetime(frame)


def load_scada_frame(path: str | Path | None = None) -> pd.DataFrame:
    if load_scada_data is None:
        raise ModuleNotFoundError(
            "load_scada_data is unavailable because utils.py is not included in this clean codebase."
        )
    path = Path(path or DATA_DIR / "scada_data.csv")
    return load_scada_data(str(path))
