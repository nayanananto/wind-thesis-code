from pathlib import Path

import pandas as pd

from app.config.paths import DATA_DIR

DEFAULT_WIND_DATA = DATA_DIR / "noaa_5min" / "KBOS_2024_5min.parquet"


def _normalize_datetime(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    frame = df.copy()
    if time_col not in frame.columns:
        raise ValueError(f"Expected a '{time_col}' column in the wind dataset.")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return frame


def load_local_wind_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or DEFAULT_WIND_DATA)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return _normalize_datetime(frame)
