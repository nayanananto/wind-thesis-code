import numpy as np
import pandas as pd


def ensure_datetime_frame(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    frame = df.copy()
    if time_col not in frame.columns:
        raise ValueError(f"Expected '{time_col}' in the input frame.")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return frame


def add_cyclical_time_features(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    frame = ensure_datetime_frame(df, time_col=time_col)
    hour = frame[time_col].dt.hour.to_numpy()
    day_of_week = frame[time_col].dt.dayofweek.to_numpy()

    if "hour_sin" not in frame.columns:
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    if "hour_cos" not in frame.columns:
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if "dow_sin" not in frame.columns:
        frame["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    if "dow_cos" not in frame.columns:
        frame["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    return frame


def add_wind_vector_features(
    df: pd.DataFrame,
    speed_col: str = "wind_speed",
    direction_col: str = "wind_direction",
    u_col: str = "u100",
    v_col: str = "v100",
) -> pd.DataFrame:
    frame = df.copy()

    if speed_col in frame.columns and direction_col in frame.columns:
        if u_col not in frame.columns or v_col not in frame.columns:
            radians = np.deg2rad(pd.to_numeric(frame[direction_col], errors="coerce"))
            speeds = pd.to_numeric(frame[speed_col], errors="coerce")
            frame[u_col] = speeds * np.cos(radians)
            frame[v_col] = speeds * np.sin(radians)

    if speed_col not in frame.columns and u_col in frame.columns and v_col in frame.columns:
        u = pd.to_numeric(frame[u_col], errors="coerce")
        v = pd.to_numeric(frame[v_col], errors="coerce")
        frame[speed_col] = np.sqrt((u ** 2) + (v ** 2))

    if direction_col not in frame.columns and u_col in frame.columns and v_col in frame.columns:
        u = pd.to_numeric(frame[u_col], errors="coerce")
        v = pd.to_numeric(frame[v_col], errors="coerce")
        frame[direction_col] = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0

    return frame


def prepare_semantic_frame(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    frame = add_cyclical_time_features(df, time_col=time_col)
    frame = add_wind_vector_features(frame)

    for column in frame.columns:
        if column == time_col:
            continue
        if frame[column].dtype == object:
            converted = pd.to_numeric(frame[column], errors="ignore")
            frame[column] = converted

    return frame

