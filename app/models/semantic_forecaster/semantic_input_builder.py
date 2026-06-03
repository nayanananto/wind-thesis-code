from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.preprocessing.feature_engineering import prepare_semantic_frame
from app.semantic.summarizers.physics_summary import summarize_window
from app.semantic.windowing.window_builder import SemanticWindow, WindowConfig, build_windows


SEMANTIC_META_COLUMNS = [
    "window_id",
    "window_start",
    "window_end",
    "window_rows",
    "time_span_hours",
]


@dataclass
class SemanticSampleBundle:
    feature_frame: pd.DataFrame
    targets: np.ndarray
    latest_feature_row: pd.DataFrame
    all_feature_frame: pd.DataFrame
    training_window_ids: list[str]
    training_windows: list[SemanticWindow]
    all_windows: list[SemanticWindow]
    latest_window: SemanticWindow
    future_index: pd.DatetimeIndex


def _window_to_summary_row(window: SemanticWindow) -> pd.DataFrame:
    return pd.DataFrame([summarize_window(window)])


def _infer_time_step(prepared: pd.DataFrame) -> pd.Timedelta:
    series = pd.to_datetime(prepared["datetime"], errors="coerce").dropna().sort_values()
    if len(series) < 2:
        return pd.Timedelta(hours=1)
    diffs = series.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    mode = diffs.mode()
    if not mode.empty:
        return pd.to_timedelta(mode.iloc[0])
    return pd.to_timedelta(diffs.median())


def build_semantic_training_samples(
    train_df: pd.DataFrame,
    horizon: int,
    window_size: int = 48,
    step_size: int = 1,
) -> SemanticSampleBundle:
    prepared = prepare_semantic_frame(train_df)
    time_step = _infer_time_step(prepared)
    if "wind_speed" not in prepared.columns:
        raise ValueError("Semantic forecasting requires a 'wind_speed' column.")
    if len(prepared) < window_size:
        raise ValueError("Not enough history to build the requested semantic window.")

    windows = build_windows(
        prepared,
        WindowConfig(window_size=window_size, step_size=step_size),
    )

    rows: list[dict] = []
    targets: list[np.ndarray] = []
    training_window_ids: list[str] = []
    training_windows: list[SemanticWindow] = []

    for window in windows:
        start_idx = window.row_end + 1
        end_idx = start_idx + int(horizon)
        if end_idx > len(prepared):
            continue

        target_slice = pd.to_numeric(
            prepared.iloc[start_idx:end_idx]["wind_speed"],
            errors="coerce",
        )
        if target_slice.isna().any():
            continue

        rows.append(summarize_window(window))
        targets.append(target_slice.to_numpy(dtype=float))
        training_window_ids.append(window.window_id)
        training_windows.append(window)

    if not rows:
        raise ValueError("Not enough data to build semantic forecasting samples.")

    feature_frame = pd.DataFrame(rows).reset_index(drop=True)
    all_feature_frame = pd.DataFrame(
        [summarize_window(window) for window in windows]
    ).reset_index(drop=True)

    latest_window = SemanticWindow(
        window_id=f"window_{len(prepared) - window_size:06d}_{len(prepared) - 1:06d}",
        start_time=prepared["datetime"].iloc[-window_size],
        end_time=prepared["datetime"].iloc[-1],
        row_start=len(prepared) - window_size,
        row_end=len(prepared) - 1,
        frame=prepared.iloc[-window_size:].copy(),
    )
    latest_feature_row = _window_to_summary_row(latest_window)
    future_index = pd.date_range(
        start=prepared["datetime"].iloc[-1] + time_step,
        periods=int(horizon),
        freq=time_step,
    )

    return SemanticSampleBundle(
        feature_frame=feature_frame,
        targets=np.asarray(targets, dtype=float),
        latest_feature_row=latest_feature_row,
        all_feature_frame=all_feature_frame,
        training_window_ids=training_window_ids,
        training_windows=training_windows,
        all_windows=windows,
        latest_window=latest_window,
        future_index=future_index,
    )
