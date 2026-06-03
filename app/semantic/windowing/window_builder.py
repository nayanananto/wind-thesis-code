from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 24
    step_size: int = 1
    time_col: str = "datetime"
    columns: tuple[str, ...] | None = None
    min_periods: int | None = None


@dataclass
class SemanticWindow:
    window_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    row_start: int
    row_end: int
    frame: pd.DataFrame


def build_windows(df: pd.DataFrame, config: WindowConfig) -> list[SemanticWindow]:
    if config.window_size <= 0:
        raise ValueError("window_size must be positive.")
    if config.step_size <= 0:
        raise ValueError("step_size must be positive.")

    frame = df.copy()
    frame[config.time_col] = pd.to_datetime(frame[config.time_col], errors="coerce")
    frame = frame.dropna(subset=[config.time_col]).sort_values(config.time_col).reset_index(drop=True)

    if config.columns:
        selected_cols = [config.time_col] + [c for c in config.columns if c in frame.columns]
        frame = frame[selected_cols]

    min_periods = config.min_periods or config.window_size
    windows: list[SemanticWindow] = []

    for end_idx in range(config.window_size, len(frame) + 1, config.step_size):
        start_idx = end_idx - config.window_size
        window_frame = frame.iloc[start_idx:end_idx].copy()
        if len(window_frame) < min_periods:
            continue

        windows.append(
            SemanticWindow(
                window_id=f"window_{start_idx:06d}_{end_idx - 1:06d}",
                start_time=window_frame[config.time_col].iloc[0],
                end_time=window_frame[config.time_col].iloc[-1],
                row_start=start_idx,
                row_end=end_idx - 1,
                frame=window_frame,
            )
        )

    return windows

