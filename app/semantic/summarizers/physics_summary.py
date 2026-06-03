import numpy as np
import pandas as pd

from app.semantic.windowing.window_builder import SemanticWindow


OPTIONAL_MEAN_STD_COLUMNS = (
    "u100",
    "v100",
    "wind_gust_10m_ms",
    "temperature_2m_c",
    "relative_humidity_2m",
    "pressure_msl_hpa",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)


def _as_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").dropna()


def _circular_delta_degrees(direction: pd.Series) -> np.ndarray:
    direction = _as_numeric(direction)
    if direction.size < 2:
        return np.array([], dtype=float)
    radians = np.unwrap(np.deg2rad(direction.to_numpy()))
    return np.rad2deg(np.diff(radians))


def summarize_window(window: SemanticWindow) -> dict[str, float | int | str]:
    frame = window.frame
    result: dict[str, float | int | str] = {
        "window_id": window.window_id,
        "window_start": window.start_time.isoformat(),
        "window_end": window.end_time.isoformat(),
        "window_rows": int(len(frame)),
        "time_span_hours": float(
            (window.end_time - window.start_time).total_seconds() / 3600.0
        ),
    }

    wind_speed = _as_numeric(frame["wind_speed"]) if "wind_speed" in frame.columns else pd.Series(dtype=float)
    wind_direction = (
        _as_numeric(frame["wind_direction"])
        if "wind_direction" in frame.columns
        else pd.Series(dtype=float)
    )

    total_cells = max(frame.shape[0] * max(frame.shape[1] - 1, 1), 1)
    missing_cells = int(frame.drop(columns=["datetime"], errors="ignore").isna().sum().sum())
    result["missing_fraction"] = float(missing_cells / total_cells)

    if not wind_speed.empty:
        wind_diff = wind_speed.diff().dropna()
        result.update(
            {
                "wind_speed_mean": float(wind_speed.mean()),
                "wind_speed_std": float(wind_speed.std(ddof=0)),
                "wind_speed_min": float(wind_speed.min()),
                "wind_speed_max": float(wind_speed.max()),
                "wind_speed_p10": float(np.percentile(wind_speed, 10)),
                "wind_speed_p90": float(np.percentile(wind_speed, 90)),
                "wind_speed_range": float(wind_speed.max() - wind_speed.min()),
                "ramp_abs_mean": float(wind_diff.abs().mean()) if not wind_diff.empty else 0.0,
                "ramp_abs_max": float(wind_diff.abs().max()) if not wind_diff.empty else 0.0,
                "calm_fraction": float((wind_speed < 2.0).mean()),
                "strong_fraction": float((wind_speed >= 10.0).mean()),
            }
        )

        gust = None
        for candidate in ("wind_gust_10m_ms", "wind_gust", "gust"):
            if candidate in frame.columns:
                gust = _as_numeric(frame[candidate])
                break

        gust_peak = float(gust.max()) if gust is not None and not gust.empty else float(wind_speed.max())
        result["gust_factor"] = gust_peak / max(float(wind_speed.mean()), 1e-6)

    if not wind_direction.empty:
        direction_delta = _circular_delta_degrees(wind_direction)
        result.update(
            {
                "direction_mean_deg": float(wind_direction.mean()),
                "direction_span_deg": float(wind_direction.max() - wind_direction.min()),
                "direction_abs_change_mean_deg": float(np.mean(np.abs(direction_delta)))
                if direction_delta.size
                else 0.0,
                "direction_abs_change_max_deg": float(np.max(np.abs(direction_delta)))
                if direction_delta.size
                else 0.0,
                "direction_net_turn_deg": float(direction_delta.sum()) if direction_delta.size else 0.0,
            }
        )

    if "u100" in frame.columns and "v100" in frame.columns:
        u = _as_numeric(frame["u100"])
        v = _as_numeric(frame["v100"])
        if not u.empty and not v.empty:
            vector_speed = np.sqrt((u ** 2) + (v ** 2))
            resultant = float(np.sqrt((u.mean() ** 2) + (v.mean() ** 2)))
            result.update(
                {
                    "vector_speed_mean": float(vector_speed.mean()),
                    "vector_speed_std": float(vector_speed.std(ddof=0)),
                    "vector_resultant_strength": resultant / max(float(vector_speed.mean()), 1e-6),
                    "uv_covariance": float(np.cov(u, v, ddof=0)[0, 1]) if len(u) > 1 else 0.0,
                }
            )

    for column in OPTIONAL_MEAN_STD_COLUMNS:
        if column not in frame.columns:
            continue
        values = _as_numeric(frame[column])
        if values.empty:
            continue
        result[f"{column}_mean"] = float(values.mean())
        result[f"{column}_std"] = float(values.std(ddof=0))

    return result


def summarize_windows(windows: list[SemanticWindow]) -> pd.DataFrame:
    rows = [summarize_window(window) for window in windows]
    return pd.DataFrame(rows)
