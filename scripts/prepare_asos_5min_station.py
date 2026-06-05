from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KNOT_TO_MS = 0.514444


def prepare_asos_5min_csv(input_csv: Path, output_parquet: Path) -> None:
    raw = pd.read_csv(input_csv, na_values=["M", ""])
    if "valid" not in raw.columns or "sknt" not in raw.columns:
        raise ValueError("Expected ASOS columns 'valid' and 'sknt'.")

    dt = pd.to_datetime(raw["valid"], utc=True, errors="coerce").dt.tz_convert(None)
    speed_ms = pd.to_numeric(raw["sknt"], errors="coerce") * KNOT_TO_MS
    gust_ms = pd.to_numeric(raw.get("gust"), errors="coerce") * KNOT_TO_MS
    direction = pd.to_numeric(raw.get("drct"), errors="coerce")

    frame = pd.DataFrame(
        {
            "datetime": dt,
            "wind_speed": speed_ms,
            "wind_gust_10m_ms": gust_ms,
            "wind_direction": direction,
        }
    )

    station = "UNKNOWN"
    if "station" in raw.columns and raw["station"].dropna().size:
        station = str(raw["station"].dropna().mode().iloc[0])
    frame["station_id"] = station
    frame["station_call_sign"] = station

    frame = frame.dropna(subset=["datetime", "wind_speed"])
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last")
    frame = frame.set_index("datetime").resample("5min").asfreq()

    for col in ["wind_speed", "wind_gust_10m_ms", "wind_direction"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").interpolate(limit_direction="both")

    frame["wind_gust_10m_ms"] = frame["wind_gust_10m_ms"].fillna(frame["wind_speed"])
    frame["wind_gust_10m_ms"] = frame[["wind_gust_10m_ms", "wind_speed"]].max(axis=1)
    frame["station_id"] = station
    frame["station_call_sign"] = station

    radians = np.deg2rad(frame["wind_direction"].astype(float))
    frame["u100"] = frame["wind_speed"].astype(float) * np.cos(radians)
    frame["v100"] = frame["wind_speed"].astype(float) * np.sin(radians)

    seconds = frame.index.hour * 3600.0 + frame.index.minute * 60.0 + frame.index.second
    frame["hour_sin"] = np.sin(2 * np.pi * seconds / 86400.0)
    frame["hour_cos"] = np.cos(2 * np.pi * seconds / 86400.0)
    dow = frame.index.dayofweek
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    final = frame.reset_index()
    final.to_parquet(output_parquet, index=False)

    print(f"saved: {output_parquet}")
    print(f"station in file: {station}")
    print(f"rows: {len(final)}")
    print(f"range: {final['datetime'].min()} to {final['datetime'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser("Normalize a 5-minute ASOS CSV to the thesis benchmark schema.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_parquet", type=Path, required=True)
    args = parser.parse_args()
    prepare_asos_5min_csv(args.input_csv, args.output_parquet)


if __name__ == "__main__":
    main()
