from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


KNOT_TO_MS = 0.514444


@dataclass(frozen=True)
class ParsedAsosRecord:
    datetime: pd.Timestamp
    datetime_local: pd.Timestamp
    station_id: str
    station_call_sign: str
    wban: str
    wind_direction: float | None
    wind_speed: float | None
    wind_speed_2min_kt: float | None
    peak_wind_direction_5s: float | None
    peak_wind_speed_5s_ms: float | None
    peak_wind_speed_5s_kt: float | None


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() == "M":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _derive_utc_datetime(
    local_date: pd.Timestamp,
    local_hour: int,
    local_minute: int,
    utc_hour: int,
    utc_minute: int,
) -> pd.Timestamp:
    local_dt = local_date + pd.Timedelta(hours=local_hour, minutes=local_minute)
    utc_dt = local_date + pd.Timedelta(hours=utc_hour, minutes=utc_minute)
    delta_minutes = (utc_hour * 60 + utc_minute) - (local_hour * 60 + local_minute)
    if delta_minutes <= -720:
        utc_dt += pd.Timedelta(days=1)
    elif delta_minutes >= 720:
        utc_dt -= pd.Timedelta(days=1)
    return utc_dt


def parse_noaa_asos_1min_line(line: str) -> ParsedAsosRecord | None:
    if not line:
        return None
    line = line.rstrip("\n")
    if len(line) < 89:
        return None

    wban = line[0:5].strip()
    station_id = line[5:9].strip()
    station_call_sign = line[9:13].strip()

    try:
        year = int(line[13:17])
        month = int(line[17:19])
        day = int(line[19:21])
        local_hour = int(line[21:23])
        local_minute = int(line[23:25])
        utc_hour = int(line[25:27])
        utc_minute = int(line[27:29])
    except ValueError:
        return None

    local_date = pd.Timestamp(year=year, month=month, day=day)
    datetime_local = local_date + pd.Timedelta(hours=local_hour, minutes=local_minute)
    datetime_utc = _derive_utc_datetime(
        local_date=local_date,
        local_hour=local_hour,
        local_minute=local_minute,
        utc_hour=utc_hour,
        utc_minute=utc_minute,
    )

    avg_dir = _parse_optional_int(line[71:74])
    avg_speed_kt = _parse_optional_int(line[74:79])
    peak_dir = _parse_optional_int(line[79:84])
    peak_speed_kt = _parse_optional_int(line[84:89])

    return ParsedAsosRecord(
        datetime=datetime_utc,
        datetime_local=datetime_local,
        station_id=station_id,
        station_call_sign=station_call_sign,
        wban=wban,
        wind_direction=float(avg_dir) if avg_dir is not None else None,
        wind_speed=float(avg_speed_kt) * KNOT_TO_MS if avg_speed_kt is not None else None,
        wind_speed_2min_kt=float(avg_speed_kt) if avg_speed_kt is not None else None,
        peak_wind_direction_5s=float(peak_dir) if peak_dir is not None else None,
        peak_wind_speed_5s_ms=float(peak_speed_kt) * KNOT_TO_MS if peak_speed_kt is not None else None,
        peak_wind_speed_5s_kt=float(peak_speed_kt) if peak_speed_kt is not None else None,
    )


def parse_noaa_asos_1min_file(path: str | Path, drop_all_missing_wind: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            record = parse_noaa_asos_1min_line(raw_line)
            if record is None:
                continue
            if drop_all_missing_wind and record.wind_speed is None and record.peak_wind_speed_5s_ms is None:
                continue
            rows.append(
                {
                    "datetime": record.datetime,
                    "datetime_local": record.datetime_local,
                    "station_id": record.station_id,
                    "station_call_sign": record.station_call_sign,
                    "wban": record.wban,
                    "wind_speed": record.wind_speed,
                    "wind_speed_2min_kt": record.wind_speed_2min_kt,
                    "wind_direction": record.wind_direction,
                    "wind_gust_10m_ms": record.peak_wind_speed_5s_ms,
                    "peak_wind_speed_5s_kt": record.peak_wind_speed_5s_kt,
                    "peak_wind_direction_5s": record.peak_wind_direction_5s,
                    "source_file": path.name,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "datetime_local",
                "station_id",
                "station_call_sign",
                "wban",
                "wind_speed",
                "wind_speed_2min_kt",
                "wind_direction",
                "wind_gust_10m_ms",
                "peak_wind_speed_5s_kt",
                "peak_wind_direction_5s",
                "source_file",
            ]
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("datetime").reset_index(drop=True)
    return frame
