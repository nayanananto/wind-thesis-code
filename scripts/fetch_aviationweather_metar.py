from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://aviationweather.gov/api/data/metar"
KT_TO_MS = 0.514444


def _first(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    text = str(value).strip()
    if text.isdigit():
        timestamp = float(text)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat()
    except ValueError:
        return str(value)


def fetch_metar(station: str, hours: int = 2, timeout: int = 30) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "ids": station.upper(),
            "format": "json",
            "taf": "false",
            "hours": str(hours),
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "wind-forecasting-hitl-live-metar/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, dict):
        # Keep compatibility if the API wraps observations in a top-level key.
        for key in ("data", "metars", "features"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected METAR API response type: {type(payload).__name__}")
    return [row for row in payload if isinstance(row, dict)]


def normalize_metar(row: dict[str, Any], station: str) -> dict[str, Any]:
    station_id = str(_first(row, ["icaoId", "station_id", "station", "id"], station)).upper()
    observation_time = _parse_time(
        _first(row, ["obsTime", "reportTime", "observation_time", "datetime", "receiptTime"])
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    wind_dir = _as_int(_first(row, ["wdir", "windDirDegrees", "wind_direction", "wind_dir_degrees"]))
    wind_speed_kt = _as_float(_first(row, ["wspd", "windSpeedKt", "wind_speed_kt"]))
    wind_gust_kt = _as_float(_first(row, ["wgst", "windGustKt", "wind_gust_kt"]))

    return {
        "fetched_at_utc": fetched_at,
        "observation_time_utc": observation_time,
        "station_id": station_id,
        "wind_direction": wind_dir,
        "wind_speed_kt": wind_speed_kt,
        "wind_speed": round(wind_speed_kt * KT_TO_MS, 6) if wind_speed_kt is not None else None,
        "wind_gust_kt": wind_gust_kt,
        "wind_gust_10m_ms": round(wind_gust_kt * KT_TO_MS, 6) if wind_gust_kt is not None else None,
        "temperature_c": _as_float(_first(row, ["temp", "temp_c", "temperature_c"])),
        "dewpoint_c": _as_float(_first(row, ["dewp", "dewpoint_c"])),
        "altimeter_hpa": _as_float(_first(row, ["altim", "altimeter_hpa"])),
        "raw_metar": _first(row, ["rawOb", "raw_text", "raw_metar", "raw"], ""),
        "source": "aviationweather_metar",
    }


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("station_id") or ""),
        str(row.get("observation_time_utc") or ""),
        str(row.get("raw_metar") or ""),
    )


def append_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    existing_keys: set[tuple[str, str, str]] = set()
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                existing_keys.add(_dedupe_key(row))

    new_rows = [row for row in rows if _dedupe_key(row) not in existing_keys]
    if not new_rows:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def write_latest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live-ish METAR wind observations from AviationWeather.")
    parser.add_argument("--station", default="KBOS", help="ICAO station id, e.g. KBOS")
    parser.add_argument("--hours", type=int, default=12, help="How many recent METAR hours to request")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/live/aviationweather_metar"),
        help="Directory for live METAR CSV/JSON outputs",
    )
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    station = args.station.upper()
    raw_rows = fetch_metar(station=station, hours=args.hours, timeout=args.timeout)
    normalized = [normalize_metar(row, station=station) for row in raw_rows]
    normalized = [
        row
        for row in normalized
        if row.get("observation_time_utc") and row.get("wind_speed") is not None
    ]
    normalized.sort(key=lambda row: str(row.get("observation_time_utc") or ""))

    station_dir = args.output_dir / station
    appended = append_csv(station_dir / "metar_live.csv", normalized)
    if normalized:
        write_latest(station_dir / "latest_metar.json", normalized[-1])

    print(
        json.dumps(
            {
                "station": station,
                "fetched": len(raw_rows),
                "usable_wind_rows": len(normalized),
                "appended": appended,
                "latest_observation_time_utc": normalized[-1]["observation_time_utc"] if normalized else None,
                "output_csv": str(station_dir / "metar_live.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
