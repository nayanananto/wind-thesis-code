# Data Card

## Offline benchmark

The committed benchmark files contain surface-wind observations in a normalized five-minute schema:

| Station | Role | Rows | Time range in file |
|---|---|---:|---|
| KBOS | Primary/coastal station | 92,021 | 2024-01-01 05:00 to 2024-12-23 10:55 |
| DDC | Robustness/inland station | 102,887 | 2024-01-01 05:00 to 2024-12-23 10:50 |

Files:

- `data/noaa_5min/KBOS_2024_5min.parquet`
- `data/noaa_5min/DDC_2024_5min.parquet`

The columns used by the experiments include UTC datetime, wind speed, direction-derived `u/v` components, gust, cyclical calendar features, and station identifiers. `scripts/prepare_asos_5min_station.py` documents the normalization procedure for an additional ASOS-format station CSV.

## Live HITL data

`scripts/fetch_aviationweather_metar.py` obtains recent KBOS observations from the AviationWeather METAR API. METAR reports can arrive at an irregular cadence, so the live adapter uses the latest available rows to construct a review state; it is not presented as another five-minute benchmark sample. The committed files under `data/live/` are a reproducible snapshot and an offline fallback.

## Intended use and limitations

- The data supports research on semantic compression, local regime forecasting, analog retrieval, and controlled review workflows.
- Regime token IDs are learned separately for each station; numeric IDs are not assumed to have the same physical meaning across KBOS and DDC.
- The benchmark is not a certified aviation forecast product and should not be used for operational safety decisions.
- Missingness and preprocessing choices can affect learned regimes. The semantic feature set retains a per-window missing-fraction indicator for inspection.
