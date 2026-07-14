"""Reproduce the paper's transparent CPU energy/CO2 estimate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "metadata" / "carbon_workloads.json"
DEFAULT_OUTPUT = ROOT / "results" / "carbon_footprint.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.input.read_text(encoding="utf-8"))
    watts = float(config["cpu_power_watts"])
    intensity = float(config["grid_intensity_kg_co2_per_kwh"])
    rows = []
    for workload in config["workloads"]:
        seconds = float(workload["seconds"])
        energy_kwh = (watts * seconds) / (1000.0 * 3600.0)
        rows.append(
            {
                "workload": workload["name"],
                "seconds": int(seconds),
                "measured": bool(workload["measured"]),
                "energy_kwh": energy_kwh,
                "co2_g": round(energy_kwh * intensity * 1000.0, 1),
            }
        )

    frame = pd.DataFrame(rows)
    total = {
        "workload": "Total",
        "seconds": int(frame["seconds"].sum()),
        "measured": False,
        "energy_kwh": float(frame["energy_kwh"].sum()),
        "co2_g": round(float(frame["co2_g"].sum()), 1),
    }
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    print(frame.to_string(index=False, formatters={"co2_g": "{:.1f}".format}))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
