from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from app.hitl.semantic_review import load_semantic_metadata


ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_primary_metadata_paths_are_portable_and_resolvable(self) -> None:
        for name in ("kbos_5min_phase_semantic_build.json", "ddc_5min_phase_semantic_build.json"):
            path = ROOT / "data" / "metadata" / name
            raw = json.loads(path.read_text(encoding="utf-8"))
            for value in raw["output_paths"].values():
                self.assertFalse(Path(value).is_absolute(), value)
            resolved = load_semantic_metadata(path)
            for value in resolved["output_paths"].values():
                self.assertTrue(Path(value).exists(), value)

    def test_station_files_have_correct_station_ids(self) -> None:
        expected = {"KBOS": "KBOS_2024_5min.parquet", "DDC": "DDC_2024_5min.parquet"}
        for station, filename in expected.items():
            frame = pd.read_parquet(ROOT / "data" / "noaa_5min" / filename, columns=["station_id"])
            observed = set(frame["station_id"].dropna().astype(str).str.upper().unique())
            self.assertEqual(observed, {station})

    def test_reported_two_seed_outputs_are_complete(self) -> None:
        phase = pd.read_csv(
            ROOT / "results" / "5min_two_seed_experiments" / "gru_phase_5min_two_seed_summary.csv"
        )
        self.assertEqual(phase["horizon"].astype(int).tolist(), [1, 3, 6, 12])
        token_lstm = pd.read_csv(
            ROOT / "results" / "token_state_lstm_experiments" / "token_state_lstm_summary.csv"
        )
        self.assertEqual(set(token_lstm["station"]), {"KBOS", "DDC"})
        self.assertTrue((token_lstm["seeds"] == 2).all())

    def test_ddc_gradient_boosting_uses_kbos_selected_configs(self) -> None:
        columns = ["design", "horizon_hours", "config_id"]
        kbos = pd.read_csv(
            ROOT / "results" / "kbos_gradient_boosting_experiments" / "gb_best_configs.csv",
            usecols=columns,
        ).sort_values(columns[:2]).reset_index(drop=True)
        ddc = pd.read_csv(
            ROOT / "results" / "ddc_gradient_boosting_experiments" / "gb_best_configs.csv",
            usecols=columns,
        ).sort_values(columns[:2]).reset_index(drop=True)
        pd.testing.assert_frame_equal(kbos, ddc)
        self.assertFalse(
            (ROOT / "results" / "ddc_gradient_boosting_experiments" / "gb_tuning_results.csv").exists()
        )

    def test_no_obsolete_second_station_names_remain(self) -> None:
        offenders = []
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".bak":
                continue
            if "kama" in path.name.lower():
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
