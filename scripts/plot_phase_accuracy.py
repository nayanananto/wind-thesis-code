"""Generate the paper's phase-accuracy-by-horizon figure from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSITION = ROOT / "results" / "5min_two_seed_experiments" / "phase_transition_5min_two_seed_summary.csv"
DEFAULT_GRU = ROOT / "results" / "5min_two_seed_experiments" / "gru_phase_5min_two_seed_summary.csv"
DEFAULT_OUTPUT = ROOT / "figures" / "acc_horizon.png"


def _series(path: Path, metric: str) -> tuple[list[int], list[float]]:
    frame = pd.read_csv(path).sort_values("horizon")
    return frame["horizon"].astype(int).tolist(), frame[metric].astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--gru", type=Path, default=DEFAULT_GRU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    panels = [("top1_accuracy_mean", "Top-1 accuracy"), ("top3_accuracy_mean", "Top-3 accuracy")]
    for axis, (metric, title) in zip(axes, panels):
        horizons, transition = _series(args.transition, metric)
        gru_horizons, gru = _series(args.gru, metric)
        if horizons != gru_horizons:
            raise ValueError("Transition and GRU result files use different horizons.")

        axis.plot(horizons, transition, color="#1a6fa5", marker="o", linewidth=2, label="Transition-count (deployed)")
        axis.plot(horizons, gru, color="#c0392b", marker="s", linestyle="--", linewidth=2, label="GRU (neural baseline)")
        axis.set_title(title, pad=8)
        axis.set_xlabel("Forecast horizon (steps ahead)")
        axis.set_ylabel("Accuracy")
        axis.set_xticks(horizons)
        axis.set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)
        axis.set_ylim(0, 1.05)
        axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        axis.legend(framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout(pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
