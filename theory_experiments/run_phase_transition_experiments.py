from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "wind_compression_experiments"
STATES_PATH = PROJECT_ROOT / "data" / "semantic" / "kbos_5min_phase_semantic_states.csv"
HORIZONS = [1, 3, 6, 12]
HISTORY_LENGTHS = [1, 2, 3, 4, 6]
MIN_SUPPORTS = [1, 3, 5, 10]
TOP_K = 3


def prior_counts(tokens: np.ndarray) -> Counter[int]:
    return Counter(int(token) for token in tokens)


def build_transition_index(
    train_tokens: np.ndarray,
    horizon: int,
    history_length: int,
) -> tuple[dict[tuple[int, ...], Counter[int]], dict[int, Counter[int]], Counter[int]]:
    exact: dict[tuple[int, ...], Counter[int]] = {}
    markov: dict[int, Counter[int]] = {}
    for end_idx in range(history_length - 1, len(train_tokens) - horizon):
        sequence = tuple(int(token) for token in train_tokens[end_idx - history_length + 1 : end_idx + 1])
        target = int(train_tokens[end_idx + horizon])
        exact.setdefault(sequence, Counter())[target] += 1
    for end_idx in range(0, len(train_tokens) - horizon):
        token = int(train_tokens[end_idx])
        target = int(train_tokens[end_idx + horizon])
        markov.setdefault(token, Counter())[target] += 1
    return exact, markov, prior_counts(train_tokens)


def predict_distribution(
    sequence: np.ndarray,
    min_support: int,
    exact_index: dict[tuple[int, ...], Counter[int]],
    markov_index: dict[int, Counter[int]],
    prior: Counter[int],
) -> tuple[list[int], int, str]:
    counts = exact_index.get(tuple(int(token) for token in sequence), Counter())
    support = sum(counts.values())
    source = "exact_sequence"
    if support < min_support:
        counts = markov_index.get(int(sequence[-1]), Counter())
        support = sum(counts.values())
        source = "last_token_markov"
    if support <= 0:
        counts = prior
        support = sum(counts.values())
        source = "prior"
    ranked = [token for token, _ in counts.most_common(TOP_K)]
    return ranked, int(support), source


def evaluate(
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    horizon: int,
    history_length: int,
    min_support: int,
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    topk_hits: list[float] = []
    supports: list[int] = []
    source_counts: Counter[str] = Counter()

    if len(eval_tokens) <= history_length + horizon:
        raise ValueError("Not enough evaluation tokens for horizon/history setting.")

    exact_index, markov_index, prior = build_transition_index(train_tokens, horizon, history_length)

    for end_idx in range(history_length - 1, len(eval_tokens) - horizon):
        sequence = eval_tokens[end_idx - history_length + 1 : end_idx + 1].astype(int)
        actual = int(eval_tokens[end_idx + horizon])
        ranked, support, source = predict_distribution(
            sequence,
            min_support,
            exact_index=exact_index,
            markov_index=markov_index,
            prior=prior,
        )
        pred = int(ranked[0]) if ranked else int(train_tokens[-1])
        y_true.append(actual)
        y_pred.append(pred)
        topk_hits.append(float(actual in ranked[:TOP_K]))
        supports.append(int(support))
        source_counts[source] += 1

    labels = sorted(set(y_true) | set(y_pred))
    return {
        "horizon": int(horizon),
        "history_length": int(history_length),
        "min_support": int(min_support),
        "n_eval": int(len(y_true)),
        "top1_accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "top3_accuracy": float(np.mean(topk_hits)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "avg_support": float(np.mean(supports)),
        "median_support": float(np.median(supports)),
        "exact_sequence_rate": float(source_counts["exact_sequence"] / max(1, len(y_true))),
        "markov_fallback_rate": float(source_counts["last_token_markov"] / max(1, len(y_true))),
        "prior_fallback_rate": float(source_counts["prior"] / max(1, len(y_true))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(STATES_PATH)
    if "window_start" in states.columns:
        states["window_start"] = pd.to_datetime(states["window_start"], errors="coerce")
        states = states.sort_values("window_start").reset_index(drop=True)
    tokens = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int).to_numpy()
    n = len(tokens)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    train_val_tokens = tokens[:val_end]
    test_tokens = tokens[val_end:]

    manifest = {
        "states_path": str(STATES_PATH),
        "n_tokens": int(n),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "test_tokens": int(len(test_tokens)),
        "horizons": HORIZONS,
        "history_lengths": HISTORY_LENGTHS,
        "min_supports": MIN_SUPPORTS,
        "top_k": TOP_K,
    }
    (OUTPUT_DIR / "phase_transition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tuning_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for history_length in HISTORY_LENGTHS:
            for min_support in MIN_SUPPORTS:
                row = evaluate(train_tokens, val_tokens, horizon, history_length, min_support)
                row["split"] = "val"
                tuning_rows.append(row)
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(OUTPUT_DIR / "phase_transition_tuning_results.csv", index=False)

    best = (
        tuning.sort_values(
            ["horizon", "top1_accuracy", "top3_accuracy", "macro_f1", "avg_support"],
            ascending=[True, False, False, False, False],
        )
        .groupby("horizon", as_index=False)
        .first()
    )
    best.to_csv(OUTPUT_DIR / "phase_transition_best_configs.csv", index=False)

    final_rows: list[dict[str, Any]] = []
    for _, row in best.iterrows():
        result = evaluate(
            train_val_tokens,
            test_tokens,
            horizon=int(row["horizon"]),
            history_length=int(row["history_length"]),
            min_support=int(row["min_support"]),
        )
        result["split"] = "test"
        final_rows.append(result)
    final = pd.DataFrame(final_rows)
    final.to_csv(OUTPUT_DIR / "phase_transition_final_results.csv", index=False)
    print("Best phase configs:")
    print(best.to_string(index=False))
    print("\nFinal phase test results:")
    print(final.to_string(index=False))
    print(f"\nSaved phase results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
