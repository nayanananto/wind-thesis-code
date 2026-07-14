# Interpretable Semantic Wind-Regime Forecasting

Reproducibility code and saved outputs for the thesis/paper **Interpretable Semantic Wind-Regime Forecasting with a Controlled Human-in-the-Loop Layer**. The repository contains the complete offline benchmark, semantic artifacts, second-station evaluation, paper figure data, and the live HITL review interface.

## Method at a Glance

1. Convert NOAA/ASOS observations to a consistent five-minute schema.
2. Build overlapping four-hour windows (`48` observations, one-hour stride).
3. Summarize each window with `36` physical/statistical descriptors.
4. Standardize and compress the descriptors to an `8`-dimensional PCA embedding.
5. Discretize embeddings with mini-batch k-means into `8` wind-regime tokens and retain distance to the assigned centroid as an atypicality score.
6. Evaluate continuous forecasting from raw, PCA, autoencoder, and regime-token representations.
7. Evaluate next-regime classification with persistence, a variable-order transition-count model, and a GRU.
8. Deploy the transition-count predictor in a LangGraph-controlled HITL workflow with exact nearest-neighbor analog retrieval, OpenAI-backed grounded explanations/follow-ups, session memory, and accept/flag/note feedback.

The language model is used after clustering to assign concise human-readable regime names from cluster profiles and in the HITL explanation layer. It does not assign token IDs, alter embeddings, train forecasting models, or change quantitative predictions.

## Data and Evaluation

- **Primary station:** KBOS (Boston Logan International Airport).
- **Robustness station:** DDC (Dodge City Regional Airport).
- **Data:** NOAA/ASOS five-minute observations from 2024.
- **Chronological split:** 70% train, 15% validation, 15% test for next-regime models.
- **Forecast horizons:** 1, 3, 6, and 12 one-hour window advances.
- **Neural seeds:** 42 and 123; saved summaries average both seeds.
- **Phase metrics:** top-1 accuracy, top-3 accuracy, and macro-F1.
- **Continuous metrics:** MAE, RMSE, sMAPE, and skill relative to persistence.

The DDC benchmark transfers the KBOS-selected neural and gradient-boosting configurations unchanged under the same horizons and seeds. This avoids second-station retuning and makes DDC a deliberately conservative cross-station evaluation; the saved manifests and configuration tables record the transfer protocol.

## Repository Layout

- `app/` - preprocessing, semantic encoding/tokenization, exact nearest-neighbor retrieval, phase forecasting, LangGraph routing, LLM grounding, memory, and feedback modules.
- `scripts/` - semantic build, live METAR adapter, HITL CLI/UI, plotting, carbon calculation, metadata normalization, and anonymous artifact export.
- `theory_experiments/` - two-seed continuous/phase experiments, GRU and transition studies, gradient boosting, and token-state LSTM evaluation.
- `data/noaa_5min/` - normalized KBOS and DDC five-minute datasets.
- `data/semantic/` - saved window features, embeddings, token assignments, profiles, and representative examples.
- `artifacts/` - fitted PCA encoders, tokenizers, retrieval indexes, and HITL model artifacts.
- `results/` - final result tables and experiment manifests.
- `figures/` - generated paper figures.
- `tests/` - repository-integrity and portability checks.
- `DATA_CARD.md` - station ranges, provenance notes, live-data distinction, and use limitations.

## Environment

The committed environment was tested with Python 3.10.4 on CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the integrity checks before experiments:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app scripts theory_experiments backtest_wind.py
```

## Reproduce the Offline Results

The committed CSV/JSON files already contain the reported outputs. Full neural runs can take substantial CPU time.

Primary KBOS two-seed benchmark and compression metrics:

```powershell
python theory_experiments\run_5min_two_seed_final_experiments.py
python theory_experiments\compute_compression_metrics.py
```

DDC robustness benchmark:

```powershell
python theory_experiments\run_ddc_5min_two_seed_experiments.py
python theory_experiments\compute_ddc_compression_metrics.py
```

Gradient-boosting representation probes:

```powershell
python theory_experiments\run_kbos_gradient_boosting_experiments.py
python theory_experiments\run_ddc_gradient_boosting_experiments.py
```

Regime-token-state LSTM extension used in the representation comparison:

```powershell
python theory_experiments\run_token_state_lstm_experiments.py
```

Rebuild paper artifacts from saved results:

```powershell
python scripts\plot_phase_accuracy.py
python scripts\calculate_carbon_footprint.py
```

## Canonical Result Files

Primary KBOS:

- `results/5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/kbos_gradient_boosting_experiments/gb_summary.csv`

Second-station DDC:

- `results/ddc_5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/ddc_5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/ddc_5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/ddc_5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/ddc_gradient_boosting_experiments/gb_summary.csv`

Cross-station token-state LSTM:

- `results/token_state_lstm_experiments/token_state_lstm_summary.csv`
- `results/token_state_lstm_experiments/token_state_lstm_metadata.json`

Paper-supporting artifacts:

- `figures/acc_horizon.png`
- `figures/acc_horizon.pdf`
- `results/carbon_footprint.csv`

## Run the HITL Interface

For LLM explanations and memory-grounded follow-up, set an API key and model:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:SEMANTIC_LLM_MODEL="gpt-4o-mini"
python scripts\run_hitl_ui.py --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861`. On startup the UI attempts to fetch current KBOS observations from the AviationWeather METAR API and rebuild the live semantic state with the committed KBOS encoder/tokenizer. If the network is unavailable, it uses the committed live snapshot. An alternative remote snapshot can be configured with `WIND_LIVE_DATA_BASE_URL`; no personal repository URL is hard-coded.

The fixed UI configuration uses the transition-count phase predictor, top-3 candidates, five analog windows, LangGraph routing, retrieval/memory grounding, and feedback logging. The continuous numeric forecaster is intentionally disabled in the final HITL workflow.

## Build a New Station

```powershell
python scripts\prepare_asos_5min_station.py `
  --input_csv "path\to\station_5min.csv" `
  --output_parquet "data\noaa_5min\STATION_2024_5min.parquet"

python scripts\run_semantic_build.py `
  --data "data\noaa_5min\STATION_2024_5min.parquet" `
  --window_size 48 `
  --step_size 12 `
  --components 8 `
  --clusters 8 `
  --run_name station_5min_phase `
  --enable_llm_labels
```

## Anonymous Submission Artifact

The public development repository and its Git commit history are **not anonymous**. After committing final changes, create a history-free submission ZIP:

```powershell
python scripts\create_anonymous_artifact.py
```

The script rejects known identity/local-path markers and uses `git archive`, so the ZIP contains neither `.git/` nor commit history. Upload that ZIP or a separately anonymized mirror for double-blind review; do not submit the personal GitHub repository URL.

## Notes

- Retrieval is implemented with scikit-learn's exact cosine `NearestNeighbors` index; FAISS is not required.
- All committed metadata uses project-relative paths and resolves against the current checkout.
- API keys, virtual environments, caches, session memory, and feedback logs are excluded from version control.
- Code is released under the MIT License. NOAA/ASOS and AviationWeather data remain subject to their source terms and provenance.
