# Wind Semantic Forecasting Thesis Codebase

Clean, independent codebase for the wind semantic forecasting thesis. It contains the final 5-minute NOAA benchmark, semantic compression artifacts, phase-prediction experiments, second-station robustness outputs, and the controlled HITL review interface.

## Repository Contents

- `app/` - thesis-relevant Python modules for preprocessing, semantic compression, retrieval, HITL routing, memory follow-up, and review feedback.
- `scripts/` - runnable scripts for data preparation, semantic builds, the HITL UI, and live-data workflow.
- `theory_experiments/` - offline experiment scripts for numeric forecasting, semantic phase prediction, GRU phase prediction, and compression metrics.
- `data/noaa_5min/` - final NOAA/ASOS 5-minute station datasets used by the offline theory benchmark.
- `data/semantic/` - 5-minute semantic states, embeddings, regime profiles, and supporting files.
- `artifacts/` - saved encoder/tokenizer/search-index/model artifacts needed by the pipeline.
- `results/5min_two_seed_experiments/` - primary KBOS two-seed result tables and compression metrics used for thesis writing.
- `results/kbos_llm_label_5min_two_seed_experiments/` - KBOS rerun after LLM-assisted regime-label refinement.
- `results/kama_5min_two_seed_experiments/` - second-station robustness result tables using the same two-seed protocol.
- `results/kama_gradient_boosting_experiments/` - KAMA tabular gradient-boosting tests over raw and compressed representations.

## Final Experimental Setup

The final offline theory benchmark uses NOAA/ASOS 5-minute observations, not the older hourly setup. The primary reported station is KBOS. A second station run is included as a robustness check under `results/kama_5min_two_seed_experiments/`. It uses the same fixed horizons, seeds, semantic compression setup, and tuned model configurations so the comparison tests whether the KBOS trends are station-specific.

The current KBOS semantic labels have also been rerun through post-clustering LLM label refinement. This changes only the human-readable regime names/explanations; token IDs, embeddings, train/test splits, and quantitative evaluation remain non-LLM.

The HITL/live demo is separate: it uses the live AviationWeather/METAR stream through a semantic-state adapter, so it should be described as a live METAR semantic review layer rather than as a 5-minute live benchmark.

- Seeds: `42`, `123`
- Numeric horizons: `1h`, `3h`, `6h`, `12h`
- Five-minute horizon steps: `12`, `36`, `72`, `144`
- Numeric methods: raw LSTM, PCA-compressed LSTM, LSTM-compressed LSTM
- Phase methods: transition-count predictor and GRU phase predictor
- HITL deployment path: transition-count predictor with evidence, historical analogs, memory/RAG follow-up, and review feedback

## Final Output Files

Use these files when writing primary KBOS tables/figures:

- `results/5min_two_seed_experiments/numeric_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/phase_transition_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/gru_phase_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/5min_two_seed_experiments/compression_metrics_5min_summary.json`
- `results/5min_two_seed_experiments/5min_two_seed_manifest.json`

Updated KBOS outputs after LLM-assisted regime-label refinement:

- `results/kbos_llm_label_5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/kbos_llm_label_5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/kbos_llm_label_5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/kbos_llm_label_5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/kbos_llm_label_5min_two_seed_experiments/compression_metrics_5min_summary.json`

Use these files for the second-station robustness check:

- `results/kama_5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/kama_5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/kama_5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/kama_5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/kama_5min_two_seed_experiments/compression_metrics_5min_summary.json`

Use these files for the KAMA gradient-boosting extension:

- `results/kama_gradient_boosting_experiments/gb_summary.csv`
- `results/kama_gradient_boosting_experiments/gb_best_configs.csv`
- `results/kama_gradient_boosting_experiments/gb_final_results.csv`
- `results/kama_gradient_boosting_experiments/gb_vs_existing_lstm_numeric_summary.csv`

## Main Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the primary two-seed 5-minute benchmark:

```powershell
python theory_experiments\run_5min_two_seed_final_experiments.py
```

Compute primary compression metrics:

```powershell
python theory_experiments\compute_compression_metrics.py
```

Run the KBOS rerun after LLM-assisted regime-label refinement:

```powershell
python theory_experiments\run_kbos_llm_label_two_seed_experiments.py
python theory_experiments\compute_kbos_llm_label_compression_metrics.py
```

Normalize an additional ASOS station CSV to the thesis schema:

```powershell
python scripts\prepare_asos_5min_station.py `
  --input_csv "C:\Users\Admin\Downloads\kama_5min_2024.csv" `
  --output_parquet "data\noaa_5min\KAMA_2024_5min.parquet"
```

Build semantic states for the second station:

```powershell
python scripts\run_semantic_build.py `
  --data "data\noaa_5min\KAMA_2024_5min.parquet" `
  --window_size 48 `
  --step_size 12 `
  --components 8 `
  --clusters 8 `
  --run_name kama_5min_phase
```

Run the second-station robustness benchmark:

```powershell
python theory_experiments\run_kama_5min_two_seed_experiments.py
python theory_experiments\compute_kama_compression_metrics.py
```

Run the KAMA gradient-boosting extension:

```powershell
python theory_experiments\run_kama_gradient_boosting_experiments.py
```

Run the HITL UI:

```powershell
$env:SEMANTIC_LLM_MODEL="gpt-4o-mini"
python scripts\run_hitl_ui.py
```

Open the browser URL printed by the server. The UI uses fixed thesis defaults: LLM follow-up on, memory/RAG on, live METAR semantic state on, transition predictor, top-3 phase candidates, and five historical analogs.

## Result Interpretation Notes

- Raw LSTM is the strongest direct numeric forecaster on the final 5-minute benchmark.
- PCA-compressed LSTM is the stronger compressed numeric variant.
- Semantic compression should be framed as an interpretability/reviewability layer, not as a universal numeric-accuracy improvement.
- GRU is the strongest phase predictor offline.
- Transition-count remains the deployment/HITL predictor because it returns inspectable support counts and historical analogs.
- Compression metrics: 144 raw scalar values per 4-hour window become a 36-dimensional physical summary, then an 8-dimensional PCA embedding, and finally one of 8 regime tokens.
- The second-station run follows the same trend: raw LSTM remains strongest for exact numeric forecasting, PCA is the stronger compressed numeric variant, and GRU is the strongest offline phase predictor.
- The KAMA gradient-boosting extension shows that compressed semantic features are more competitive when paired with a tabular nonlinear learner, especially at medium/long horizons.

## Notes

- API keys are not included. Set `OPENAI_API_KEY` only if LLM-backed follow-up or LLM-assisted regime-label refinement is needed.
- LLM assistance is only used after clustering to refine human-readable regime names and interpretations. Embeddings, token assignments, forecasting models, and quantitative metrics are non-LLM.
- Runtime logs, virtual environments, cache folders, and feedback logs are excluded.
- The older hourly offline benchmark is not the final theory benchmark and should only be treated as background/ablation material. The live HITL adapter may still operate at METAR/live-feed cadence depending on available observations.
