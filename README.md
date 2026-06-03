# Wind Semantic Forecasting Thesis Codebase

Clean, independent codebase for the wind semantic forecasting thesis. It contains the final 5-minute NOAA benchmark, semantic compression artifacts, phase-prediction experiments, and the controlled HITL review interface.

## Repository Contents

- `app/` - thesis-relevant Python modules for preprocessing, semantic compression, retrieval, HITL routing, memory follow-up, and review feedback.
- `scripts/` - runnable scripts for the HITL UI and live-data workflow.
- `theory_experiments/` - offline experiment scripts for numeric forecasting, semantic phase prediction, GRU phase prediction, and compression metrics.
- `data/noaa_5min/` - final NOAA 5-minute KBOS dataset used by the benchmark.
- `data/semantic/` - 5-minute semantic states, embeddings, regime profiles, and supporting files.
- `artifacts/` - saved encoder/tokenizer/search-index/model artifacts needed by the pipeline.
- `results/5min_two_seed_experiments/` - final two-seed result tables and compression metrics used for thesis writing.

## Final Experimental Setup

The final benchmark uses NOAA 5-minute KBOS observations, not the older hourly setup.

- Seeds: `42`, `123`
- Numeric horizons: `1h`, `3h`, `6h`, `12h`
- Five-minute horizon steps: `12`, `36`, `72`, `144`
- Numeric methods: raw LSTM, PCA-compressed LSTM, LSTM-compressed LSTM
- Phase methods: transition-count predictor and GRU phase predictor
- HITL deployment path: transition-count predictor with evidence, historical analogs, memory/RAG follow-up, and review feedback

## Final Output Files

Use these files when writing tables/figures:

- `results/5min_two_seed_experiments/numeric_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/numeric_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/phase_transition_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/phase_transition_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/gru_phase_5min_two_seed_final_results.csv`
- `results/5min_two_seed_experiments/gru_phase_5min_two_seed_summary.csv`
- `results/5min_two_seed_experiments/compression_metrics_5min.csv`
- `results/5min_two_seed_experiments/compression_metrics_5min_summary.json`
- `results/5min_two_seed_experiments/5min_two_seed_manifest.json`

## Main Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the final two-seed 5-minute benchmark:

```powershell
python theory_experimentsun_5min_two_seed_final_experiments.py
```

Compute compression metrics:

```powershell
python theory_experiments\compute_compression_metrics.py
```

Run the HITL UI:

```powershell
$env:SEMANTIC_LLM_MODEL="gpt-4o-mini"
python scriptsun_hitl_ui.py
```

Open the browser URL printed by the server. The UI uses fixed thesis defaults: LLM follow-up on, memory/RAG on, live 5-minute state on, transition predictor, top-3 phase candidates, and five historical analogs.

## Result Interpretation Notes

- Raw LSTM is the strongest direct numeric forecaster on the final 5-minute benchmark.
- PCA-compressed LSTM is the stronger compressed numeric variant.
- Semantic compression should be framed as an interpretability/reviewability layer, not as a universal numeric-accuracy improvement.
- GRU is the strongest phase predictor offline.
- Transition-count remains the deployment/HITL predictor because it returns inspectable support counts and historical analogs.
- Compression metrics: 144 raw scalar values per 4-hour window become a 36-dimensional physical summary, then an 8-dimensional PCA embedding, and finally one of 8 regime tokens.

## Notes

- API keys are not included. Set `OPENAI_API_KEY` only if LLM-backed follow-up is needed.
- Runtime logs, virtual environments, cache folders, and feedback logs are excluded.
- The older hourly benchmark is not the final benchmark and should only be treated as background/ablation material.
