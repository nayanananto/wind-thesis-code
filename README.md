# Wind Semantic Forecasting Thesis Codebase

This is the clean independent thesis codebase. It intentionally does not contain the old mixed wind_forecasting_app/ folder dump.

## What this repository contains

- pp/: thesis-relevant Python modules only.
  - semantic compression and tokenization
  - semantic retrieval
  - phase forecasting
  - controlled HITL/agentic review pipeline
  - LLM-backed memory follow-up utilities
- scripts/: runnable HITL and semantic workflow scripts.
- 	heory_experiments/: offline experiments for:
  - raw LSTM wind-speed forecasting
  - PCA/statistical-compressed LSTM forecasting
  - LSTM-compressed LSTM forecasting
  - transition-count phase prediction
  - GRU phase-prediction tuning/evaluation
- data/: minimal wind, semantic-state, metadata, and live-state files needed by the thesis code.
- rtifacts/: minimal encoder, tokenizer, search-index, and optional HITL GRU artifacts.
- esults/: saved experiment result tables and reports.

## Thesis framing

The thesis compares raw wind-speed forecasting with compressed semantic forecasting, then uses the semantic representation in a human-in-the-loop review layer. The main offline evaluation studies how raw LSTM, PCA/statistical compression, learned LSTM compression, transition-count phase prediction, and GRU phase prediction behave across horizons.

The HITL layer is a controlled agentic workflow, not a fully autonomous multi-agent system. Specialized nodes handle phase prediction, current-state explanation, retrieval of similar historical windows, memory-grounded follow-up, and human feedback logging.

## Main commands

Install dependencies:

`powershell
pip install -r requirements.txt
`

Run theory experiments:

`powershell
python theory_experiments\run_wind_compression_experiments.py
python theory_experiments\run_phase_transition_experiments.py
python theory_experiments\temp_gru_phase_roi.py
`

Run the HITL UI:

`powershell
$env:SEMANTIC_LLM_MODEL="gpt-4o-mini"
python scripts\run_hitl_ui.py
`

## Notes

- API keys are not included. Set OPENAI_API_KEY in your environment if using LLM-backed follow-up.
- Runtime logs, virtual environments, cache folders, and unrelated old project files are excluded.
- The active HITL phase prediction defaults to transition-count evidence because it is inspectable by human reviewers; GRU is included as the stronger offline neural baseline.
