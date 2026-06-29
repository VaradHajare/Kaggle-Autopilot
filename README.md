# Kaggle Auto Competitor

An autonomous, LLM-driven agent that takes a Kaggle competition URL and runs the
full modeling loop end to end: download → EDA → feature engineering → model
selection → Optuna-tuned cross-validation → ensembling → submission → leaderboard
tracking → conditional iteration.

LLM decisions are made at the genuinely ambiguous steps — EDA interpretation,
feature strategy, model ranking, ensemble choice — while the deterministic
mechanics (CV, encoding, validation) are plain code. The LLM provider is
pluggable (`LLM_PROVIDER`): Google Gemini by default, Anthropic Claude optional.

## Install

```bash
uv sync                # full stack (ML libs are heavy)
cp .env.example .env   # then fill in credentials
```

Required credentials in `.env`: an LLM key (`GEMINI_API_KEY` by default — get a free
key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey); or set
`LLM_PROVIDER=anthropic` and provide `ANTHROPIC_API_KEY`), plus `KAGGLE_USERNAME`
and `KAGGLE_KEY`.

## Usage

```bash
# Full pipeline (no auto-submit by default — generates a submission file only)
python -m agent run "https://www.kaggle.com/competitions/<slug>"

# Enable automatic submission to Kaggle
python -m agent run <url> --submit

# Resume from the last checkpoint / discard stale state
python -m agent resume <slug>
python -m agent run <url> --force-restart
```

By default the agent **does not submit** — pass `--submit` (or set `auto_submit: true`
in `configs/agent.yaml`) to upload. The effective daily submission cap is always
`min(SUBMISSION_DAILY_LIMIT, the competition's own limit)`.

## How it works

Nine phases run in order, with `RunState` (in `agent/memory.py`) serialized to
`runs/<slug>/state.json` after each one:

| Phase | What it does |
|------|--------------|
| 0 Bootstrap | Validate URL, credentials, rules acceptance, disk space |
| 1 Ingestion | Download, unzip, detect train/test/sample_submission |
| 2 EDA | Typing, target-column ID, time-series + leakage detection, LLM analysis |
| 3 Feature Engineering | 15-op registry; deferred target/group encoding stored for in-fold use |
| 4 Model Selection | LLM-ranked models, filtered + GBM hard rule |
| 5a Pruning | Importance probe → prune to `max_features` |
| 5b Training | Optuna search per model; deferred encodings fit **in-fold** |
| 6 Ensembling | Weighted avg / rank avg / stacking (fold-averaged test preds) |
| 7 Submission | Map predictions to sample format, post-process, validate |
| 8 Leaderboard | Submit best file, poll for public score |
| 9 Iteration | Re-enter a specific earlier phase if a trigger fires |

The pipeline is designed to keep the validation signal leakage-free; see the module docstrings and tests for the invariants.

## Development

```bash
pytest tests/ --cov=agent --cov-report=term-missing   # 80% coverage gate
pytest tests/test_trainer.py -v                        # one module
```

Tests run entirely against mocked Kaggle/Anthropic APIs — no credentials or
network needed.
