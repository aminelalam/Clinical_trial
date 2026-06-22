# Trial Matcher

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Trial Matcher is an agentic AI system for matching patient profiles against
clinical trials. It combines ClinicalTrials.gov ingestion, hybrid retrieval,
criterion-level eligibility assessment, benchmark evaluation, question
generation, and clinician-oriented dossier output.

The project was benchmarked on the TREC Clinical Trials 2021 task and includes
a reproducible final delivery profile.

## What It Does

- Parses patient topics and ClinicalTrials.gov trial records.
- Retrieves candidate trials with BM25, fielded BM25, dense retrieval, and RRF.
- Evaluates inclusion and exclusion criteria with deterministic rules and LLM
  judging.
- Ranks trials with eligibility labels, evidence, rationales, and audit traces.
- Generates follow-up questions and trial dossiers for downstream review.
- Ships evaluation scripts for TREC ranking, eligibility, questions, dossiers,
  error analysis, and stage auditing.

## Repository Layout

```text
.
|-- src/trial_matcher/        Core package
|-- tests/                    Unit and integration tests
|-- scripts/                  Data, indexing, benchmark, and utility scripts
|-- scripts/dev/              Local debugging helpers
|-- eval/                     Evaluation and analysis entry points
|-- banco_few_shot/           Versioned few-shot examples used by tests/runtime
|-- data/corpus/              Lightweight corpus manifests
|-- docs/                     Project documentation
|-- docs/final-delivery/      Final benchmark artifacts and reports
|-- docker-compose.yml        Qdrant service for vector retrieval
|-- Dockerfile                Container build
|-- pyproject.toml            Python package and tool configuration
```

Large datasets, generated indexes, caches, Qdrant storage, and run outputs are
ignored by Git. Recreate them with the scripts below.

## Final Benchmark Snapshot

Final TREC 2021 delivery artifacts are stored in
[`docs/final-delivery/`](docs/final-delivery/).

| Metric | Value |
| --- | ---: |
| MAP | 0.0803 |
| NDCG@10 | 0.5204 |
| Recall@20 | 0.0941 |
| Reciprocal Rank | 0.7638 |
| P@10 | 0.6307 |
| Topics | 75 |

Main artifacts:

- [`trec2021-final-predictions.json`](docs/final-delivery/trec2021-final-predictions.json)
- [`trec-metrics-final.json`](docs/final-delivery/trec-metrics-final.json)
- [`eligibility-metrics-final.json`](docs/final-delivery/eligibility-metrics-final.json)
- [`how-to-run-final-delivery.es.md`](docs/final-delivery/how-to-run-final-delivery.es.md)

## Requirements

- Python 3.11 or 3.12
- Git
- Bash for the download helper scripts
- Optional: Docker, for running Qdrant locally
- Optional: CUDA-capable GPU for faster dense embedding/index work

## Quick Start

```powershell
git clone https://github.com/aminelalam/Clinical_trial.git
cd Clinical_trial

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

copy .env.example .env
```

Edit `.env` with your LLM provider credentials. The project supports Azure
OpenAI, the OpenAI API, Groq, and Ollama through the settings in
`.env.example`.

## Data And Indexes

Download the TREC Clinical Trials snapshot and build the fielded BM25 index:

```powershell
bash scripts/download_trec.sh
python scripts\download_trec_ct_snapshot.py
python scripts\build_fielded_bm25_index.py `
  --ctgov-dir data\trec_ct\clinical_trials_2021_04_27 `
  --output-dir data\indices\bm25_trec2021_fielded `
  --input-format ctgov-legacy-xml `
  --write-index-manifest
```

For vector retrieval, start Qdrant and build a dense index:

```powershell
docker compose up -d qdrant
python scripts\build_dense_index.py --ctgov-dir data\ctgov_snapshot --batch-size 32
```

## Run The Final Delivery Profile

Full 75-topic TREC 2021 benchmark run:

```powershell
python scripts\run_final_delivery.py --overwrite
```

Fast smoke check on topic 75 without question or dossier generation:

```powershell
python scripts\run_final_delivery.py `
  --topic-ids 75 `
  --metric-only `
  --run-name CHECK_TOPIC_75 `
  --overwrite
```

The main generated output is written under:

```text
results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75/
```

## Evaluate Existing Predictions

```powershell
python eval\trec_eval.py `
  --predictions docs\final-delivery\trec2021-final-predictions.json `
  --qrels data\trec_ct\raw\qrels_2021.txt `
  --output results\trec_metrics_check.json

python eval\eligibility_eval.py `
  --predictions docs\final-delivery\trec2021-final-predictions.json `
  --qrels data\trec_ct\raw\qrels_2021.txt `
  --output results\eligibility_metrics_check.json
```

## Run Tests

```powershell
python -m pytest tests\unit -q
python -m compileall src eval scripts tests
```

## API

Start the FastAPI service:

```powershell
uvicorn trial_matcher.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /match`

## Documentation

Project notes, Spanish technical reports, defense notes, and final delivery
artifacts live under [`docs/`](docs/). Start with
[`docs/README.md`](docs/README.md) for the full index.
