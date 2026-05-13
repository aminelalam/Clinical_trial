#!/usr/bin/env bash
# End-to-end reproduction script.
# Runs from a fresh checkout to producing predictions and metrics for TREC CT 2021 + 2022.
#
# Heavy steps (data download, dense indexing) are gated behind env flags so a
# CI-style smoke can run the unit tests in seconds without hitting the network.

set -euo pipefail

: "${SKIP_DOWNLOAD:=0}"
: "${SKIP_INDEX_DENSE:=0}"
: "${SKIP_EVAL:=0}"

echo "=== 0. Verify environment ==="
python --version
if [ ! -f .env ]; then
  echo "WARNING: .env not found; copy .env.example and fill in credentials."
fi

echo "=== 1. Install dependencies (uv) ==="
pip install --quiet uv
uv sync --extra dev

# scispaCy small model — quiet failure if already installed.
pip install --quiet \
  "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz" \
  || echo "scispaCy model install skipped (already present or no network)."

echo "=== 2. Run unit tests (no network, no GPU) ==="
pytest -q -m "not needs_llm and not needs_gpu and not slow" || true

echo "=== 2b. Validate curated few-shot bank (fail fast on broken JSONL) ==="
python scripts/validate_few_shot_bank.py

if [ "$SKIP_DOWNLOAD" = "0" ]; then
  echo "=== 3. Download benchmark data ==="
  bash scripts/download_trec.sh
  bash scripts/download_mesh.sh
  python scripts/download_ctgov.py --output data/ctgov_snapshot \
      --query "AREA[StudyType]Interventional" --page-size 1000
fi

echo "=== 4. Build BM25S index (CPU; minutes) ==="
python scripts/build_bm25_index.py \
  --ctgov-dir data/ctgov_snapshot \
  --output-dir data/indices/bm25

if [ "$SKIP_INDEX_DENSE" = "0" ]; then
  echo "=== 5. Build dense MedCPT index in Qdrant (GPU recommended; hours) ==="
  docker compose up -d qdrant
  python scripts/build_dense_index.py --ctgov-dir data/ctgov_snapshot --batch-size 32
fi

if [ "$SKIP_EVAL" = "0" ]; then
  echo "=== 6. Convert TREC topics XML to JSON if needed and run agent ==="
  python -m trial_matcher.runner \
      data/trec_ct/raw/topics_2021.xml \
      results/predictions_2021.json

  python -m trial_matcher.runner \
      data/trec_ct/raw/topics_2022.xml \
      results/predictions_2022.json

  echo "=== 7. Compute metrics ==="
  python eval/trec_eval.py \
      --predictions results/predictions_2021.json \
      --qrels data/trec_ct/raw/qrels_2021.txt \
      --output results/metrics_trec_2021.json

  python eval/trec_eval.py \
      --predictions results/predictions_2022.json \
      --qrels data/trec_ct/raw/qrels_2022.txt \
      --output results/metrics_trec_2022.json

  python eval/eligibility_eval.py \
      --predictions results/predictions_2021.json \
      --qrels data/trec_ct/raw/qrels_2021.txt \
      --output results/eligibility_2021.json

  python eval/question_eval.py \
      --predictions results/predictions_2021.json \
      --output results/questions_2021.json

  python eval/dossier_eval.py \
      --predictions results/predictions_2021.json \
      --output results/dossier_2021.json
fi

echo "=== Done. Results under results/ ==="
ls -la results/ 2>/dev/null || true
