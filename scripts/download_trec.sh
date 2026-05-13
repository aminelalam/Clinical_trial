#!/usr/bin/env bash
# Download TREC Clinical Trials 2021 + 2022 topics and qrels.
# Source: https://trec.nist.gov/data/trials.html
#
# Notes:
# - Topics for 2021 (75 topics) come as a single XML file.
# - Topics for 2022 (50 topics) likewise.
# - qrels are TSV with columns: topic_id 0 nct_id grade
# - URLs may change over time; verify against the official portal if a 404 occurs.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/trec_ct}"
mkdir -p "$OUTPUT_DIR/raw"

cd "$OUTPUT_DIR/raw"

echo "Downloading TREC CT 2021 topics..."
wget -nc -q "https://trec.nist.gov/data/trials/topics2021.xml" -O topics_2021.xml || true

echo "Downloading TREC CT 2021 qrels..."
wget -nc -q "https://trec.nist.gov/data/trials/qrels2021.txt" -O qrels_2021.txt || true

echo "Downloading TREC CT 2022 topics..."
wget -nc -q "https://trec.nist.gov/data/trials/topics2022.xml" -O topics_2022.xml || true

echo "Downloading TREC CT 2022 qrels..."
wget -nc -q "https://trec.nist.gov/data/trials/qrels2022.txt" -O qrels_2022.txt || true

cd - > /dev/null

echo ""
echo "If any file is empty/missing, manually download from the portal:"
echo "  https://trec.nist.gov/data/trials.html"
echo ""
echo "Files placed in: $OUTPUT_DIR/raw"
ls -la "$OUTPUT_DIR/raw" || true
