#!/usr/bin/env bash
# Download MeSH descriptor and supplemental XML files from NLM.
# https://www.nlm.nih.gov/databases/download/mesh.html

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/mesh}"
YEAR="${MESH_YEAR:-2025}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Downloading MeSH descriptor XML for year ${YEAR}..."
wget -nc "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc${YEAR}.gz" -O "desc${YEAR}.xml.gz" || true

echo "Downloading MeSH supplemental records XML for year ${YEAR}..."
wget -nc "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp${YEAR}.gz" -O "supp${YEAR}.xml.gz" || true

echo "Decompressing..."
gunzip -f -k "desc${YEAR}.xml.gz" 2>/dev/null || true
gunzip -f -k "supp${YEAR}.xml.gz" 2>/dev/null || true

cd - > /dev/null

echo ""
echo "MeSH XML placed in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
