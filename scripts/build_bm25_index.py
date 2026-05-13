"""Build the BM25S index over a ClinicalTrials.gov corpus.

Usage:
    python scripts/build_bm25_index.py \\
        --ctgov-dir data/ctgov_snapshot \\
        --output-dir data/indices/bm25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trial_matcher.ingestion import filter_corpus, parse_ctgov_dump
from trial_matcher.ingestion.benchmark_manifest import (
    BENCHMARK_INDEX_MANIFEST,
    make_bm25_index_manifest,
    make_corpus_manifest,
    write_json,
)
from trial_matcher.ingestion.ctgov_xml_parser import parse_ctgov_xml_dump
from trial_matcher.logging import logger, setup_logging
from trial_matcher.retrieval.bm25 import BM25Retriever


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ctgov-dir", type=Path, default=Path("data/ctgov_snapshot"))
    p.add_argument("--output-dir", type=Path, default=Path("data/indices/bm25"))
    p.add_argument(
        "--input-format",
        choices=["auto", "ctgov-v2-json", "ctgov-legacy-xml"],
        default="auto",
        help="Corpus format. Use ctgov-legacy-xml for the official TREC snapshot.",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Index the whole corpus (interventional+criteria filter is the default)",
    )
    p.add_argument("--source", default="", help="Manifest source label, e.g. 'TREC Clinical Trials 2021'")
    p.add_argument("--snapshot-date", default="", help="Manifest snapshot date, e.g. 2021-04-27")
    p.add_argument("--corpus-manifest-out", type=Path, default=None)
    p.add_argument("--write-index-manifest", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Streaming trials from {args.ctgov_dir}")
    input_format = args.input_format
    if input_format == "auto":
        input_format = "ctgov-legacy-xml" if any(args.ctgov_dir.rglob("*.xml")) else "ctgov-v2-json"
    raw = (
        parse_ctgov_xml_dump(args.ctgov_dir)
        if input_format == "ctgov-legacy-xml"
        else parse_ctgov_dump(args.ctgov_dir)
    )
    filters_applied: list[str] = []
    if not args.no_filter:
        trials_iter = filter_corpus(raw)
        filters_applied = ["interventional_only", "criteria_non_empty_only", "last_update_min_2018"]
    else:
        trials_iter = raw

    texts: list[str] = []
    nct_ids: list[str] = []
    metadata: list[dict] = []
    for t in trials_iter:
        texts.append(t.primary_text())
        nct_ids.append(t.nct_id)
        metadata.append(
            {
                "nct_id": t.nct_id,
                "title": t.title,
                "phase": t.phase.value,
                "status": t.status.value,
                "last_update": str(t.last_update_date) if t.last_update_date else None,
            }
        )

    logger.info(f"Building BM25 index over {len(texts)} trials")
    retriever = BM25Retriever(args.output_dir)
    retriever.build(texts, nct_ids)
    retriever.save()

    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    logger.info(f"Wrote metadata: {metadata_path}")
    should_write_manifest = bool(
        args.write_index_manifest or args.source or args.snapshot_date or args.corpus_manifest_out
    )
    if should_write_manifest:
        source = args.source or "local ClinicalTrials.gov snapshot"
        snapshot_date = args.snapshot_date or "unknown"
        source_format = "ctgov_legacy_xml" if input_format == "ctgov-legacy-xml" else "ctgov_v2_json"
        if args.corpus_manifest_out:
            corpus_manifest = make_corpus_manifest(
                corpus_dir=args.ctgov_dir,
                doc_count=len(texts),
                source=source,
                snapshot_date=snapshot_date,
                source_format=source_format,
                filters_applied=filters_applied,
            )
            write_json(args.corpus_manifest_out, corpus_manifest)
            logger.info(f"Wrote corpus manifest: {args.corpus_manifest_out}")
        index_manifest = make_bm25_index_manifest(
            index_dir=args.output_dir,
            doc_count=len(texts),
            source=source,
            snapshot_date=snapshot_date,
            source_format=source_format,
            filters_applied=filters_applied,
            corpus_manifest_path=args.corpus_manifest_out,
        )
        index_manifest_path = args.output_dir / BENCHMARK_INDEX_MANIFEST
        write_json(index_manifest_path, index_manifest)
        logger.info(f"Wrote index manifest: {index_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
