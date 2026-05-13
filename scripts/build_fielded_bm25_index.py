"""Build fielded BM25S indexes over a ClinicalTrials.gov corpus.

Example:
    python scripts/build_fielded_bm25_index.py ^
        --ctgov-dir data/trec_ct/clinical_trials_2021_04_27 ^
        --output-dir data/indices/bm25_trec2021_fielded ^
        --input-format ctgov-legacy-xml ^
        --no-filter ^
        --source "TREC Clinical Trials 2021" ^
        --snapshot-date 2021-04-27 ^
        --corpus-manifest-out data/corpus/trec2021_snapshot_manifest.json ^
        --write-index-manifest
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trial_matcher.ingestion import filter_corpus, parse_ctgov_dump
from trial_matcher.ingestion.benchmark_manifest import (
    BENCHMARK_INDEX_MANIFEST,
    make_corpus_manifest,
    make_fielded_bm25_index_manifest,
    write_json,
)
from trial_matcher.ingestion.ctgov_xml_parser import parse_ctgov_xml_dump
from trial_matcher.logging import logger, setup_logging
from trial_matcher.models.trial import Trial
from trial_matcher.retrieval.bm25 import BM25Retriever
from trial_matcher.retrieval.fielded_bm25 import (
    BM25_FIELD_NAMES,
    DEFAULT_FIELD_WEIGHTS,
    trial_text_for_bm25_field,
)


def _raw_trials(ctgov_dir: Path, input_format: str) -> Iterator[Trial]:
    if input_format == "auto":
        input_format = "ctgov-legacy-xml" if any(ctgov_dir.rglob("*.xml")) else "ctgov-v2-json"
    raw = (
        parse_ctgov_xml_dump(ctgov_dir)
        if input_format == "ctgov-legacy-xml"
        else parse_ctgov_dump(ctgov_dir)
    )
    yield from raw


def _trial_iter(ctgov_dir: Path, input_format: str, *, no_filter: bool) -> Iterable[Trial]:
    raw = _raw_trials(ctgov_dir, input_format)
    return raw if no_filter else filter_corpus(raw)


def _resolved_input_format(ctgov_dir: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    return "ctgov-legacy-xml" if any(ctgov_dir.rglob("*.xml")) else "ctgov-v2-json"


def _parse_weights(raw: str) -> dict[str, float]:
    weights = dict(DEFAULT_FIELD_WEIGHTS)
    if not raw.strip():
        return weights
    for part in raw.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Invalid field weight '{part}', expected field=value")
        if key not in BM25_FIELD_NAMES:
            raise ValueError(f"Unknown BM25 field in weight: {key}")
        weights[key] = float(value)
    return weights


def _copy_existing_bm25_index(src: Path, dst: Path) -> list[str]:
    """Copy a validated BM25 index into a field subdirectory and return NCT IDs."""
    nct_ids_path = src / "nct_ids.txt"
    if not nct_ids_path.exists():
        raise FileNotFoundError(f"Cannot reuse BM25 index without nct_ids.txt: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return nct_ids_path.read_text(encoding="utf-8").splitlines()


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctgov-dir", type=Path, default=Path("data/ctgov_snapshot"))
    p.add_argument("--output-dir", type=Path, default=Path("data/indices/bm25_trec2021_fielded"))
    p.add_argument(
        "--input-format",
        choices=["auto", "ctgov-v2-json", "ctgov-legacy-xml"],
        default="auto",
    )
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--source", default="")
    p.add_argument("--snapshot-date", default="")
    p.add_argument("--corpus-manifest-out", type=Path, default=None)
    p.add_argument("--write-index-manifest", action="store_true")
    p.add_argument("--field-weights", default="")
    p.add_argument(
        "--reuse-all-index-dir",
        type=Path,
        default=None,
        help="Copy an existing official BM25 index as the 'all' sub-index instead of rebuilding it",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_format = _resolved_input_format(args.ctgov_dir, args.input_format)
    source_format = "ctgov_legacy_xml" if input_format == "ctgov-legacy-xml" else "ctgov_v2_json"
    filters_applied = [] if args.no_filter else [
        "interventional_only",
        "criteria_non_empty_only",
        "last_update_min_2018",
    ]
    weights = _parse_weights(args.field_weights)

    nct_ids: list[str] = []
    metadata: list[dict] = []
    metadata_copied = False
    doc_count = 0
    for field in BM25_FIELD_NAMES:
        logger.info(f"Building fielded BM25 sub-index: {field}")
        if field == "all" and args.reuse_all_index_dir is not None:
            nct_ids = _copy_existing_bm25_index(args.reuse_all_index_dir, args.output_dir / field)
            doc_count = len(nct_ids)
            source_metadata = args.reuse_all_index_dir / "metadata.jsonl"
            if source_metadata.exists():
                shutil.copy2(source_metadata, args.output_dir / "metadata.jsonl")
                metadata_copied = True
            logger.info(
                f"Reused existing BM25 index for field 'all': "
                f"{args.reuse_all_index_dir} ({doc_count} docs)"
            )
            continue
        texts: list[str] = []
        field_ids: list[str] = []
        for trial in _trial_iter(args.ctgov_dir, input_format, no_filter=args.no_filter):
            texts.append(trial_text_for_bm25_field(trial, field))
            field_ids.append(trial.nct_id)
            if field == "all":
                metadata.append(
                    {
                        "nct_id": trial.nct_id,
                        "title": trial.title,
                        "phase": trial.phase.value,
                        "status": trial.status.value,
                        "last_update": str(trial.last_update_date) if trial.last_update_date else None,
                    }
                )
        if field == "all":
            nct_ids = list(field_ids)
            doc_count = len(nct_ids)
        elif field_ids != nct_ids:
            raise RuntimeError(f"Field {field} produced a different document order/count")

        retriever = BM25Retriever(args.output_dir / field)
        retriever.build(texts, field_ids)
        retriever.save()

    metadata_path = args.output_dir / "metadata.jsonl"
    if not metadata_copied:
        with metadata_path.open("w", encoding="utf-8") as f:
            for row in metadata:
                f.write(json.dumps(row) + "\n")
        logger.info(f"Wrote metadata: {metadata_path}")
    else:
        logger.info(f"Copied metadata: {metadata_path}")

    should_write_manifest = bool(
        args.write_index_manifest or args.source or args.snapshot_date or args.corpus_manifest_out
    )
    if should_write_manifest:
        source = args.source or "local ClinicalTrials.gov snapshot"
        snapshot_date = args.snapshot_date or "unknown"
        if args.corpus_manifest_out:
            corpus_manifest = make_corpus_manifest(
                corpus_dir=args.ctgov_dir,
                doc_count=doc_count,
                source=source,
                snapshot_date=snapshot_date,
                source_format=source_format,
                filters_applied=filters_applied,
            )
            write_json(args.corpus_manifest_out, corpus_manifest)
            logger.info(f"Wrote corpus manifest: {args.corpus_manifest_out}")
        index_manifest = make_fielded_bm25_index_manifest(
            index_dir=args.output_dir,
            doc_count=doc_count,
            source=source,
            snapshot_date=snapshot_date,
            source_format=source_format,
            fields=list(BM25_FIELD_NAMES),
            weights=weights,
            filters_applied=filters_applied,
            corpus_manifest_path=args.corpus_manifest_out,
        )
        index_manifest_path = args.output_dir / BENCHMARK_INDEX_MANIFEST
        write_json(index_manifest_path, index_manifest)
        logger.info(f"Wrote fielded index manifest: {index_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
