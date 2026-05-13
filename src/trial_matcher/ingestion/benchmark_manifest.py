"""Benchmark corpus/index manifests and validation helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

BENCHMARK_INDEX_MANIFEST = "index_manifest.json"
CORPUS_MANIFEST_VERSION = 1
INDEX_MANIFEST_VERSION = 1
DESTRUCTIVE_FILTER_KEYS = {
    "active_status_only",
    "criteria_non_empty_only",
    "interventional_only",
    "last_update_min",
    "last_update_min_2018",
    "sex_filter",
    "age_filter",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_corpus_manifest(
    *,
    corpus_dir: Path,
    doc_count: int,
    source: str,
    snapshot_date: str,
    source_format: Literal["ctgov_v2_json", "ctgov_legacy_xml"],
    filters_applied: list[str] | None = None,
) -> dict[str, Any]:
    files = sorted(corpus_dir.rglob("*.xml" if source_format == "ctgov_legacy_xml" else "page_*.json"))
    return {
        "manifest_type": "trial_matcher_benchmark_corpus",
        "version": CORPUS_MANIFEST_VERSION,
        "source": source,
        "snapshot_date": snapshot_date,
        "source_format": source_format,
        "doc_count": int(doc_count),
        "filters_applied": list(filters_applied or []),
        "corpus_dir": str(corpus_dir),
        "file_count": len(files),
        "created_at": manifest_now(),
    }


def make_bm25_index_manifest(
    *,
    index_dir: Path,
    doc_count: int,
    source: str,
    snapshot_date: str,
    source_format: Literal["ctgov_v2_json", "ctgov_legacy_xml"],
    filters_applied: list[str] | None = None,
    corpus_manifest_path: Path | None = None,
    indexed_fields: list[str] | None = None,
    index_kind: str = "bm25",
    subindices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nct_ids_path = index_dir / "nct_ids.txt"
    return {
        "manifest_type": "trial_matcher_benchmark_index",
        "version": INDEX_MANIFEST_VERSION,
        "index_kind": index_kind,
        "source": source,
        "snapshot_date": snapshot_date,
        "source_format": source_format,
        "doc_count": int(doc_count),
        "filters_applied": list(filters_applied or []),
        "index_dir": str(index_dir),
        "corpus_manifest_path": str(corpus_manifest_path) if corpus_manifest_path else None,
        "nct_ids_sha256": sha256_file(nct_ids_path) if nct_ids_path.exists() else None,
        "field_policy": {
            "indexed_fields": indexed_fields
            or [
                "brief_title",
                "official_title",
                "brief_summary",
                "detailed_description",
                "condition",
                "keyword",
                "intervention",
                "eligibility_criteria",
            ]
        },
        "subindices": subindices or {},
        "created_at": manifest_now(),
    }


def make_fielded_bm25_index_manifest(
    *,
    index_dir: Path,
    doc_count: int,
    source: str,
    snapshot_date: str,
    source_format: Literal["ctgov_v2_json", "ctgov_legacy_xml"],
    fields: list[str],
    weights: dict[str, float],
    filters_applied: list[str] | None = None,
    corpus_manifest_path: Path | None = None,
) -> dict[str, Any]:
    subindices: dict[str, Any] = {}
    for field in fields:
        field_dir = index_dir / field
        nct_ids_path = field_dir / "nct_ids.txt"
        subindices[field] = {
            "index_kind": "bm25",
            "field": field,
            "index_dir": str(field_dir),
            "doc_count": int(doc_count),
            "weight": float(weights.get(field, 1.0)),
            "nct_ids_sha256": sha256_file(nct_ids_path) if nct_ids_path.exists() else None,
        }
    return make_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=doc_count,
        source=source,
        snapshot_date=snapshot_date,
        source_format=source_format,
        filters_applied=filters_applied,
        corpus_manifest_path=corpus_manifest_path,
        indexed_fields=fields,
        index_kind="bm25_fielded_rrf",
        subindices=subindices,
    )


def load_index_manifest(index_dir: Path | str) -> dict[str, Any] | None:
    path = Path(index_dir) / BENCHMARK_INDEX_MANIFEST
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def load_corpus_manifest(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _same_resolved_path(a: Path | str, b: Path | str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(Path(a)) == str(Path(b))


def validate_benchmark_index_manifest(
    index_dir: Path | str,
    *,
    expected_snapshot_date: str = "2021-04-27",
) -> dict[str, Any]:
    """Validate that an index is safe to use as a benchmark corpus proxy."""
    index_dir = Path(index_dir)
    manifest = load_index_manifest(index_dir)
    if manifest is None:
        return {"valid": False, "reason": "missing_index_manifest", "path": str(index_dir)}
    if manifest.get("manifest_type") != "trial_matcher_benchmark_index":
        return {"valid": False, "reason": "invalid_manifest_type", "path": str(index_dir)}
    index_kind = manifest.get("index_kind")
    if index_kind not in {"bm25", "bm25_fielded_rrf"}:
        return {"valid": False, "reason": "unsupported_index_kind", "path": str(index_dir)}
    if manifest.get("snapshot_date") != expected_snapshot_date:
        return {
            "valid": False,
            "reason": "snapshot_date_mismatch",
            "expected": expected_snapshot_date,
            "actual": manifest.get("snapshot_date"),
            "path": str(index_dir),
        }
    filters = [str(f) for f in (manifest.get("filters_applied") or [])]
    destructive = sorted(set(filters) & DESTRUCTIVE_FILTER_KEYS)
    if destructive:
        return {
            "valid": False,
            "reason": "destructive_filters_applied",
            "filters": destructive,
            "path": str(index_dir),
        }
    if index_kind == "bm25" and not manifest.get("nct_ids_sha256"):
        return {"valid": False, "reason": "missing_nct_ids_hash", "path": str(index_dir)}
    if index_kind == "bm25_fielded_rrf":
        required = {"all", "condition_title", "eligibility", "intervention", "summary_description"}
        subindices = manifest.get("subindices") or {}
        missing = sorted(required - set(subindices))
        if missing:
            return {
                "valid": False,
                "reason": "missing_fielded_subindices",
                "missing": missing,
                "path": str(index_dir),
            }
        for field in required:
            field_payload = subindices.get(field) or {}
            field_path = Path(field_payload.get("index_dir") or index_dir / field)
            if not (field_path / "nct_ids.txt").exists():
                return {
                    "valid": False,
                    "reason": "missing_fielded_nct_ids",
                    "field": field,
                    "path": str(field_path),
                }
            if int(field_payload.get("doc_count") or -1) != int(manifest.get("doc_count") or -2):
                return {
                    "valid": False,
                    "reason": "fielded_doc_count_mismatch",
                    "field": field,
                    "expected": manifest.get("doc_count"),
                    "actual": field_payload.get("doc_count"),
                    "path": str(field_path),
                }
            if not field_payload.get("nct_ids_sha256"):
                return {
                    "valid": False,
                    "reason": "missing_fielded_nct_ids_hash",
                    "field": field,
                    "path": str(field_path),
                }
    return {
        "valid": True,
        "reason": "ok",
        "path": str(index_dir),
        "doc_count": manifest.get("doc_count"),
        "snapshot_date": manifest.get("snapshot_date"),
        "source": manifest.get("source"),
        "index_kind": index_kind,
    }


def validate_benchmark_corpus_alignment(
    index_dir: Path | str,
    corpus_dir: Path | str,
    *,
    expected_snapshot_date: str = "2021-04-27",
) -> dict[str, Any]:
    """Validate that the agent corpus matches the benchmark index manifest.

    This catches the failure mode where retrieval uses the official TREC index
    but the agent then loads a different local CT.gov snapshot, making qrel
    trials unrecoverable after retrieval.
    """
    index_dir = Path(index_dir)
    corpus_dir = Path(corpus_dir)
    manifest = load_index_manifest(index_dir)
    if manifest is None:
        return {
            "valid": False,
            "reason": "missing_index_manifest",
            "index_dir": str(index_dir),
            "corpus_dir": str(corpus_dir),
        }

    corpus_manifest_path = manifest.get("corpus_manifest_path")
    if not corpus_manifest_path:
        return {
            "valid": False,
            "reason": "missing_corpus_manifest_path",
            "index_dir": str(index_dir),
            "corpus_dir": str(corpus_dir),
        }

    corpus_manifest = load_corpus_manifest(corpus_manifest_path)
    if corpus_manifest is None:
        return {
            "valid": False,
            "reason": "missing_or_invalid_corpus_manifest",
            "path": str(corpus_manifest_path),
            "index_dir": str(index_dir),
            "corpus_dir": str(corpus_dir),
        }
    if corpus_manifest.get("manifest_type") != "trial_matcher_benchmark_corpus":
        return {
            "valid": False,
            "reason": "invalid_corpus_manifest_type",
            "path": str(corpus_manifest_path),
            "actual": corpus_manifest.get("manifest_type"),
        }
    if corpus_manifest.get("snapshot_date") != expected_snapshot_date:
        return {
            "valid": False,
            "reason": "corpus_snapshot_date_mismatch",
            "expected": expected_snapshot_date,
            "actual": corpus_manifest.get("snapshot_date"),
            "path": str(corpus_manifest_path),
        }
    filters = [str(f) for f in (corpus_manifest.get("filters_applied") or [])]
    destructive = sorted(set(filters) & DESTRUCTIVE_FILTER_KEYS)
    if destructive:
        return {
            "valid": False,
            "reason": "corpus_destructive_filters_applied",
            "filters": destructive,
            "path": str(corpus_manifest_path),
        }
    if int(corpus_manifest.get("doc_count") or -1) != int(manifest.get("doc_count") or -2):
        return {
            "valid": False,
            "reason": "corpus_index_doc_count_mismatch",
            "corpus_doc_count": corpus_manifest.get("doc_count"),
            "index_doc_count": manifest.get("doc_count"),
            "path": str(corpus_manifest_path),
        }
    manifest_corpus_dir = corpus_manifest.get("corpus_dir")
    if not manifest_corpus_dir:
        return {
            "valid": False,
            "reason": "missing_corpus_dir_in_manifest",
            "path": str(corpus_manifest_path),
        }
    if not _same_resolved_path(manifest_corpus_dir, corpus_dir):
        return {
            "valid": False,
            "reason": "agent_corpus_dir_mismatch",
            "expected": str(manifest_corpus_dir),
            "actual": str(corpus_dir),
            "path": str(corpus_manifest_path),
        }
    if not corpus_dir.exists():
        return {
            "valid": False,
            "reason": "agent_corpus_dir_missing",
            "path": str(corpus_dir),
        }
    return {
        "valid": True,
        "reason": "ok",
        "path": str(corpus_manifest_path),
        "corpus_dir": str(corpus_dir),
        "doc_count": corpus_manifest.get("doc_count"),
        "snapshot_date": corpus_manifest.get("snapshot_date"),
        "source": corpus_manifest.get("source"),
    }
