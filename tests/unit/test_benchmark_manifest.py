"""Benchmark manifest validation tests."""

from __future__ import annotations


def test_validate_benchmark_index_manifest_rejects_missing(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import validate_benchmark_index_manifest

    out = validate_benchmark_index_manifest(project_tmp_path / "missing")

    assert out["valid"] is False
    assert out["reason"] == "missing_index_manifest"


def test_validate_benchmark_index_manifest_accepts_unfiltered_trec_index(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_bm25_index_manifest,
        validate_benchmark_index_manifest,
        write_json,
    )

    index_dir = project_tmp_path / "bm25"
    index_dir.mkdir()
    (index_dir / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    manifest = make_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        filters_applied=[],
    )
    write_json(index_dir / BENCHMARK_INDEX_MANIFEST, manifest)

    out = validate_benchmark_index_manifest(index_dir)

    assert out["valid"] is True
    assert out["doc_count"] == 1


def test_validate_benchmark_index_manifest_rejects_destructive_filters(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_bm25_index_manifest,
        validate_benchmark_index_manifest,
        write_json,
    )

    index_dir = project_tmp_path / "bm25"
    index_dir.mkdir()
    (index_dir / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    manifest = make_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        filters_applied=["interventional_only"],
    )
    write_json(index_dir / BENCHMARK_INDEX_MANIFEST, manifest)

    out = validate_benchmark_index_manifest(index_dir)

    assert out["valid"] is False
    assert out["reason"] == "destructive_filters_applied"


def test_validate_benchmark_index_manifest_accepts_fielded_trec_index(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_fielded_bm25_index_manifest,
        validate_benchmark_index_manifest,
        write_json,
    )
    from trial_matcher.retrieval.fielded_bm25 import BM25_FIELD_NAMES

    index_dir = project_tmp_path / "bm25_fielded"
    for field in BM25_FIELD_NAMES:
        field_dir = index_dir / field
        field_dir.mkdir(parents=True)
        (field_dir / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    manifest = make_fielded_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        fields=list(BM25_FIELD_NAMES),
        weights={field: 1.0 for field in BM25_FIELD_NAMES},
        filters_applied=[],
    )
    write_json(index_dir / BENCHMARK_INDEX_MANIFEST, manifest)

    out = validate_benchmark_index_manifest(index_dir)

    assert out["valid"] is True
    assert out["index_kind"] == "bm25_fielded_rrf"


def test_validate_benchmark_index_manifest_rejects_incomplete_fielded_index(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_fielded_bm25_index_manifest,
        validate_benchmark_index_manifest,
        write_json,
    )

    index_dir = project_tmp_path / "bm25_fielded"
    (index_dir / "all").mkdir(parents=True)
    (index_dir / "all" / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    manifest = make_fielded_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        fields=["all"],
        weights={"all": 1.0},
        filters_applied=[],
    )
    write_json(index_dir / BENCHMARK_INDEX_MANIFEST, manifest)

    out = validate_benchmark_index_manifest(index_dir)

    assert out["valid"] is False
    assert out["reason"] == "missing_fielded_subindices"


def test_validate_benchmark_corpus_alignment_rejects_agent_corpus_mismatch(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_bm25_index_manifest,
        make_corpus_manifest,
        validate_benchmark_corpus_alignment,
        write_json,
    )

    corpus_dir = project_tmp_path / "official_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "NCT00000001.xml").write_text("<clinical_study />", encoding="utf-8")
    wrong_corpus_dir = project_tmp_path / "local_snapshot"
    wrong_corpus_dir.mkdir()
    corpus_manifest_path = project_tmp_path / "corpus_manifest.json"
    corpus_manifest = make_corpus_manifest(
        corpus_dir=corpus_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        filters_applied=[],
    )
    write_json(corpus_manifest_path, corpus_manifest)

    index_dir = project_tmp_path / "bm25"
    index_dir.mkdir()
    (index_dir / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    index_manifest = make_bm25_index_manifest(
        index_dir=index_dir,
        doc_count=1,
        source="TREC Clinical Trials 2021",
        snapshot_date="2021-04-27",
        source_format="ctgov_legacy_xml",
        filters_applied=[],
        corpus_manifest_path=corpus_manifest_path,
    )
    write_json(index_dir / BENCHMARK_INDEX_MANIFEST, index_manifest)

    out = validate_benchmark_corpus_alignment(index_dir, wrong_corpus_dir)

    assert out["valid"] is False
    assert out["reason"] == "agent_corpus_dir_mismatch"


def test_validate_benchmark_corpus_alignment_accepts_matching_official_corpus(project_tmp_path):
    from trial_matcher.ingestion.benchmark_manifest import (
        BENCHMARK_INDEX_MANIFEST,
        make_bm25_index_manifest,
        make_corpus_manifest,
        validate_benchmark_corpus_alignment,
        write_json,
    )

    corpus_dir = project_tmp_path / "official_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "NCT00000001.xml").write_text("<clinical_study />", encoding="utf-8")
    corpus_manifest_path = project_tmp_path / "corpus_manifest.json"
    write_json(
        corpus_manifest_path,
        make_corpus_manifest(
            corpus_dir=corpus_dir,
            doc_count=1,
            source="TREC Clinical Trials 2021",
            snapshot_date="2021-04-27",
            source_format="ctgov_legacy_xml",
            filters_applied=[],
        ),
    )

    index_dir = project_tmp_path / "bm25"
    index_dir.mkdir()
    (index_dir / "nct_ids.txt").write_text("NCT00000001\n", encoding="utf-8")
    write_json(
        index_dir / BENCHMARK_INDEX_MANIFEST,
        make_bm25_index_manifest(
            index_dir=index_dir,
            doc_count=1,
            source="TREC Clinical Trials 2021",
            snapshot_date="2021-04-27",
            source_format="ctgov_legacy_xml",
            filters_applied=[],
            corpus_manifest_path=corpus_manifest_path,
        ),
    )

    out = validate_benchmark_corpus_alignment(index_dir, corpus_dir)

    assert out["valid"] is True
    assert out["doc_count"] == 1
