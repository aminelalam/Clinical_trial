"""Lazy CTGov corpus index cache."""

from __future__ import annotations

import json


def _study(nct_id: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
        }
    }


def test_lazy_corpus_writes_and_reuses_index_cache(project_tmp_path):
    from trial_matcher.ingestion.lazy_corpus import LazyTrialCorpus

    dump_dir = project_tmp_path / "ctgov"
    dump_dir.mkdir()
    (dump_dir / "page_00000.json").write_text(
        json.dumps([_study("NCT00000001"), _study("NCT00000002")]),
        encoding="utf-8",
    )

    first = LazyTrialCorpus(dump_dir)
    second = LazyTrialCorpus(dump_dir)

    assert len(first) == 2
    assert len(second) == 2
    assert first._index_cache_path().exists()


def test_lazy_corpus_invalidates_index_cache_when_page_changes(project_tmp_path):
    from trial_matcher.ingestion.lazy_corpus import LazyTrialCorpus

    dump_dir = project_tmp_path / "ctgov"
    dump_dir.mkdir()
    page = dump_dir / "page_00000.json"
    page.write_text(json.dumps([_study("NCT00000001")]), encoding="utf-8")
    assert len(LazyTrialCorpus(dump_dir)) == 1

    page.write_text(
        json.dumps([_study("NCT00000001"), _study("NCT00000002")]),
        encoding="utf-8",
    )

    assert len(LazyTrialCorpus(dump_dir)) == 2


def test_lazy_corpus_indexes_legacy_xml(project_tmp_path):
    from trial_matcher.ingestion.lazy_corpus import LazyTrialCorpus

    dump_dir = project_tmp_path / "ctgov_xml"
    dump_dir.mkdir()
    (dump_dir / "NCT00000003.xml").write_text(
        """
        <clinical_study>
          <id_info><nct_id>NCT00000003</nct_id></id_info>
          <brief_title>Legacy XML trial</brief_title>
          <condition>Breast Cancer</condition>
        </clinical_study>
        """,
        encoding="utf-8",
    )

    corpus = LazyTrialCorpus(dump_dir)

    assert len(corpus) == 1
    assert corpus.get("NCT00000003").title == "Legacy XML trial"
