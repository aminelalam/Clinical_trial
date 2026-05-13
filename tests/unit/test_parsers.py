"""Parsers — TREC topics/qrels and CT.gov v2."""

from __future__ import annotations

from textwrap import dedent


def test_parse_trec_topics(project_tmp_path):
    from trial_matcher.ingestion.trec_parser import parse_topics

    xml = dedent(
        """\
        <topics task="2021 TREC Clinical Trials Track">
          <topic number="1">47-year-old woman with HER2+ breast cancer.</topic>
          <topic number="2">62-year-old man with NSCLC, EGFR mutation.</topic>
        </topics>
        """
    )
    p = project_tmp_path / "topics.xml"
    p.write_text(xml, encoding="utf-8")
    topics = parse_topics(p)
    assert len(topics) == 2
    assert topics[0].topic_id == "1"
    assert "HER2" in topics[0].text


def test_parse_qrels(project_tmp_path):
    from trial_matcher.ingestion.trec_parser import parse_qrels

    p = project_tmp_path / "qrels.txt"
    p.write_text("1 0 NCT00000001 2\n1 0 NCT00000002 1\n2 0 NCT00000003 0\n", encoding="utf-8")
    qrels = parse_qrels(p)
    assert len(qrels) == 3
    assert qrels[0].grade == 2
    assert qrels[1].nct_id == "NCT00000002"


def test_parse_ctgov_v2_minimal():
    from trial_matcher.ingestion.ctgov_parser import parse_ctgov_study

    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT99999999",
                "briefTitle": "Tiny test trial",
            },
            "descriptionModule": {"briefSummary": "A summary."},
            "conditionsModule": {"conditions": ["Hypertension"]},
            "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL"},
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n- Age >= 18\nExclusion Criteria:\n- Pregnant"
                ),
                "minimumAge": "18 Years",
                "sex": "ALL",
            },
            "statusModule": {"overallStatus": "RECRUITING"},
        }
    }
    t = parse_ctgov_study(study)
    assert t is not None
    assert t.nct_id == "NCT99999999"
    assert t.phase.value == "PHASE2"
    assert t.status.value == "RECRUITING"
    assert "Pregnant" in t.eligibility.exclusion_text
