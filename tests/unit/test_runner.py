"""Runner selection helpers."""

from __future__ import annotations

import pytest


def test_parse_topic_ids_trims_empty_values():
    from trial_matcher.runner import _parse_topic_ids

    assert _parse_topic_ids(" 1, 2,,3 ") == ["1", "2", "3"]
    assert _parse_topic_ids("") is None
    assert _parse_topic_ids(None) is None


def test_select_patients_topic_ids_preserve_requested_order():
    from trial_matcher.runner import _select_patients

    patients = [
        {"topic_id": "1", "text": "one"},
        {"topic_id": "2", "text": "two"},
        {"topic_id": "3", "text": "three"},
    ]

    selected = _select_patients(patients, topic_ids=["3", "1"])

    assert [p["topic_id"] for p in selected] == ["3", "1"]


def test_select_patients_topic_ids_then_limit():
    from trial_matcher.runner import _select_patients

    patients = [
        {"topic_id": "1", "text": "one"},
        {"topic_id": "2", "text": "two"},
        {"topic_id": "3", "text": "three"},
    ]

    selected = _select_patients(patients, topic_ids=["3", "2", "1"], topic_limit=2)

    assert [p["topic_id"] for p in selected] == ["3", "2"]


def test_select_patients_missing_topic_ids_fail_fast():
    from trial_matcher.runner import _select_patients

    patients = [{"topic_id": "1", "text": "one"}]

    with pytest.raises(ValueError, match="2"):
        _select_patients(patients, topic_ids=["2"])


def test_select_patients_negative_limit_fails():
    from trial_matcher.runner import _select_patients

    with pytest.raises(ValueError, match="topic-limit"):
        _select_patients([], topic_limit=-1)


def test_benchmark_competitive_defaults_are_promoted():
    from trial_matcher.config import RunnerSettings

    settings = RunnerSettings()

    assert settings.use_hard_excluded_fill is True
    assert settings.output_top_k == 20
    assert settings.use_dense_retrieval is True
    assert settings.use_retrieval_tail_fill is False
    assert settings.use_irrelevance_heuristic is False
    assert settings.use_multisignal_irrelevance_heuristic is False
    assert settings.benchmark_candidate_selection_policy == "top_score"
    assert settings.benchmark_diverse_keep_top == 9
    assert settings.benchmark_diverse_select_total == 10
    assert settings.benchmark_entity_rerank_policy == "off"
    assert settings.benchmark_entity_rerank_weight == 0.09
    assert settings.benchmark_entity_protect_top == 3
    assert settings.benchmark_criterion_evidence_policy == "off"
    assert settings.benchmark_criterion_evidence_weight == 0.50
    assert settings.benchmark_min_inclusion_fraction == 0.1
    assert settings.benchmark_max_nei_fraction == 1.0


def test_retrieval_settings_expose_candidate_pool_knobs():
    from trial_matcher.config import RetrievalSettings

    settings = RetrievalSettings()

    assert settings.bm25_index_dir is None
    assert settings.fused_top_k == 100


def test_benchmark_index_validation_reports_missing_manifest(project_tmp_path):
    from trial_matcher.runner import _benchmark_index_validation

    out = _benchmark_index_validation(project_tmp_path / "bm25")

    assert out["valid"] is False
    assert out["reason"] == "missing_index_manifest"


def test_retrieval_traces_deduplicate_stage_candidate_ids():
    from trial_matcher.models.agent_state import AgentState, TrialCandidate
    from trial_matcher.runner import _retrieval_traces

    state = AgentState(
        patient_raw="p",
        run_id="1",
        bm25_candidates=[
            TrialCandidate(nct_id="NCT1", rank=1),
            TrialCandidate(nct_id="NCT1", rank=2),
            TrialCandidate(nct_id="NCT2", rank=3),
        ],
        fused_candidates=[TrialCandidate(nct_id="NCT2", rank=1)],
    )

    out = _retrieval_traces(state)

    assert out["bm25_candidates"] == ["NCT1", "NCT2"]
    assert out["bm25"] == ["NCT1", "NCT2"]
    assert out["fused_candidates"] == ["NCT2"]
    assert out["fused"] == ["NCT2"]
    assert out["final_candidates"] == []
    assert out["final"] == []
    assert out["ranked_trials"] == []
