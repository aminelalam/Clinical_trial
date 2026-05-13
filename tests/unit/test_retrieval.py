"""Retrieval helpers — RRF and filters (no real index needed)."""

from __future__ import annotations

import pytest


def test_rrf_combines_runs():
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.retrieval.hybrid import reciprocal_rank_fusion

    run_a = [
        TrialCandidate(nct_id="NCT001", score=10.0, source="bm25", rank=1),
        TrialCandidate(nct_id="NCT002", score=5.0, source="bm25", rank=2),
    ]
    run_b = [
        TrialCandidate(nct_id="NCT002", score=0.9, source="dense", rank=1),
        TrialCandidate(nct_id="NCT003", score=0.7, source="dense", rank=2),
    ]
    fused = reciprocal_rank_fusion([run_a, run_b], k=60, top_k=3)
    nct_ids = [c.nct_id for c in fused]
    assert nct_ids[0] == "NCT002"  # appears at rank 2 in A and 1 in B
    assert "NCT001" in nct_ids
    assert "NCT003" in nct_ids


def test_rrf_weights_can_keep_noisy_dense_secondary():
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.retrieval.hybrid import reciprocal_rank_fusion

    bm25 = [
        TrialCandidate(nct_id="BM25_TOP", score=10.0, source="bm25", rank=1),
        TrialCandidate(nct_id="SHARED", score=9.0, source="bm25", rank=2),
    ]
    dense = [
        TrialCandidate(nct_id="DENSE_ONLY", score=0.99, source="dense", rank=1),
        TrialCandidate(nct_id="SHARED", score=0.98, source="dense", rank=2),
    ]

    fused = reciprocal_rank_fusion([bm25, dense], k=60, top_k=3, weights=[1.0, 0.1])

    assert [c.nct_id for c in fused][:2] == ["SHARED", "BM25_TOP"]


def test_bm25_retriever_uses_configured_index_dir(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from trial_matcher.retrieval import bm25 as bm25_mod

    base = Path(".test_tmp")
    configured = base / "bm25_custom"
    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            bm25_index_dir=configured,
            bm25_k1=1.5,
            bm25_b=0.75,
        ),
        paths=SimpleNamespace(indices_dir=base / "indices"),
    )
    monkeypatch.setattr(bm25_mod, "get_settings", lambda: settings)

    retriever = bm25_mod.BM25Retriever()

    assert retriever.index_dir == configured


def test_trial_fielded_bm25_texts_are_separated(sample_trial):
    assert "HER2+ Breast Cancer" in sample_trial.text_for_bm25_field("condition_title")
    assert "Pregnant or breastfeeding" in sample_trial.text_for_bm25_field("eligibility")
    assert "drug X" in sample_trial.text_for_bm25_field("intervention")
    assert "Study evaluates" in sample_trial.text_for_bm25_field("summary_description")


def test_fielded_bm25_fusion_keeps_field_metadata(monkeypatch):
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.retrieval.fielded_bm25 import FieldedBM25Retriever

    class FakeRetriever:
        def __init__(self, run):
            self.run = run

        def retrieve(self, _query, k):
            return self.run[:k]

    runs = {
        "all": [
            TrialCandidate(nct_id="NCT_A", score=10.0, source="bm25", rank=1),
            TrialCandidate(nct_id="NCT_B", score=9.0, source="bm25", rank=2),
        ],
        "condition_title": [
            TrialCandidate(nct_id="NCT_B", score=11.0, source="bm25", rank=1),
            TrialCandidate(nct_id="NCT_C", score=3.0, source="bm25", rank=2),
        ],
    }
    retriever = FieldedBM25Retriever(
        "unused",
        fields=("all", "condition_title"),
        weights={"all": 1.0, "condition_title": 1.5},
    )
    monkeypatch.setattr(retriever, "_retriever", lambda field: FakeRetriever(runs[field]))

    out = retriever.fuse_field_runs({"all": "cancer", "condition_title": "cancer"}, final_k=3, field_k=10)

    assert out[0].nct_id == "NCT_B"
    assert out[0].source == "bm25_fielded"
    assert out[0].retrieval_metadata["field_ranks"] == {"all": 2, "condition_title": 1}
    assert out[0].retrieval_metadata["field_weights"]["condition_title"] == 1.5


def test_fielded_reranker_can_preserve_retrieval_order_without_loading_model(monkeypatch):
    from types import SimpleNamespace

    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.retrieval import reranker as reranker_mod

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            bm25_mode="fielded",
            fielded_rerank_retrieval_blend=1.0,
            rerank_top_k=2,
            medcpt_cross_encoder="unused-model",
        )
    )
    monkeypatch.setattr(reranker_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        reranker_mod.CrossEncoderReranker,
        "_load",
        lambda self: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    candidates = [
        TrialCandidate(nct_id="NCT_A", score=0.3, rank=1),
        TrialCandidate(nct_id="NCT_B", score=0.2, rank=2),
        TrialCandidate(nct_id="NCT_C", score=0.1, rank=3),
    ]
    out = reranker_mod.CrossEncoderReranker().rerank(
        "query",
        candidates,
        get_text=lambda _nct_id: "unused",
        top_k=2,
    )

    assert [c.nct_id for c in out] == ["NCT_A", "NCT_B"]
    assert out[0].retrieval_metadata["rerank_strategy"] == "retrieval_order"
    assert out[0].retrieval_metadata["cross_encoder_score"] is None


def test_hard_filters_age_block(sample_patient, sample_trial):
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.retrieval.filters import apply_hard_filters, viable_count

    # Make trial require age >= 80
    sample_trial.eligibility.age_range.min_days = 80 * 365
    cands = [TrialCandidate(nct_id=sample_trial.nct_id, score=0.5, source="bm25", rank=1)]
    out = apply_hard_filters(cands, {sample_trial.nct_id: sample_trial}, sample_patient)
    assert out[0].hard_excluded
    assert "age" in (out[0].excluded_reason or "")
    assert viable_count(out) == 0


@pytest.mark.parametrize("status", ["COMPLETED", "TERMINATED", "UNKNOWN"])
def test_hard_filters_benchmark_keeps_inactive_statuses(sample_patient, sample_trial, status):
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.retrieval.filters import apply_hard_filters, viable_count

    sample_trial.status = RecruitmentStatus(status)
    cands = [TrialCandidate(nct_id=sample_trial.nct_id, score=0.5, source="bm25", rank=1)]

    out = apply_hard_filters(
        cands,
        {sample_trial.nct_id: sample_trial},
        sample_patient,
        filter_status=False,
    )

    assert not out[0].hard_excluded
    assert out[0].excluded_reason is None
    assert viable_count(out) == 1


@pytest.mark.parametrize("status", ["COMPLETED", "TERMINATED", "UNKNOWN"])
def test_hard_filters_clinical_active_blocks_inactive_statuses(sample_patient, sample_trial, status):
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.retrieval.filters import apply_hard_filters, viable_count

    sample_trial.status = RecruitmentStatus(status)
    cands = [TrialCandidate(nct_id=sample_trial.nct_id, score=0.5, source="bm25", rank=1)]

    out = apply_hard_filters(cands, {sample_trial.nct_id: sample_trial}, sample_patient)

    assert out[0].hard_excluded
    assert "status" in (out[0].excluded_reason or "")
    assert viable_count(out) == 0
