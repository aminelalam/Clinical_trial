"""Retrieval audit regression tests."""

from __future__ import annotations


def test_aggregate_topics_allows_bm25_only_runs():
    from eval.retrieval_audit import aggregate_topics

    rows = [
        {
            "topic_id": "1",
            "stages": {
                "bm25": {
                    "unique_count": 10,
                    "relevant_hits@20": 1,
                    "eligible_hits@20": 0,
                    "recall_relevant@20": 0.5,
                    "recall_eligible@20": 0.0,
                    "judged_fraction@20": 0.2,
                },
                "fused": {
                    "unique_count": 10,
                    "relevant_hits@20": 1,
                    "eligible_hits@20": 0,
                    "recall_relevant@20": 0.5,
                    "recall_eligible@20": 0.0,
                    "judged_fraction@20": 0.2,
                },
            },
            "overlap": {"bm25_dense@100": None},
        }
    ]

    out = aggregate_topics(rows, cutoffs=[20])

    assert out["mean_bm25_dense_jaccard@100"] == 0.0
    assert out["stages"]["bm25"]["mean_recall_relevant@20"] == 0.5


def test_aggregate_topics_includes_fielded_bm25_stages():
    from eval.retrieval_audit import aggregate_topics

    rows = [
        {
            "topic_id": "1",
            "stages": {
                "bm25": {
                    "unique_count": 10,
                    "relevant_hits@20": 0,
                    "eligible_hits@20": 0,
                    "recall_relevant@20": 0.0,
                    "recall_eligible@20": 0.0,
                    "judged_fraction@20": 0.0,
                },
                "bm25_field_condition_title": {
                    "unique_count": 20,
                    "relevant_hits@20": 1,
                    "eligible_hits@20": 1,
                    "recall_relevant@20": 0.25,
                    "recall_eligible@20": 0.5,
                    "judged_fraction@20": 0.1,
                },
            },
            "overlap": {"bm25_dense@100": None},
        }
    ]

    out = aggregate_topics(rows, cutoffs=[20])

    assert out["stages"]["bm25_field_condition_title"]["mean_recall_relevant@20"] == 0.25


def test_missing_reason_counts_separates_corpus_index_and_retrieval():
    from eval.retrieval_audit import missing_reason_counts
    from trial_matcher.models.agent_state import TrialCandidate

    out = missing_reason_counts(
        topic_qrels={
            "NCT_RETRIEVED": 2,
            "NCT_NOT_RETRIEVED": 2,
            "NCT_NOT_INDEXED": 1,
            "NCT_NOT_CORPUS": 1,
            "NCT_IRREL": 0,
        },
        candidates=[TrialCandidate(nct_id="NCT_RETRIEVED", source="bm25", rank=1)],
        corpus_ids={"NCT_RETRIEVED", "NCT_NOT_RETRIEVED", "NCT_NOT_INDEXED"},
        index_ids={"NCT_RETRIEVED", "NCT_NOT_RETRIEVED"},
    )

    assert out["retrieved"] == 1
    assert out["not_retrieved"] == 1
    assert out["not_in_index"] == 1
    assert out["not_in_corpus"] == 1
