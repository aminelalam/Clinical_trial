"""Agent-stage qrel audit tests."""

from __future__ import annotations

import json


def test_agent_stage_audit_counts_qrel_hits_from_retrieval_traces(project_tmp_path):
    from eval.agent_stage_audit import audit_predictions

    predictions = {
        "metadata": {"settings": {"include_retrieval_traces": True}},
        "topics": [
            {
                "topic_id": "1",
                "diagnostics": {
                    "trial_not_found_count": 0,
                    "retrieval_traces": {
                        "bm25_candidates": ["NCT_REL", "NCT_IRREL"],
                        "fused_candidates": ["NCT_IRREL", "NCT_ELIG"],
                        "reranked_candidates": ["NCT_ELIG"],
                        "final_candidates": ["NCT_ELIG"],
                    },
                },
                "ranked_trials": [{"nct_id": "NCT_REL"}],
            }
        ],
    }
    predictions_path = project_tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    qrels_path = project_tmp_path / "qrels.txt"
    qrels_path.write_text(
        "\n".join(
            [
                "1 0 NCT_REL 1",
                "1 0 NCT_ELIG 2",
                "1 0 NCT_IRREL 0",
            ]
        ),
        encoding="utf-8",
    )

    out = audit_predictions(predictions_path, qrels_path, cutoffs=[1, 20])

    final = out["aggregate"]["stages"]["final_candidates"]
    assert final["mean_recall_relevant@20"] == 0.5
    assert final["mean_recall_eligible@20"] == 1.0
    ranked = out["aggregate"]["stages"]["ranked_trials"]
    assert ranked["mean_recall_relevant@20"] == 0.5
    assert ranked["mean_recall_eligible@20"] == 0.0
    assert out["aggregate"]["trial_not_found_count_total"] == 0


def test_agent_stage_audit_reports_missing_traces(project_tmp_path):
    from eval.agent_stage_audit import audit_predictions

    predictions_path = project_tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps({"topics": [{"topic_id": "1", "diagnostics": {}}]}),
        encoding="utf-8",
    )
    qrels_path = project_tmp_path / "qrels.txt"
    qrels_path.write_text("1 0 NCT_REL 2\n", encoding="utf-8")

    out = audit_predictions(predictions_path, qrels_path, cutoffs=[20])

    assert out["aggregate"]["topics_missing_traces"] == ["1"]
