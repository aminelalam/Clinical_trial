"""Evaluation metric fallbacks."""

from __future__ import annotations

import json


def test_trec_eval_pure_python_fallback(project_tmp_path):
    import eval.trec_eval as trec_eval

    predictions = {
        "topics": [
            {
                "topic_id": "1",
                "ranked_trials": [
                    {"nct_id": "A", "score": 3.0},
                    {"nct_id": "C", "score": 2.0},
                ],
            }
        ]
    }
    predictions_path = project_tmp_path / "predictions.json"
    qrels_path = project_tmp_path / "qrels.txt"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    qrels_path.write_text("1 Q0 A 2\n1 Q0 B 1\n", encoding="utf-8")

    summary = trec_eval._pure_python_evaluate(
        trec_eval.qrels_to_trec_eval_dict(trec_eval.parse_qrels(qrels_path)),
        trec_eval.predictions_to_run(
            json.loads(predictions_path.read_text(encoding="utf-8"))
        ),
    )

    assert summary["n_topics"] == 1
    assert summary["recip_rank"] == 1.0
    assert summary["recall_20"] == 0.5
    assert summary["P_10"] == 0.1
