"""Error-analysis helper tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_error_analysis_module():
    path = Path(__file__).resolve().parents[2] / "eval" / "error_analysis.py"
    spec = importlib.util.spec_from_file_location("error_analysis_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_error_analysis_uses_predicted_trec_qrel_and_fast_outputs(project_tmp_path):
    mod = _load_error_analysis_module()
    predictions = project_tmp_path / "predictions.json"
    qrels = project_tmp_path / "qrels.txt"

    predictions.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "1",
                        "ranked_trials": [
                            {
                                "nct_id": "NCT1",
                                "predicted_trec_qrel": 1,
                                "score": -1.0,
                                "label": "excludes",
                                "n_inclusion_met": 1,
                                "n_inclusion_total": 4,
                                "n_exclusion_met": 0,
                                "n_exclusion_total": 2,
                                "fraction_nei": 0.75,
                                "components": {"mandatory_veto": True},
                            },
                            {
                                "nct_id": "NCT2",
                                "predicted_trec_qrel": 2,
                                "score": 0.5,
                                "label": "eligible",
                                "n_inclusion_met": 3,
                                "n_inclusion_total": 3,
                                "n_exclusion_met": 0,
                                "n_exclusion_total": 1,
                                "fraction_nei": 0.0,
                                "components": {"retrieval_prior": 0.9},
                            },
                        ],
                        "diagnostics": {
                            "candidate_counts": {"ranked": 2},
                            "criteria_evaluated": 5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    qrels.write_text("1 0 NCT1 2\n1 0 NCT2 0\n", encoding="utf-8")

    out = mod.analyze_errors(predictions, qrels)

    assert out["confusion"]["pred1_gold2"] == 1
    assert out["confusion"]["pred2_gold0"] == 1
    assert out["category_counts"]["hard_veto_gold_eligible"] == 1
    assert out["category_counts"]["high_nei_gold_eligible"] == 1
    assert out["category_counts"]["overcalled_eligible"] == 1
    assert out["category_counts"]["retrieval_false_positive_judged"] == 1


def test_error_analysis_marks_unjudged_false_positives(project_tmp_path):
    mod = _load_error_analysis_module()
    predictions = project_tmp_path / "predictions.json"
    qrels = project_tmp_path / "qrels.txt"

    predictions.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "1",
                        "ranked_trials": [
                            {
                                "nct_id": "NCT_UNJUDGED",
                                "predicted_trec_qrel": 1,
                                "score": 0.1,
                                "label": "excludes",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    qrels.write_text("1 0 NCT_OTHER 0\n", encoding="utf-8")

    out = mod.analyze_errors(predictions, qrels)

    assert out["category_counts"]["retrieval_false_positive"] == 1
    assert out["category_counts"]["retrieval_false_positive_unjudged"] == 1
    assert out["top10_false_positives"][0]["qrel_judged"] is False
    assert out["representative_cases"][0]["qrel_judged"] is False
