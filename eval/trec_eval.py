"""TREC-style metrics: NDCG@10, Recall@20, MRR@10, P@10.

Uses pytrec_eval when installed and a small pure-Python fallback otherwise.
Reads predictions in the format produced by trial_matcher.runner and qrels in
the standard TREC format.

Usage:
    python eval/trec_eval.py --predictions predictions_2021.json --qrels data/trec_ct/raw/qrels_2021.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import pytrec_eval
except ModuleNotFoundError:  # pragma: no cover - exercised when optional wheel is absent
    pytrec_eval = None

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trial_matcher.ingestion.trec_parser import parse_qrels, qrels_to_trec_eval_dict


def predictions_to_run(predictions: dict) -> dict[str, dict[str, float]]:
    """Convert runner output to {topic_id: {nct_id: score}} expected by pytrec_eval."""
    run: dict[str, dict[str, float]] = {}
    for topic in predictions.get("topics", []):
        tid = topic["topic_id"]
        run[tid] = {}
        for r in topic.get("ranked_trials", []):
            run[tid][r["nct_id"]] = float(r.get("score", 0.0))
    return run


def _pure_python_evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Small fallback for the metrics we report when pytrec_eval is unavailable."""
    per_topic: dict[str, dict[str, float]] = {}
    for topic_id, scores in run.items():
        topic_qrels = qrels.get(topic_id, {})
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        rel_grades = [int(topic_qrels.get(nct_id, 0)) for nct_id, _score in ranked]
        relevant_total = sum(1 for grade in topic_qrels.values() if grade > 0)

        def dcg(grades: list[int]) -> float:
            return sum(
                ((2**grade) - 1) / math.log2(rank + 2)
                for rank, grade in enumerate(grades)
            )

        dcg_10 = dcg(rel_grades[:10])
        ideal_10 = dcg(sorted(topic_qrels.values(), reverse=True)[:10])
        ndcg_10 = dcg_10 / ideal_10 if ideal_10 > 0 else 0.0

        rel_top_10 = sum(1 for grade in rel_grades[:10] if grade > 0)
        rel_top_20 = sum(1 for grade in rel_grades[:20] if grade > 0)
        p_10 = rel_top_10 / 10.0
        recall_20 = rel_top_20 / relevant_total if relevant_total > 0 else 0.0

        recip_rank = 0.0
        for rank, grade in enumerate(rel_grades, start=1):
            if grade > 0:
                recip_rank = 1.0 / rank
                break

        ap_sum = 0.0
        rel_seen = 0
        for rank, grade in enumerate(rel_grades, start=1):
            if grade > 0:
                rel_seen += 1
                ap_sum += rel_seen / rank
        avg_precision = ap_sum / relevant_total if relevant_total > 0 else 0.0

        per_topic[topic_id] = {
            "ndcg_cut_10": ndcg_10,
            "recall_20": recall_20,
            "recip_rank": recip_rank,
            "P_10": p_10,
            "map": avg_precision,
        }

    metrics = ["ndcg_cut_10", "recall_20", "recip_rank", "P_10", "map"]
    summary = {
        metric: mean(topic[metric] for topic in per_topic.values()) if per_topic else 0.0
        for metric in metrics
    }
    summary["n_topics"] = len(per_topic)
    summary["evaluator"] = "pure_python"
    return summary


def evaluate(predictions_path: Path, qrels_path: Path) -> dict[str, Any]:
    qrels = qrels_to_trec_eval_dict(parse_qrels(qrels_path))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    run = predictions_to_run(predictions)

    if pytrec_eval is None:
        return _pure_python_evaluate(qrels, run)

    metrics = {"ndcg_cut_10", "recall_20", "recip_rank", "P_10", "map"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    # Mean over topics
    summary: dict[str, float] = {}
    for m in metrics:
        vals = [v[m] for v in results.values() if m in v]
        summary[m] = mean(vals) if vals else 0.0
    summary["n_topics"] = len(results)
    summary["evaluator"] = "pytrec_eval"
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    summary = evaluate(args.predictions, args.qrels)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
