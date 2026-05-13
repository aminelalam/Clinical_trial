"""Prediction error analysis against TREC qrels.

This script works with both full-agent outputs (with dossiers) and fast T1-T3
outputs (no questions/dossiers). It classifies disagreements using only the
runner's prediction fields plus optional score components.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trial_matcher.ingestion.trec_parser import parse_qrels


def _pred_grade(row: dict[str, Any]) -> int:
    return int(row.get("predicted_trec_qrel", row.get("trec_qrel", 0)) or 0)


def _gold_lookup(qrels_path: Path) -> dict[tuple[str, str], int]:
    return {(q.topic_id, q.nct_id): q.grade for q in parse_qrels(qrels_path)}


def _safe_fraction(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _error_categories(row: dict[str, Any], gold: int, *, judged: bool) -> list[str]:
    pred = _pred_grade(row)
    cats: list[str] = []
    components = row.get("components") or {}

    if gold == 0 and pred > 0:
        cats.append("retrieval_false_positive")
        cats.append(
            "retrieval_false_positive_judged"
            if judged
            else "retrieval_false_positive_unjudged"
        )
    if gold == 2 and pred < 2:
        cats.append("missed_gold_eligible")
    if gold < 2 and pred == 2:
        cats.append("overcalled_eligible")
    if gold == 1 and pred != 1:
        cats.append("missed_excludes")

    inc_total = int(row.get("n_inclusion_total", 0) or 0)
    inc_met = int(row.get("n_inclusion_met", 0) or 0)
    exc_met = int(row.get("n_exclusion_met", 0) or 0)
    frac_nei = float(row.get("fraction_nei", 0.0) or 0.0)
    inc_frac = _safe_fraction(inc_met, inc_total)

    if gold == 2 and pred < 2:
        if bool(components.get("mandatory_veto", False)) or float(row.get("score", 0.0)) <= -0.99:
            cats.append("hard_veto_gold_eligible")
        if exc_met > 0:
            cats.append("exclusion_met_gold_eligible")
        if inc_total and inc_frac < 0.6:
            cats.append("low_inclusion_support_gold_eligible")
        if frac_nei >= 0.6:
            cats.append("high_nei_gold_eligible")

    retrieval_prior = components.get("retrieval_prior")
    if retrieval_prior is not None and float(retrieval_prior) < 0.25:
        cats.append("low_retrieval_prior")

    return cats or ["generic_disagreement"]


def analyze_errors(predictions_path: Path, qrels_path: Path) -> dict[str, Any]:
    qrels = _gold_lookup(qrels_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    category_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    representative: list[dict[str, Any]] = []
    topic_rows: dict[str, dict[str, Any]] = {}
    top10_false_positives: list[dict[str, Any]] = []

    for topic in predictions.get("topics", []):
        tid = str(topic["topic_id"])
        ranked = topic.get("ranked_trials", [])
        gold_rel = 0
        gold_eligible = 0
        disagreements = 0
        first_rel_rank: int | None = None
        first_eligible_rank: int | None = None

        for rank, row in enumerate(ranked, start=1):
            key = (tid, row["nct_id"])
            judged = key in qrels
            gold = int(qrels.get(key, 0))
            pred = _pred_grade(row)
            confusion[f"pred{pred}_gold{gold}"] += 1
            if gold > 0:
                gold_rel += 1
                first_rel_rank = first_rel_rank or rank
            if gold == 2:
                gold_eligible += 1
                first_eligible_rank = first_eligible_rank or rank
            if rank <= 10 and gold == 0 and pred > 0:
                top10_false_positives.append(
                    {
                        "topic_id": tid,
                        "rank": rank,
                        "nct_id": row["nct_id"],
                        "pred_grade": pred,
                        "score": row.get("score"),
                        "label": row.get("label"),
                        "qrel_judged": judged,
                        "components": row.get("components", {}),
                    }
                )
            if pred == gold:
                continue

            disagreements += 1
            cats = _error_categories(row, gold, judged=judged)
            category_counts.update(cats)
            representative.append(
                {
                    "topic_id": tid,
                    "rank": rank,
                    "nct_id": row["nct_id"],
                    "gold_grade": gold,
                    "qrel_judged": judged,
                    "pred_grade": pred,
                    "score": row.get("score"),
                    "label": row.get("label"),
                    "categories": cats,
                    "n_inclusion_met": row.get("n_inclusion_met"),
                    "n_inclusion_total": row.get("n_inclusion_total"),
                    "n_exclusion_met": row.get("n_exclusion_met"),
                    "n_exclusion_total": row.get("n_exclusion_total"),
                    "fraction_nei": row.get("fraction_nei"),
                    "components": row.get("components", {}),
                }
            )

        diag = topic.get("diagnostics", {})
        topic_rows[tid] = {
            "ranked": len(ranked),
            "gold_relevant_in_ranked": gold_rel,
            "gold_eligible_in_ranked": gold_eligible,
            "first_relevant_rank": first_rel_rank,
            "first_eligible_rank": first_eligible_rank,
            "disagreements": disagreements,
            "top10_false_positive_count": sum(
                1 for item in top10_false_positives if item["topic_id"] == tid
            ),
            "top10_false_positive_components": [
                item
                for item in top10_false_positives
                if item["topic_id"] == tid
            ][:10],
            "candidate_counts": diag.get("candidate_counts", {}),
            "criteria_evaluated": diag.get("criteria_evaluated", 0),
            "criterion_label_counts": diag.get("criterion_label_counts", {}),
            "criterion_evaluator_counts": diag.get("criterion_evaluator_counts", {}),
        }

    no_relevant_topics = [
        tid for tid, row in topic_rows.items() if row["gold_relevant_in_ranked"] == 0
    ]
    no_eligible_topics = [
        tid for tid, row in topic_rows.items() if row["gold_eligible_in_ranked"] == 0
    ]

    def severity_key(item: dict[str, Any]) -> tuple[int, int, float]:
        gold = int(item["gold_grade"])
        pred = int(item["pred_grade"])
        return (abs(gold - pred), -int(item["rank"]), float(item.get("score") or 0.0))

    representatives = sorted(representative, key=severity_key, reverse=True)[:20]

    return {
        "predictions": str(predictions_path),
        "qrels": str(qrels_path),
        "n_topics": len(topic_rows),
        "confusion": dict(confusion),
        "category_counts": dict(category_counts),
        "no_relevant_topics": no_relevant_topics,
        "no_eligible_topics": no_eligible_topics,
        "top10_false_positive_count": len(top10_false_positives),
        "top10_false_positives": top10_false_positives[:50],
        "topic_summaries": topic_rows,
        "representative_cases": representatives,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/error_analysis.json"))
    args = p.parse_args()

    summary = analyze_errors(args.predictions, args.qrels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
