"""Sweep benchmark trial-label aggregation thresholds on existing predictions.

This is an offline calibration aid. It does not change retrieval ranks or call
the LLM; it remaps already-computed criterion counts to TREC labels so we can
separate retrieval quality from trial-level eligibility calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trial_matcher.ingestion.trec_parser import parse_qrels


def _parse_float_grid(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _label_to_qrel(label: str | None) -> int:
    return {"irrelevant": 0, "excludes": 1, "eligible": 2}.get(str(label or ""), 0)


def _row_pred_qrel(
    row: dict[str, Any],
    *,
    min_inclusion_fraction: float,
    max_nei_fraction: float,
    preserve_veto: bool = True,
) -> int:
    inc_total = int(row.get("n_inclusion_total", 0) or 0)
    inc_met = int(row.get("n_inclusion_met", 0) or 0)
    exc_met = int(row.get("n_exclusion_met", 0) or 0)
    fraction_nei = float(row.get("fraction_nei", 0.0) or 0.0)
    components = row.get("components") or {}
    veto = bool(components.get("mandatory_veto")) or float(row.get("score", 0.0) or 0.0) <= -0.99

    if preserve_veto and veto:
        return 1
    if inc_total > 0 and inc_met == inc_total and exc_met == 0:
        return 2
    required = max(1, int(min_inclusion_fraction * inc_total))
    if inc_total > 0 and inc_met >= required and exc_met == 0 and fraction_nei < max_nei_fraction:
        return 2
    return 1


def _rows(predictions: dict[str, Any], qrels_path: Path) -> list[tuple[int, dict[str, Any]]]:
    qrels = {(q.topic_id, q.nct_id): q.grade for q in parse_qrels(qrels_path)}
    rows: list[tuple[int, dict[str, Any]]] = []
    for topic in predictions.get("topics", []):
        topic_id = str(topic["topic_id"])
        for row in topic.get("ranked_trials", []):
            gold = qrels.get((topic_id, row["nct_id"]))
            if gold is not None:
                rows.append((int(gold), row))
    return rows


def _f1_per_class(labels: list[tuple[int, int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    correct = 0
    for klass in range(3):
        tp = sum(1 for gold, pred in labels if gold == klass and pred == klass)
        fp = sum(1 for gold, pred in labels if gold != klass and pred == klass)
        fn = sum(1 for gold, pred in labels if gold == klass and pred != klass)
        correct += tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"class_{klass}_precision"] = precision
        out[f"class_{klass}_recall"] = recall
        out[f"class_{klass}_f1"] = f1
    out["micro_f1"] = correct / len(labels) if labels else 0.0
    return out


def _evaluate(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    min_inclusion_fraction: float,
    max_nei_fraction: float,
) -> dict[str, Any]:
    labels = [
        (
            gold,
            _row_pred_qrel(
                row,
                min_inclusion_fraction=min_inclusion_fraction,
                max_nei_fraction=max_nei_fraction,
            ),
        )
        for gold, row in rows
    ]
    metrics = _f1_per_class(labels)
    metrics["min_inclusion_fraction"] = min_inclusion_fraction
    metrics["max_nei_fraction"] = max_nei_fraction
    return metrics


def sweep(
    predictions_path: Path,
    qrels_path: Path,
    *,
    min_grid: list[float],
    max_nei_grid: list[float],
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    judged_rows = _rows(predictions, qrels_path)
    current = _f1_per_class(
        [
            (gold, int(row.get("predicted_trec_qrel", _label_to_qrel(row.get("label")))))
            for gold, row in judged_rows
        ]
    )
    grid = [
        _evaluate(
            judged_rows,
            min_inclusion_fraction=min_inc,
            max_nei_fraction=max_nei,
        )
        for min_inc in min_grid
        for max_nei in max_nei_grid
    ]
    best_class2 = max(grid, key=lambda item: (item["class_2_f1"], item["micro_f1"]))
    best_micro = max(grid, key=lambda item: (item["micro_f1"], item["class_2_f1"]))
    return {
        "predictions": str(predictions_path),
        "qrels": str(qrels_path),
        "judged_rows": len(judged_rows),
        "current": current,
        "best_by_class_2_f1": best_class2,
        "best_by_micro_f1": best_micro,
        "grid": grid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--min-inclusion-grid",
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--max-nei-grid",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    args = parser.parse_args()

    result = sweep(
        args.predictions,
        args.qrels,
        min_grid=_parse_float_grid(args.min_inclusion_grid),
        max_nei_grid=_parse_float_grid(args.max_nei_grid),
    )
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
