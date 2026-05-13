"""Micro-F1 evaluation at trial level (TREC qrels) and criterion level (custom).

Trial-level (3 classes: irrelevant=0, excludes=1, eligible=2):
- The system's TrialEval.label maps directly to {0,1,2} via .trec_qrel.
- Compute Micro-F1 over (topic_id, nct_id) pairs against the official qrels.

Criterion-level (3 classes: met / not_met / NEI):
- Requires a hand-annotated test set (we provide a small fixture in tests/).
- Computed against `data/trec_ct/criterion_annotations.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trial_matcher.ingestion.trec_parser import parse_qrels


_LABEL_MAP_TRIAL = {"irrelevant": 0, "excludes": 1, "eligible": 2}


def _f1_per_class(y_true: list[int], y_pred: list[int], n_classes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[f"class_{c}_precision"] = precision
        out[f"class_{c}_recall"] = recall
        out[f"class_{c}_f1"] = f1
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    out["micro_precision"] = micro_p
    out["micro_recall"] = micro_r
    out["micro_f1"] = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    out["macro_f1"] = float(
        np.mean([out[f"class_{c}_f1"] for c in range(n_classes)])
    )
    out["accuracy"] = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return out


def trial_level_f1(predictions_path: Path, qrels_path: Path) -> dict[str, float]:
    qrels = parse_qrels(qrels_path)
    qrel_lookup = {(q.topic_id, q.nct_id): q.grade for q in qrels}

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    y_true: list[int] = []
    y_pred: list[int] = []
    for topic in predictions.get("topics", []):
        tid = topic["topic_id"]
        for r in topic.get("ranked_trials", []):
            key = (tid, r["nct_id"])
            if key in qrel_lookup:
                y_true.append(qrel_lookup[key])
                y_pred.append(int(r.get("predicted_trec_qrel", r.get("trec_qrel", 0))))

    if not y_true:
        return {"error": "No overlap between predictions and qrels", "micro_f1": 0.0}
    return _f1_per_class(y_true, y_pred, n_classes=3)


def criterion_level_f1(predictions_path: Path, annotations_path: Path) -> dict[str, float]:
    """Evaluate criterion-level classification against a hand-annotated jsonl.

    Expected annotations format:
      {"topic_id": "1", "nct_id": "NCTxxxxxxx", "criterion_id": "i_3",
       "label": "met"|"not_met"|"NEI"}
    """
    if not annotations_path.exists():
        return {"error": f"Annotations file not found: {annotations_path}"}
    label_map = {"met": 0, "not_met": 1, "NEI": 2}

    gold: dict[tuple[str, str, str], int] = {}
    for line in annotations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        gold[(a["topic_id"], a["nct_id"], a["criterion_id"])] = label_map[a["label"]]

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    y_true: list[int] = []
    y_pred: list[int] = []
    for topic in predictions.get("topics", []):
        tid = topic["topic_id"]
        for d in topic.get("dossiers", []):
            nct = d["nct_id"]
            for row in d.get("eligibility_table", []):
                key = (tid, nct, row["id"])
                if key in gold:
                    y_true.append(gold[key])
                    y_pred.append(label_map[row["label"]])
    if not y_true:
        return {"error": "No overlap between predictions and annotations", "micro_f1": 0.0}
    return _f1_per_class(y_true, y_pred, n_classes=3)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--qrels", type=Path, default=None)
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    summary: dict = {}
    if args.qrels:
        summary["trial_level"] = trial_level_f1(args.predictions, args.qrels)
    if args.annotations:
        summary["criterion_level"] = criterion_level_f1(args.predictions, args.annotations)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
