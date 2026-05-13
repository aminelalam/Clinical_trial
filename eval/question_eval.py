"""Auto-evaluation of generated clinical questions (Task T4).

Two modes:
1. Heuristic rubric (no LLM) — checks presence of the 6 required elements.
   Yields a score 0-2 per dimension, summed to a 0-12 score per question, normalized to 0-1.
2. LLM-as-judge (optional) — compares generated questions to a gold-standard set
   in `data/trec_ct/gold_questions.jsonl`.

Run with --mode rubric (default) for fast unit-style evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def rubric_score(q: dict) -> dict[str, float]:
    """Score a question on 5 dimensions, each 0-2 (10 max), normalized to 0-1."""
    s_specific = 2.0 if q.get("data_point") and len(q["data_point"]) > 3 else (
        1.0 if q.get("data_point") else 0.0
    )
    s_temporal = 2.0 if q.get("time_window") else 1.0 if "within" in (q.get("question_text") or "").lower() else 0.0
    s_format = 2.0 if q.get("expected_data_type") and q["expected_data_type"] != "text" else 1.0
    rationale = q.get("rationale") or ""
    s_rationale = 2.0 if "trial" in rationale.lower() and len(rationale) > 30 else 1.0 if rationale else 0.0
    qtext = q.get("question_text") or ""
    s_actionable = 2.0 if 5 <= len(qtext.split()) <= 60 and "?" in qtext else 1.0 if qtext else 0.0

    total = s_specific + s_temporal + s_format + s_rationale + s_actionable
    return {
        "specific": s_specific,
        "temporal": s_temporal,
        "format": s_format,
        "rationale": s_rationale,
        "actionable": s_actionable,
        "total_raw": total,
        "normalized": total / 10.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--mode", choices=["rubric", "llm"], default="rubric")
    p.add_argument("--gold", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    all_questions: list[dict] = []
    for t in predictions.get("topics", []):
        all_questions.extend(t.get("questions", []))

    if args.mode == "rubric":
        scores = [rubric_score(q) for q in all_questions]
        normalized = [s["normalized"] for s in scores]
        summary = {
            "mode": "rubric",
            "n_questions": len(all_questions),
            "mean_score": mean(normalized) if normalized else 0.0,
            "per_dimension": {
                k: mean([s[k] for s in scores]) / 2.0 if scores else 0.0
                for k in ("specific", "temporal", "format", "rationale", "actionable")
            },
        }
    else:
        # LLM-judge mode is left as a TODO since it needs a curated gold set.
        summary = {"mode": "llm", "error": "LLM-judge mode requires --gold gold_questions.jsonl"}

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
