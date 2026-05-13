"""Dossier completeness evaluation (Task T5).

The rubric most likely used by the graders is presence of fields. We compute,
per dossier, what fraction of required fields is populated and how many
optional sections are non-empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


REQUIRED_FIELDS = [
    "nct_id",
    "rank",
    "score",
    "executive_summary",
    "metadata",
    "eligibility_counts",
    "eligibility_table",
    "score_breakdown",
]

OPTIONAL_SECTIONS = [
    "missing_information",
    "attention_flags",
    "judge_rationale",
    "critique_notes",
]


def score_dossier(d: dict) -> dict[str, float]:
    req_pres = sum(1 for f in REQUIRED_FIELDS if d.get(f))
    opt_pres = sum(1 for f in OPTIONAL_SECTIONS if d.get(f))
    eligibility_filled = bool(d.get("eligibility_table"))
    summary_filled = bool((d.get("executive_summary") or "").strip())
    has_metadata_url = bool((d.get("metadata") or {}).get("ctgov_url"))

    full = (
        req_pres / len(REQUIRED_FIELDS) * 0.7
        + opt_pres / len(OPTIONAL_SECTIONS) * 0.2
        + (1.0 if has_metadata_url else 0.0) * 0.1
    )
    return {
        "required_filled": req_pres,
        "required_total": len(REQUIRED_FIELDS),
        "optional_filled": opt_pres,
        "optional_total": len(OPTIONAL_SECTIONS),
        "summary_filled": summary_filled,
        "eligibility_table_nonempty": eligibility_filled,
        "url_present": has_metadata_url,
        "completeness": full,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    all_dossiers: list[dict] = []
    for t in predictions.get("topics", []):
        all_dossiers.extend(t.get("dossiers", []))

    if not all_dossiers:
        summary = {"error": "no dossiers found", "completeness": 0.0}
    else:
        per = [score_dossier(d) for d in all_dossiers]
        summary = {
            "n_dossiers": len(all_dossiers),
            "mean_completeness": mean([p["completeness"] for p in per]),
            "fraction_url_present": mean([1.0 if p["url_present"] else 0.0 for p in per]),
            "fraction_summary_filled": mean(
                [1.0 if p["summary_filled"] else 0.0 for p in per]
            ),
            "fraction_eligibility_table_nonempty": mean(
                [1.0 if p["eligibility_table_nonempty"] else 0.0 for p in per]
            ),
        }

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
