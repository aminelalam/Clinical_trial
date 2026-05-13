"""Validate the curated few-shot bank.

Checks:
- Every example parses as a FewShotExample.
- decision is one of {met, not_met, NEI}.
- File-level category matches the JSONL filename's stem prefix
  (e.g., examples in negation.jsonl must have category starting with 'negation').
- No two examples share an identical criterion_text (dedup).
- Every category file has at least 1 example with each of (NEI, not_met) — the
  curation rule that prevents the model defaulting to a single label.
- Total count is the documented v1 size (42).

Exits non-zero if any check fails so reproduce_results.sh can fail fast.

Usage:
    python scripts/validate_few_shot_bank.py [--dir banco_few_shot] [--strict-count]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from trial_matcher.llm.few_shot import FewShotExample

ALLOWED_DECISIONS = {"met", "not_met", "NEI"}
EXPECTED_FILES = {
    "negation": "negation",
    "temporal": "temporal",
    "biomarker_synonym": "biomarker",
    "generic": "",  # generic.jsonl: any sub-category allowed
}
EXPECTED_TOTAL = 42


def validate(directory: Path, strict_count: bool) -> int:
    if not directory.exists():
        print(f"ERROR: {directory} does not exist", file=sys.stderr)
        return 2

    failures: list[str] = []
    by_file: dict[str, list[FewShotExample]] = defaultdict(list)
    seen_criterion_hashes: set[str] = set()
    seen_ids: set[str] = set()

    files = sorted(directory.glob("*.jsonl"))
    if not files:
        failures.append(f"no .jsonl files found in {directory}")

    for f in files:
        stem = f.stem
        if stem not in EXPECTED_FILES:
            failures.append(f"unexpected file: {f.name} (allowed: {sorted(EXPECTED_FILES)})")
            continue
        cat_prefix = EXPECTED_FILES[stem]

        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                ex = FewShotExample.model_validate_json(line)
            except Exception as e:
                failures.append(f"{f.name}:{line_no}: parse error — {e}")
                continue

            # Decision is in the allowed set
            if ex.decision not in ALLOWED_DECISIONS:
                failures.append(
                    f"{f.name}:{line_no}: invalid decision {ex.decision!r}"
                )

            # File-prefix match (skip for generic.jsonl)
            if cat_prefix and not ex.category.startswith(cat_prefix):
                failures.append(
                    f"{f.name}:{line_no}: category {ex.category!r} does not "
                    f"start with expected prefix {cat_prefix!r}"
                )

            # Dedup by criterion text
            ctext = ex.criterion_text.strip().lower()
            if ctext in seen_criterion_hashes:
                failures.append(f"{f.name}:{line_no}: duplicate criterion_text")
            seen_criterion_hashes.add(ctext)

            # Dedup by id (when set)
            if ex.id:
                if ex.id in seen_ids:
                    failures.append(f"{f.name}:{line_no}: duplicate id {ex.id!r}")
                seen_ids.add(ex.id)

            by_file[stem].append(ex)

    # Per-file balance: at least 1 NEI and at least 1 not_met
    for stem, examples in by_file.items():
        decisions = Counter(e.decision for e in examples)
        if decisions.get("NEI", 0) < 1:
            failures.append(f"{stem}.jsonl has 0 NEI examples (need ≥1)")
        if decisions.get("not_met", 0) < 1:
            failures.append(f"{stem}.jsonl has 0 not_met examples (need ≥1)")

    total = sum(len(v) for v in by_file.values())
    if strict_count and total != EXPECTED_TOTAL:
        failures.append(f"expected {EXPECTED_TOTAL} examples total, found {total}")

    # Report
    if failures:
        print("FAIL — few-shot bank validation:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    cats_per_file = {
        stem: sorted({e.category for e in examples})
        for stem, examples in sorted(by_file.items())
    }
    decisions_per_file = {
        stem: dict(Counter(e.decision for e in examples).most_common())
        for stem, examples in sorted(by_file.items())
    }
    print(f"OK — {total} examples across {len(by_file)} files")
    for stem in sorted(by_file):
        print(
            f"  {stem}.jsonl: {len(by_file[stem])} examples, "
            f"sub-categories={cats_per_file[stem]}, "
            f"decisions={decisions_per_file[stem]}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("banco_few_shot"))
    p.add_argument(
        "--strict-count",
        action="store_true",
        default=True,
        help=f"Require exactly {EXPECTED_TOTAL} total examples",
    )
    p.add_argument("--no-strict-count", dest="strict_count", action="store_false")
    args = p.parse_args()
    return validate(args.dir, args.strict_count)


if __name__ == "__main__":
    raise SystemExit(main())
