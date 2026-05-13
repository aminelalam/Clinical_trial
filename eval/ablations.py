"""Ablation runner: produces predictions with each major component turned off.

Run after the agent works end-to-end. Compares:
- baseline (everything on)
- no listwise reranker
- no self-consistency
- no verifier (devil's advocate)
- no MeSH expansion
- no agentic re-retrieval

Each variant calls the runner with appropriate flags and stores the metrics
side by side for the memoria's ablation table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


VARIANTS = [
    ("baseline", []),
    ("no_listwise", ["--no-listwise"]),
    ("no_hyde", ["--no-hyde"]),
    ("no_verifier", ["--no-verifier"]),
]


def run_variant(name: str, flags: list[str], input_path: Path, out_dir: Path) -> Path:
    out = out_dir / f"predictions_{name}.json"
    cmd = [
        sys.executable, "-m", "trial_matcher.runner",
        str(input_path), str(out), *flags,
    ]
    print(f"Running variant '{name}': {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def evaluate_variant(predictions: Path, qrels: Path) -> dict:
    # Robust import: works whether the script is launched from project root
    # (`python eval/ablations.py`) or from inside `eval/` (`python ablations.py`).
    try:
        from eval.trec_eval import evaluate as trec_evaluate
    except ImportError:  # pragma: no cover
        from trec_eval import evaluate as trec_evaluate  # type: ignore[no-redef]
    return trec_evaluate(predictions, qrels)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Topics input (XML or JSON)")
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/ablations"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    for name, flags in VARIANTS:
        try:
            preds = run_variant(name, flags, args.input, args.out_dir)
            table[name] = evaluate_variant(preds, args.qrels)
        except Exception as e:
            table[name] = {"error": str(e)}

    out_path = args.out_dir / "ablations_summary.json"
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
