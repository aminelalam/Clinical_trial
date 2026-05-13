"""Run the accepted final delivery profile with a short command.

Default:
    python scripts/run_final_delivery.py --overwrite

Topic 75 metric-only check:
    python scripts/run_final_delivery.py --topic-ids 75 --metric-only --run-name CHECK_TOPIC_75 --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FINAL_RUN_NAME = "FINAL_DELIVERY_TREC2021_P12_FULL_75"


def _rel(*parts: str) -> str:
    return str(Path(*parts))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the accepted TREC 2021 P12 delivery profile without having to "
            "paste the full run_mini_eval command."
        )
    )
    parser.add_argument("--year", choices=["2021"], default="2021")
    parser.add_argument("--topic-limit", type=int, default=75)
    parser.add_argument("--topic-ids", default="")
    parser.add_argument("--run-name", default=FINAL_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metric-only",
        action="store_true",
        help="Skip question and dossier generation; TREC and eligibility metrics still run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded run_mini_eval command without executing it.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    if args.topic_limit < 0:
        raise SystemExit("--topic-limit must be >= 0")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_mini_eval.py"),
        "--year",
        args.year,
        "--topic-limit",
        str(args.topic_limit),
        "--top-k",
        "20",
        "--ctgov-dir",
        _rel("data", "trec_ct", "clinical_trials_2021_04_27"),
        "--bm25-mode",
        "fielded",
        "--fielded-bm25-index-dir",
        _rel("data", "indices", "bm25_trec2021_fielded"),
        "--bm25-top-k",
        "1000",
        "--no-dense",
        "--fused-top-k",
        "1000",
        "--rerank-top-k",
        "50",
        "--require-benchmark-index-manifest",
        "--concurrency",
        str(args.concurrency),
        "--mode",
        "benchmark",
        "--max-trials-per-topic",
        "10",
        "--max-criteria-per-trial",
        "15",
        "--output-root",
        str(args.output_root),
        "--run-name",
        args.run_name,
        "--enable-few-shot",
        "--no-self-consistency",
        "--no-llm-judge",
        "--no-self-critique",
        "--include-retrieval-traces",
        "--benchmark-min-inclusion-fraction",
        "0.1",
        "--benchmark-max-nei-fraction",
        "1.0",
        "--enable-multisignal-irrel-heuristic",
        "--benchmark-candidate-selection-policy",
        "top_score",
        "--benchmark-entity-rerank-policy",
        "rerank_final",
        "--benchmark-entity-rerank-weight",
        "0.09",
        "--benchmark-entity-protect-top",
        "3",
        "--benchmark-criterion-evidence-policy",
        "score_adjust",
        "--benchmark-criterion-evidence-weight",
        "0.50",
    ]

    if args.topic_ids:
        cmd.extend(["--topic-ids", args.topic_ids])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.metric_only:
        cmd.extend(["--no-questions", "--no-dossiers"])

    return cmd


def main() -> int:
    args = _parse_args()
    cmd = build_command(args)
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
