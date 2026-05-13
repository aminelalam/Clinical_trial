"""Run a reproducible P0 mini-evaluation.

Default run:
    python scripts/run_mini_eval.py --overwrite

The script intentionally keeps the first mini-eval bounded. It disables the
expensive optional agent features by default and records every command in a
manifest so a run can be repeated or audited later.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _set_bool_env(env: dict[str, str], key: str, value: bool) -> None:
    env[key] = "true" if value else "false"


def _run_command(
    name: str,
    cmd: list[str],
    out_dir: Path,
    manifest: dict[str, Any],
    env: dict[str, str],
) -> None:
    stdout_path = out_dir / f"{name}.out"
    stderr_path = out_dir / f"{name}.err"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    record = {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    manifest.setdefault("commands", []).append(record)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}; see {stderr_path}")


def _prediction_summary(predictions_path: Path) -> dict[str, Any]:
    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    topics = data.get("topics", [])
    with_errors = [t["topic_id"] for t in topics if t.get("error")]
    ranked_counts = {
        t["topic_id"]: len(t.get("ranked_trials", []))
        for t in topics
    }
    return {
        "n_topics": len(topics),
        "topics_with_error": with_errors,
        "topics_with_ranked_trials": sum(1 for n in ranked_counts.values() if n > 0),
        "ranked_trials_by_topic": ranked_counts,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the bounded P0 TREC mini-eval.")
    p.add_argument("--year", choices=["2021", "2022"], default="2021")
    p.add_argument("--topic-limit", type=int, default=5)
    p.add_argument("--topic-ids", default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ctgov-dir", type=Path, default=None)
    p.add_argument("--bm25-mode", choices=["single", "fielded"], default=None)
    p.add_argument("--bm25-index-dir", type=Path, default=None)
    p.add_argument("--fielded-bm25-index-dir", type=Path, default=None)
    p.add_argument("--bm25-top-k", type=int, default=0)
    p.add_argument("--dense-top-k", type=int, default=0)
    p.add_argument("--dense-rrf-weight", type=float, default=-1.0)
    p.add_argument("--fused-top-k", type=int, default=0)
    p.add_argument("--rerank-top-k", type=int, default=0)
    p.add_argument("--listwise-top-k", type=int, default=0)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--require-benchmark-index-manifest", action="store_true")
    p.add_argument("--no-benchmark-index-manifest-check", action="store_true")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--mode", choices=["benchmark", "clinical_active"], default="benchmark")
    p.add_argument("--max-trials-per-topic", type=int, default=3)
    p.add_argument("--max-criteria-per-trial", type=int, default=10)
    p.add_argument("--output-root", type=Path, default=Path("results/experiments"))
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--enable-listwise", action="store_true")
    p.add_argument("--enable-hyde", action="store_true")
    p.add_argument("--enable-verifier", action="store_true")
    p.add_argument("--enable-few-shot", action="store_true")
    p.add_argument("--no-self-consistency", action="store_true")
    p.add_argument("--no-sc", action="store_true")
    p.add_argument("--no-llm-judge", action="store_true")
    p.add_argument("--no-self-critique", action="store_true")
    p.add_argument("--no-questions", action="store_true")
    p.add_argument("--no-dossiers", action="store_true")
    p.add_argument("--enable-criterion-triage", action="store_true")
    p.add_argument("--enable-section-header-policy", action="store_true")
    p.add_argument("--benchmark-soft-veto", action="store_true")
    p.add_argument("--enable-hard-excluded-fill", action="store_true")
    p.add_argument("--no-hard-excluded-fill", action="store_true")
    p.add_argument("--enable-retrieval-tail-fill", action="store_true")
    p.add_argument("--include-retrieval-traces", action="store_true")
    p.add_argument(
        "--benchmark-candidate-selection-policy",
        choices=["top_score", "diverse_top10"],
        default="top_score",
    )
    p.add_argument("--benchmark-diverse-keep-top", type=int, default=9)
    p.add_argument("--benchmark-diverse-select-total", type=int, default=10)
    p.add_argument(
        "--benchmark-entity-rerank-policy",
        choices=["off", "audit", "rerank_final"],
        default="off",
    )
    p.add_argument("--benchmark-entity-rerank-weight", type=float, default=None)
    p.add_argument("--benchmark-entity-protect-top", type=int, default=None)
    p.add_argument(
        "--benchmark-criterion-evidence-policy",
        choices=["off", "score_adjust"],
        default="off",
    )
    p.add_argument("--benchmark-criterion-evidence-weight", type=float, default=None)
    p.add_argument("--enable-irrel-heuristic", action="store_true")
    p.add_argument("--no-irrel-heuristic", action="store_true")
    p.add_argument("--enable-multisignal-irrel-heuristic", action="store_true")
    p.add_argument("--no-multisignal-irrel-heuristic", action="store_true")
    p.add_argument("--irrelevant-max-retrieval-prior", type=float, default=None)
    p.add_argument("--irrelevant-min-signal-count", type=int, default=None)
    p.add_argument("--benchmark-min-inclusion-fraction", type=float, default=None)
    p.add_argument("--benchmark-max-nei-fraction", type=float, default=None)
    p.add_argument("--topic-timeout-seconds", type=int, default=0)
    return p.parse_args()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> int:
    args = _parse_args()
    if args.topic_limit < 0:
        raise SystemExit("--topic-limit must be >= 0")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    run_name = args.run_name or f"mini_trec{args.year}_p0"
    out_dir = ROOT / args.output_root / run_name
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topics_path = ROOT / "data" / "trec_ct" / "raw" / f"topics_{args.year}.xml"
    qrels_path = ROOT / "data" / "trec_ct" / "raw" / f"qrels_{args.year}.txt"
    predictions_path = out_dir / "predictions.json"
    trec_metrics_path = out_dir / "trec_metrics.json"
    eligibility_metrics_path = out_dir / "eligibility_metrics.json"
    manifest_path = out_dir / "manifest.json"

    for path in (topics_path, qrels_path):
        if not path.exists():
            raise SystemExit(f"Required input not found: {path}")

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "args": _jsonable_args(args),
        "files": {
            "topics": str(topics_path),
            "qrels": str(qrels_path),
            "predictions": str(predictions_path),
            "trec_metrics": str(trec_metrics_path),
            "eligibility_metrics": str(eligibility_metrics_path),
        },
    }
    env = _build_env()
    # Make experiment manifests authoritative. The runner still accepts
    # negative CLI flags for direct use, but this script should not silently
    # inherit local .env feature toggles that contradict its own arguments.
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_LISTWISE", args.enable_listwise)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_DENSE_RETRIEVAL", not args.no_dense)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_HYDE", args.enable_hyde)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_VERIFIER", args.enable_verifier)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_FEW_SHOT", args.enable_few_shot)
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_SELF_CONSISTENCY",
        not (args.no_self_consistency or args.no_sc),
    )
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_LLM_JUDGE", not args.no_llm_judge)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_SELF_CRITIQUE", not args.no_self_critique)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_QUESTIONS", not args.no_questions)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_DOSSIERS", not args.no_dossiers)
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__USE_CRITERION_TRIAGE", args.enable_criterion_triage)
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_SECTION_HEADER_POLICY",
        args.enable_section_header_policy,
    )
    _set_bool_env(env, "TRIAL_MATCHER__RUNNER__BENCHMARK_SOFT_VETO", args.benchmark_soft_veto)
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_HARD_EXCLUDED_FILL",
        not args.no_hard_excluded_fill,
    )
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_RETRIEVAL_TAIL_FILL",
        args.enable_retrieval_tail_fill,
    )
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__INCLUDE_RETRIEVAL_TRACES",
        args.include_retrieval_traces,
    )
    env["TRIAL_MATCHER__RUNNER__BENCHMARK_CANDIDATE_SELECTION_POLICY"] = (
        args.benchmark_candidate_selection_policy
    )
    env["TRIAL_MATCHER__RUNNER__BENCHMARK_DIVERSE_KEEP_TOP"] = str(
        max(0, args.benchmark_diverse_keep_top)
    )
    env["TRIAL_MATCHER__RUNNER__BENCHMARK_DIVERSE_SELECT_TOTAL"] = str(
        max(1, args.benchmark_diverse_select_total)
    )
    env["TRIAL_MATCHER__RUNNER__BENCHMARK_ENTITY_RERANK_POLICY"] = (
        args.benchmark_entity_rerank_policy
    )
    if args.benchmark_entity_rerank_weight is not None:
        env["TRIAL_MATCHER__RUNNER__BENCHMARK_ENTITY_RERANK_WEIGHT"] = str(
            max(0.0, min(1.0, args.benchmark_entity_rerank_weight))
        )
    if args.benchmark_entity_protect_top is not None:
        env["TRIAL_MATCHER__RUNNER__BENCHMARK_ENTITY_PROTECT_TOP"] = str(
            max(0, args.benchmark_entity_protect_top)
        )
    env["TRIAL_MATCHER__RUNNER__BENCHMARK_CRITERION_EVIDENCE_POLICY"] = (
        args.benchmark_criterion_evidence_policy
    )
    if args.benchmark_criterion_evidence_weight is not None:
        env["TRIAL_MATCHER__RUNNER__BENCHMARK_CRITERION_EVIDENCE_WEIGHT"] = str(
            max(0.0, min(1.0, args.benchmark_criterion_evidence_weight))
        )
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_IRRELEVANCE_HEURISTIC",
        args.enable_irrel_heuristic and not args.no_irrel_heuristic,
    )
    _set_bool_env(
        env,
        "TRIAL_MATCHER__RUNNER__USE_MULTISIGNAL_IRRELEVANCE_HEURISTIC",
        args.enable_multisignal_irrel_heuristic
        and not args.no_multisignal_irrel_heuristic,
    )
    if args.irrelevant_max_retrieval_prior is not None:
        env["TRIAL_MATCHER__RUNNER__IRRELEVANT_MAX_RETRIEVAL_PRIOR"] = str(
            args.irrelevant_max_retrieval_prior
        )
    if args.irrelevant_min_signal_count is not None:
        env["TRIAL_MATCHER__RUNNER__IRRELEVANT_MIN_SIGNAL_COUNT"] = str(
            args.irrelevant_min_signal_count
        )
    if args.benchmark_min_inclusion_fraction is not None:
        env["TRIAL_MATCHER__RUNNER__BENCHMARK_MIN_INCLUSION_FRACTION"] = str(
            args.benchmark_min_inclusion_fraction
        )
    if args.benchmark_max_nei_fraction is not None:
        env["TRIAL_MATCHER__RUNNER__BENCHMARK_MAX_NEI_FRACTION"] = str(
            args.benchmark_max_nei_fraction
        )

    runner_cmd = [
        sys.executable,
        "-m",
        "trial_matcher.runner",
        str(topics_path),
        str(predictions_path),
        "--concurrency",
        str(args.concurrency),
        "--top-k",
        str(args.top_k),
        "--topic-limit",
        str(args.topic_limit),
        "--mode",
        args.mode,
        "--max-trials-per-topic",
        str(max(0, args.max_trials_per_topic)),
        "--max-criteria-per-trial",
        str(max(0, args.max_criteria_per_trial)),
        "--topic-timeout-seconds",
        str(max(0, args.topic_timeout_seconds)),
        "--benchmark-candidate-selection-policy",
        args.benchmark_candidate_selection_policy,
        "--benchmark-diverse-keep-top",
        str(max(0, args.benchmark_diverse_keep_top)),
        "--benchmark-diverse-select-total",
        str(max(1, args.benchmark_diverse_select_total)),
        "--benchmark-entity-rerank-policy",
        args.benchmark_entity_rerank_policy,
        "--benchmark-criterion-evidence-policy",
        args.benchmark_criterion_evidence_policy,
    ]
    if args.benchmark_entity_rerank_weight is not None:
        runner_cmd.extend(
            [
                "--benchmark-entity-rerank-weight",
                str(max(0.0, min(1.0, args.benchmark_entity_rerank_weight))),
            ]
        )
    if args.benchmark_entity_protect_top is not None:
        runner_cmd.extend(
            [
                "--benchmark-entity-protect-top",
                str(max(0, args.benchmark_entity_protect_top)),
            ]
        )
    if args.benchmark_criterion_evidence_weight is not None:
        runner_cmd.extend(
            [
                "--benchmark-criterion-evidence-weight",
                str(max(0.0, min(1.0, args.benchmark_criterion_evidence_weight))),
            ]
        )
    if args.topic_ids:
        runner_cmd.extend(["--topic-ids", args.topic_ids])
    if args.ctgov_dir:
        runner_cmd.extend(["--ctgov-dir", str(args.ctgov_dir)])
    if args.bm25_mode:
        runner_cmd.extend(["--bm25-mode", args.bm25_mode])
    if args.bm25_index_dir:
        runner_cmd.extend(["--bm25-index-dir", str(args.bm25_index_dir)])
    if args.fielded_bm25_index_dir:
        runner_cmd.extend(["--fielded-bm25-index-dir", str(args.fielded_bm25_index_dir)])
    if args.bm25_top_k > 0:
        runner_cmd.extend(["--bm25-top-k", str(args.bm25_top_k)])
    if args.dense_top_k > 0:
        runner_cmd.extend(["--dense-top-k", str(args.dense_top_k)])
    if args.no_dense:
        runner_cmd.append("--no-dense")
    if args.dense_rrf_weight >= 0:
        runner_cmd.extend(["--dense-rrf-weight", str(args.dense_rrf_weight)])
    if args.fused_top_k > 0:
        runner_cmd.extend(["--fused-top-k", str(args.fused_top_k)])
    if args.rerank_top_k > 0:
        runner_cmd.extend(["--rerank-top-k", str(args.rerank_top_k)])
    if args.listwise_top_k > 0:
        runner_cmd.extend(["--listwise-top-k", str(args.listwise_top_k)])
    if args.require_benchmark_index_manifest:
        runner_cmd.append("--require-benchmark-index-manifest")
    if args.no_benchmark_index_manifest_check:
        runner_cmd.append("--no-benchmark-index-manifest-check")
    if not args.enable_listwise:
        runner_cmd.append("--no-listwise")
    if not args.enable_hyde:
        runner_cmd.append("--no-hyde")
    if not args.enable_verifier:
        runner_cmd.append("--no-verifier")
    if not args.enable_few_shot:
        runner_cmd.append("--no-few-shot")
    if args.no_self_consistency or args.no_sc:
        runner_cmd.append("--no-self-consistency")
    if args.no_llm_judge:
        runner_cmd.append("--no-llm-judge")
    if args.no_self_critique:
        runner_cmd.append("--no-self-critique")
    if args.no_questions:
        runner_cmd.append("--no-questions")
    if args.no_dossiers:
        runner_cmd.append("--no-dossiers")
    if args.enable_criterion_triage:
        runner_cmd.append("--enable-criterion-triage")
    if args.enable_section_header_policy:
        runner_cmd.append("--enable-section-header-policy")
    if args.benchmark_soft_veto:
        runner_cmd.append("--benchmark-soft-veto")
    if args.enable_hard_excluded_fill:
        runner_cmd.append("--enable-hard-excluded-fill")
    if args.no_hard_excluded_fill:
        runner_cmd.append("--no-hard-excluded-fill")
    if args.enable_retrieval_tail_fill:
        runner_cmd.append("--enable-retrieval-tail-fill")
    if args.include_retrieval_traces:
        runner_cmd.append("--include-retrieval-traces")
    if args.enable_irrel_heuristic:
        runner_cmd.append("--enable-irrel-heuristic")
    if args.no_irrel_heuristic:
        runner_cmd.append("--no-irrel-heuristic")
    if args.enable_multisignal_irrel_heuristic:
        runner_cmd.append("--enable-multisignal-irrel-heuristic")
    if args.no_multisignal_irrel_heuristic:
        runner_cmd.append("--no-multisignal-irrel-heuristic")
    if args.irrelevant_max_retrieval_prior is not None:
        runner_cmd.extend(
            ["--irrelevant-max-retrieval-prior", str(args.irrelevant_max_retrieval_prior)]
        )
    if args.irrelevant_min_signal_count is not None:
        runner_cmd.extend(["--irrelevant-min-signal-count", str(args.irrelevant_min_signal_count)])
    if args.benchmark_min_inclusion_fraction is not None:
        runner_cmd.extend(
            ["--benchmark-min-inclusion-fraction", str(args.benchmark_min_inclusion_fraction)]
        )
    if args.benchmark_max_nei_fraction is not None:
        runner_cmd.extend(
            ["--benchmark-max-nei-fraction", str(args.benchmark_max_nei_fraction)]
        )

    try:
        _run_command("runner", runner_cmd, out_dir, manifest, env)
        _run_command(
            "trec_eval",
            [
                sys.executable,
                "eval/trec_eval.py",
                "--predictions",
                str(predictions_path),
                "--qrels",
                str(qrels_path),
                "--output",
                str(trec_metrics_path),
            ],
            out_dir,
            manifest,
            env,
        )
        _run_command(
            "eligibility_eval",
            [
                sys.executable,
                "eval/eligibility_eval.py",
                "--predictions",
                str(predictions_path),
                "--qrels",
                str(qrels_path),
                "--output",
                str(eligibility_metrics_path),
            ],
            out_dir,
            manifest,
            env,
        )
        manifest["summary"] = _prediction_summary(predictions_path)
    except Exception as exc:
        manifest["error"] = repr(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise SystemExit(str(exc)) from exc

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
