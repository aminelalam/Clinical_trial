"""Batch CLI runner. Reads patients JSON/XML and writes predictions JSON."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import click

from .config import get_settings
from .ingestion.benchmark_manifest import (
    validate_benchmark_corpus_alignment,
    validate_benchmark_index_manifest,
)
from .ingestion.trec_parser import parse_topics
from .logging import logger, set_run_id, setup_logging
from .models.agent_state import AgentState, initial_state
from .retrieval.filters import active_status_set
from .runtime import AgentRuntime, build_agent_runtime


def _load_patients(input_path: Path) -> list[dict[str, str]]:
    """Auto-detect input format and return a list of {topic_id, text}."""
    text = input_path.read_text(encoding="utf-8")
    suffix = input_path.suffix.lower()
    if suffix == ".xml":
        topics = parse_topics(input_path)
        return [{"topic_id": t.topic_id, "text": t.text} for t in topics]
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [{"topic_id": str(k), "text": str(v)} for k, v in data.items()]
        if isinstance(data, list):
            out = []
            for item in data:
                if "topic_id" in item:
                    out.append(
                        {
                            "topic_id": str(item["topic_id"]),
                            "text": item.get("text") or item.get("raw_text", ""),
                        }
                    )
                elif "patient_id" in item:
                    out.append(
                        {
                            "topic_id": str(item["patient_id"]),
                            "text": item.get("raw_text") or item.get("text", ""),
                        }
                    )
            return out
    raise ValueError(f"Unsupported input format: {input_path}")


def _parse_topic_ids(topic_ids: str | None) -> list[str] | None:
    """Parse a comma-separated topic id filter."""
    if topic_ids is None or not topic_ids.strip():
        return None
    parsed = [part.strip() for part in topic_ids.split(",") if part.strip()]
    return parsed or None


def _select_patients(
    patients: Sequence[dict[str, str]],
    topic_ids: Sequence[str] | None = None,
    topic_limit: int = 0,
) -> list[dict[str, str]]:
    """Select a deterministic patient subset for smoke and mini-eval runs."""
    if topic_limit < 0:
        raise ValueError("--topic-limit must be >= 0")

    selected = list(patients)
    if topic_ids:
        by_id = {p["topic_id"]: p for p in patients}
        missing = [tid for tid in topic_ids if tid not in by_id]
        if missing:
            raise ValueError(f"Requested topic ids not found: {', '.join(missing)}")
        selected = [by_id[tid] for tid in topic_ids]

    if topic_limit > 0:
        selected = selected[:topic_limit]
    return selected


def _benchmark_index_validation(index_dir: Path) -> dict[str, Any]:
    """Return benchmark index-manifest validation diagnostics."""
    return validate_benchmark_index_manifest(index_dir)


def _benchmark_corpus_validation(index_dir: Path, corpus_dir: Path) -> dict[str, Any]:
    """Return diagnostics proving the agent corpus matches the benchmark index."""
    return validate_benchmark_corpus_alignment(index_dir, corpus_dir)


def _enforce_benchmark_index_policy(index_dir: Path) -> dict[str, Any]:
    s = get_settings()
    if s.runner.mode != "benchmark" or s.runner.benchmark_index_manifest_policy == "off":
        return {"valid": None, "reason": "disabled", "path": str(index_dir)}
    validation = _benchmark_index_validation(index_dir)
    if validation.get("valid"):
        logger.info(
            "Benchmark index manifest valid: "
            f"index={index_dir} doc_count={validation.get('doc_count')}"
        )
        return validation
    message = f"Benchmark index manifest check failed for {index_dir}: {validation}"
    if s.runner.benchmark_index_manifest_policy == "require":
        raise click.ClickException(message)
    logger.warning(message)
    return validation


def _enforce_benchmark_corpus_policy(index_dir: Path, corpus_dir: Path) -> dict[str, Any]:
    s = get_settings()
    if s.runner.mode != "benchmark" or s.runner.benchmark_index_manifest_policy == "off":
        return {"valid": None, "reason": "disabled", "path": str(corpus_dir)}
    validation = _benchmark_corpus_validation(index_dir, corpus_dir)
    if validation.get("valid"):
        logger.info(
            "Benchmark corpus manifest valid: "
            f"corpus={corpus_dir} doc_count={validation.get('doc_count')}"
        )
        return validation
    message = (
        f"Benchmark corpus/index alignment check failed for index={index_dir} "
        f"corpus={corpus_dir}: {validation}"
    )
    if s.runner.benchmark_index_manifest_policy == "require":
        raise click.ClickException(message)
    logger.warning(message)
    return validation


def _filter_reason_counts(final_state: AgentState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in final_state.final_candidates:
        key = c.excluded_reason or "viable"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_id_trace(candidates: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        nct_id = str(getattr(cand, "nct_id", "") or "")
        if nct_id and nct_id not in seen:
            out.append(nct_id)
            seen.add(nct_id)
    return out


def _retrieval_traces(final_state: AgentState) -> dict[str, list[str]]:
    """Return candidate IDs by stage for qrel/golden-trial auditing."""
    traces = {
        "bm25_candidates": _candidate_id_trace(final_state.bm25_candidates),
        "dense_candidates": _candidate_id_trace(final_state.dense_candidates),
        "fused_candidates": _candidate_id_trace(final_state.fused_candidates),
        "reranked_candidates": _candidate_id_trace(final_state.reranked_candidates),
        "listwise_candidates": _candidate_id_trace(final_state.listwise_candidates),
        "final_candidates": _candidate_id_trace(final_state.final_candidates),
        "ranked_trials": _candidate_id_trace(final_state.ranked_trials),
        "judged_top10": _candidate_id_trace(final_state.judged_top10),
    }
    traces.update(
        {
            # Backward-compatible short names used by early P9 audit outputs.
            "bm25": traces["bm25_candidates"],
            "dense": traces["dense_candidates"],
            "fused": traces["fused_candidates"],
            "reranked": traces["reranked_candidates"],
            "listwise": traces["listwise_candidates"],
            "final": traces["final_candidates"],
            "ranked": traces["ranked_trials"],
            "judged": traces["judged_top10"],
        }
    )
    return traces


def _no_ranked_reason(final_state: AgentState) -> str | None:
    if final_state.judged_top10:
        return None
    if final_state.errors:
        return "state_errors"
    if not final_state.bm25_candidates and not final_state.dense_candidates:
        return "retrieval_empty"
    if not final_state.final_candidates:
        return "no_final_candidates"
    if not any(not c.hard_excluded for c in final_state.final_candidates):
        return "all_candidates_hard_excluded"
    if not final_state.trial_evals:
        return "no_trial_evals"
    if not final_state.ranked_trials:
        return "no_ranked_trials"
    return "judge_or_dossier_empty"


def _planned_statuses(final_state: AgentState) -> list[str] | None:
    if final_state.search_plan is None or not final_state.search_plan.mandatory_filters:
        return None
    statuses = final_state.search_plan.mandatory_filters.get("status")
    if isinstance(statuses, list):
        return [str(s) for s in statuses]
    return None


def _status_diagnostics(final_state: AgentState, runtime: AgentRuntime) -> dict[str, Any]:
    planned = _planned_statuses(final_state)
    clinical_statuses = active_status_set(planned)
    status_counts: dict[str, int] = {}
    clinical_status_hard_exclusion_count = 0
    trial_not_found_count = 0

    for c in final_state.final_candidates:
        trial = runtime.trials_by_id.get(c.nct_id)
        if trial is None:
            status = "trial not found in corpus"
            trial_not_found_count += 1
        else:
            status = trial.status.value
            if status not in clinical_statuses:
                clinical_status_hard_exclusion_count += 1
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "status_counts": status_counts,
        "clinical_active_allowed_statuses": sorted(clinical_statuses),
        "clinical_status_hard_exclusion_count": clinical_status_hard_exclusion_count,
        "trial_not_found_count": trial_not_found_count,
    }


def _eligibility_diagnostics(final_state: AgentState) -> dict[str, Any]:
    criteria_extracted = sum(len(v) for v in final_state.extracted_criteria.values())
    evals = [ev for values in final_state.criterion_evals.values() for ev in values]
    label_counts: dict[str, int] = {}
    evaluator_counts: dict[str, int] = {}
    for ev in evals:
        label = getattr(ev.label, "value", str(ev.label))
        evaluator = str(ev.evaluator)
        label_counts[label] = label_counts.get(label, 0) + 1
        evaluator_counts[evaluator] = evaluator_counts.get(evaluator, 0) + 1
    triage = _criterion_triage_diagnostics(final_state)
    return {
        "criteria_extracted": criteria_extracted,
        "criteria_evaluated": len(evals),
        "criterion_label_counts": label_counts,
        "criterion_evaluator_counts": evaluator_counts,
        "criterion_llm_calls": sum(int(ev.llm_calls or 0) for ev in evals),
        "self_consistency_calls": sum(
            int(ev.llm_calls or 0) for ev in evals if ev.evaluator == "self_consistency"
        ),
        "verifier_calls": sum(
            1 for ev in evals if ev.rebuttal is not None or ev.flipped_by_verifier
        ),
        "criterion_triage": triage,
    }


def _merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value or 0)


def _criterion_triage_diagnostics(final_state: AgentState) -> dict[str, Any]:
    diagnostics = final_state.criterion_selection_diagnostics or {}
    selected_by_type: dict[str, int] = {}
    dropped_by_type: dict[str, int] = {}
    selected_by_polarity: dict[str, int] = {}
    dropped_by_polarity: dict[str, int] = {}
    top_dropped: list[dict[str, Any]] = []
    total_seen = 0
    selected = 0
    dropped = 0
    score_sum = 0.0
    score_n = 0
    section_headers_merged = 0
    section_headers_dropped = 0
    section_headers_excluded_before_eval = 0

    for trial_id, diag in diagnostics.items():
        total_seen += int(diag.get("total_seen", 0) or 0)
        selected += int(diag.get("selected", 0) or 0)
        dropped += int(diag.get("dropped", 0) or 0)
        section_headers_merged += int(diag.get("section_headers_merged", 0) or 0)
        section_headers_dropped += int(diag.get("section_headers_dropped", 0) or 0)
        section_headers_excluded_before_eval += int(
            diag.get("section_headers_excluded_before_eval", 0) or 0
        )
        _merge_counts(selected_by_type, diag.get("selected_by_type", {}) or {})
        _merge_counts(dropped_by_type, diag.get("dropped_by_type", {}) or {})
        _merge_counts(selected_by_polarity, diag.get("selected_by_polarity", {}) or {})
        _merge_counts(dropped_by_polarity, diag.get("dropped_by_polarity", {}) or {})
        if diag.get("mean_selected_score") is not None and int(diag.get("selected", 0) or 0) > 0:
            n = int(diag.get("selected", 0) or 0)
            score_sum += float(diag.get("mean_selected_score", 0.0) or 0.0) * n
            score_n += n
        for item in diag.get("top_dropped", []) or []:
            if isinstance(item, dict):
                top_dropped.append({"trial_id": trial_id, **item})

    top_dropped = sorted(
        top_dropped,
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )[:10]
    return {
        "trials_with_triage": len(diagnostics),
        "total_seen": total_seen,
        "selected": selected,
        "dropped": dropped,
        "selected_by_type": selected_by_type,
        "dropped_by_type": dropped_by_type,
        "selected_by_polarity": selected_by_polarity,
        "dropped_by_polarity": dropped_by_polarity,
        "section_headers_merged": section_headers_merged,
        "section_headers_dropped": section_headers_dropped,
        "section_headers_excluded_before_eval": section_headers_excluded_before_eval,
        "mean_selected_score": round(score_sum / score_n, 3) if score_n else 0.0,
        "top_dropped": top_dropped,
    }


def _final_to_ranked_gap(final_state: AgentState) -> dict[str, Any]:
    final_top20 = _candidate_id_trace(final_state.final_candidates)[:20]
    final_viable_top20 = _candidate_id_trace(
        [c for c in final_state.final_candidates if not c.hard_excluded]
    )[:20]
    ranked_top20 = _candidate_id_trace(final_state.ranked_trials)[:20]
    ranked_set = set(ranked_top20)
    final_missing = [nct_id for nct_id in final_top20 if nct_id not in ranked_set]
    viable_missing = [nct_id for nct_id in final_viable_top20 if nct_id not in ranked_set]
    return {
        "final_candidates_at_20": len(final_top20),
        "final_viable_at_20": len(final_viable_top20),
        "ranked_trials_at_20": len(ranked_top20),
        "final_candidates_missing_from_ranked_at_20_count": len(final_missing),
        "final_candidates_missing_from_ranked_at_20_ids": final_missing,
        "final_viable_missing_from_ranked_at_20_count": len(viable_missing),
        "final_viable_missing_from_ranked_at_20_ids": viable_missing,
    }


async def _process_one(
    runtime: AgentRuntime,
    patient: dict[str, str],
    semaphore: asyncio.Semaphore,
    top_k: int,
) -> dict[str, Any]:
    async with semaphore:
        topic_id = patient["topic_id"]
        set_run_id(topic_id)
        logger.info(f"Processing topic {topic_id}")
        s = get_settings()
        initial = initial_state(
            patient_raw=patient["text"],
            run_id=topic_id,
            max_retrieval_attempts=s.runner.max_retrieval_attempts,
            max_critique_iterations=s.runner.max_critique_iterations,
        )
        try:
            if s.runner.topic_timeout_seconds > 0:
                final_dict = await asyncio.wait_for(
                    runtime.agent.ainvoke(initial),
                    timeout=s.runner.topic_timeout_seconds,
                )
            else:
                final_dict = await runtime.agent.ainvoke(initial)
            final_state = (
                AgentState.model_validate(final_dict)
                if isinstance(final_dict, dict)
                else final_dict
            )
        except Exception as e:
            logger.exception(f"Agent failed on topic {topic_id}: {e}")
            return {
                "topic_id": topic_id,
                "ranked_trials": [],
                "questions": [],
                "dossiers": [],
                "error": str(e),
                "diagnostics": {
                    "runner_mode": s.runner.mode,
                    "topic_timeout_seconds": s.runner.topic_timeout_seconds,
                    "failure_type": type(e).__name__,
                },
            }

        status_diagnostics = _status_diagnostics(final_state, runtime)
        eligibility_diagnostics = _eligibility_diagnostics(final_state)
        return {
            "topic_id": topic_id,
            "ranked_trials": [
                {
                    "nct_id": j.nct_id,
                    "rank": j.rank,
                    "score": round(float(j.score), 4),
                    "label": j.eval.label.value,
                    "predicted_trec_qrel": j.eval.trec_qrel,
                    # Backward-compatible alias used by older eval scripts.
                    # It is a system prediction, not the official qrel.
                    "trec_qrel": j.eval.trec_qrel,
                    "rationale": j.rationale,
                    "n_inclusion_met": j.eval.n_inclusion_met,
                    "n_inclusion_total": j.eval.n_inclusion,
                    "n_exclusion_met": j.eval.n_exclusion_met,
                    "n_exclusion_total": j.eval.n_exclusion,
                    "fraction_nei": round(j.eval.fraction_nei, 3),
                    "components": j.components,
                    "hard_excluded_fill": bool(j.hard_excluded_fill),
                    "retrieval_tail_fill": bool(j.retrieval_tail_fill),
                    "excluded_reason": j.excluded_reason,
                }
                for j in final_state.judged_top10[:top_k]
            ],
            "questions": [q.model_dump(mode="json") for q in final_state.questions],
            "dossiers": [d.model_dump(mode="json") for d in final_state.dossiers[:top_k]],
            "stats": {
                "llm_calls": final_state.llm_calls,
                "cache_hits": final_state.cache_hits,
                "node_timings": [t.model_dump() for t in final_state.node_timings],
                "errors": final_state.errors,
            },
            "diagnostics": {
                "runner_mode": s.runner.mode,
                "no_ranked_reason": _no_ranked_reason(final_state),
                "retrieval_attempts": final_state.retrieval_attempts,
                "re_retrieval_triggered": final_state.re_retrieval_triggered,
                "relaxed_plan_used": final_state.relaxed_plan_used,
                "critique_iterations": final_state.critique_iterations,
                "candidate_counts": {
                    "bm25": len(final_state.bm25_candidates),
                    "dense": len(final_state.dense_candidates),
                    "fused": len(final_state.fused_candidates),
                    "reranked": len(final_state.reranked_candidates),
                    "listwise": len(final_state.listwise_candidates),
                    "final": len(final_state.final_candidates),
                    "viable": sum(1 for c in final_state.final_candidates if not c.hard_excluded),
                    "trial_evals": len(final_state.trial_evals),
                    "ranked": len(final_state.ranked_trials),
                    "judged": len(final_state.judged_top10),
                },
                "filter_reasons": _filter_reason_counts(final_state),
                "candidate_selection": final_state.candidate_selection_diagnostics,
                "entity_negation": final_state.entity_negation_diagnostics,
                "criterion_evidence": final_state.criterion_evidence_diagnostics,
                "irrelevance_policy": final_state.irrelevance_diagnostics,
                "final_to_ranked_gap": _final_to_ranked_gap(final_state),
                "hard_excluded_fill_count": final_state.hard_excluded_fill_count,
                "retrieval_tail_fill_count": final_state.retrieval_tail_fill_count,
                "hard_excluded_fill_reasons": final_state.hard_excluded_fill_reasons,
                "hard_excluded_fill_skipped_corpus_miss": (
                    final_state.hard_excluded_fill_skipped_corpus_miss
                ),
                **(
                    {"retrieval_traces": _retrieval_traces(final_state)}
                    if s.runner.include_retrieval_traces
                    else {}
                ),
                "status_filter_applied": s.runner.mode == "clinical_active",
                **status_diagnostics,
                **eligibility_diagnostics,
                "corpus": {
                    "loaded": runtime.corpus_loaded,
                    "size": runtime.corpus_size,
                },
            },
        }


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--top-k", default=20, help="Trials per topic in the predictions output")
@click.option("--ctgov-dir", default=None, type=click.Path(path_type=Path), help="Override trial corpus directory loaded by the agent")
@click.option("--bm25-index-dir", default=None, type=click.Path(path_type=Path), help="Override BM25 index directory")
@click.option("--bm25-mode", default=None, type=click.Choice(["single", "fielded"]), help="BM25 mode: single P8 index or fielded P9 indexes")
@click.option("--fielded-bm25-index-dir", default=None, type=click.Path(path_type=Path), help="Override fielded BM25 container directory")
@click.option("--bm25-top-k", default=None, type=int, help="BM25 first-stage candidate count")
@click.option("--dense-top-k", default=None, type=int, help="Dense first-stage candidate count")
@click.option("--dense-rrf-weight", default=None, type=float, help="Dense run weight in RRF fusion")
@click.option("--fused-top-k", default=None, type=int, help="Fused candidate pool passed to pointwise rerank")
@click.option("--rerank-top-k", default=None, type=int, help="Pointwise reranker output count")
@click.option("--listwise-top-k", default=None, type=int, help="Candidate count kept after optional listwise rerank")
@click.option("--no-dense", is_flag=True, default=False, help="Disable dense retrieval for BM25-only experiments")
@click.option(
    "--require-benchmark-index-manifest",
    is_flag=True,
    default=False,
    help="Fail benchmark runs unless the BM25 index has a valid TREC benchmark manifest",
)
@click.option(
    "--no-benchmark-index-manifest-check",
    is_flag=True,
    default=False,
    help="Disable benchmark BM25 index manifest validation",
)
@click.option("--concurrency", default=None, type=int, help="Override concurrency from env")
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["benchmark", "clinical_active"]),
    help="Runner mode: benchmark skips hard status filtering; clinical_active applies it",
)
@click.option("--topic-limit", default=0, type=int, help="Run only the first N selected topics")
@click.option("--topic-ids", default="", help="Comma-separated topic ids to run before applying --topic-limit")
@click.option("--no-listwise", is_flag=True, default=False, help="Disable RankZephyr listwise reranker")
@click.option("--no-hyde", is_flag=True, default=False, help="Disable HyDE query expansion")
@click.option("--no-verifier", is_flag=True, default=False, help="Disable devil's-advocate verifier")
@click.option("--no-self-consistency", is_flag=True, default=False, help="Disable k-sample self-consistency in eligibility")
@click.option("--no-sc", is_flag=True, default=False, help="Alias for --no-self-consistency")
@click.option("--no-llm-judge", is_flag=True, default=False, help="Disable LLM top-10 judge reranker")
@click.option("--no-self-critique", is_flag=True, default=False, help="Disable final self-critique pass")
@click.option("--no-questions", is_flag=True, default=False, help="Disable T4 missing-information questions")
@click.option("--no-dossiers", is_flag=True, default=False, help="Disable T5 dossier generation")
@click.option(
    "--enable-criterion-triage",
    is_flag=True,
    default=False,
    help="Use experimental clinical criterion triage before eligibility evaluation",
)
@click.option(
    "--enable-section-header-policy",
    is_flag=True,
    default=False,
    help="Use experimental CT.gov section-header merge/drop policy before criterion classification",
)
@click.option(
    "--benchmark-soft-veto",
    is_flag=True,
    default=False,
    help="In benchmark mode, score safety vetoes as a penalty instead of -1.0",
)
@click.option(
    "--enable-hard-excluded-fill",
    is_flag=True,
    default=False,
    help="In benchmark mode, fill short rankings with hard-excluded retrieved candidates",
)
@click.option(
    "--no-hard-excluded-fill",
    is_flag=True,
    default=False,
    help="Disable benchmark hard-excluded fill even if enabled in settings",
)
@click.option(
    "--enable-retrieval-tail-fill",
    is_flag=True,
    default=False,
    help="In benchmark mode, fill long outputs with unevaluated retrieved viable candidates",
)
@click.option(
    "--include-retrieval-traces",
    is_flag=True,
    default=False,
    help="Write candidate NCT IDs by retrieval stage for qrel/golden-trial auditing",
)
@click.option(
    "--benchmark-candidate-selection-policy",
    default=None,
    type=click.Choice(["top_score", "diverse_top10"]),
    help="Benchmark-only eligibility candidate selector",
)
@click.option(
    "--benchmark-diverse-keep-top",
    default=None,
    type=int,
    help="For diverse_top10, keep this many highest-score candidates before MMR fill",
)
@click.option(
    "--benchmark-diverse-select-total",
    default=None,
    type=int,
    help="For diverse_top10, total viable candidates evaluated per topic",
)
@click.option(
    "--benchmark-entity-rerank-policy",
    default=None,
    type=click.Choice(["off", "audit", "rerank_final"]),
    help="Benchmark-only entity/negation audit or rerank over final candidates",
)
@click.option(
    "--benchmark-entity-rerank-weight",
    default=None,
    type=float,
    help="Blend weight for benchmark entity/negation rerank",
)
@click.option(
    "--benchmark-entity-protect-top",
    default=None,
    type=int,
    help="For entity/negation rerank, preserve the first N candidates from demotion",
)
@click.option(
    "--benchmark-criterion-evidence-policy",
    default=None,
    type=click.Choice(["off", "score_adjust"]),
    help="Benchmark-only criterion-level evidence score adjustment",
)
@click.option(
    "--benchmark-criterion-evidence-weight",
    default=None,
    type=float,
    help="Blend weight for benchmark criterion-level evidence scoring",
)
@click.option(
    "--no-irrel-heuristic",
    is_flag=True,
    default=False,
    help="Disable the strict high-NEI trial-level IRRELEVANT heuristic",
)
@click.option(
    "--enable-irrel-heuristic",
    is_flag=True,
    default=False,
    help="Enable the experimental high-NEI trial-level IRRELEVANT heuristic",
)
@click.option(
    "--enable-multisignal-irrel-heuristic",
    is_flag=True,
    default=False,
    help="Enable benchmark-only multi-signal class-0 policy",
)
@click.option(
    "--no-multisignal-irrel-heuristic",
    is_flag=True,
    default=False,
    help="Disable benchmark-only multi-signal class-0 policy",
)
@click.option(
    "--irrelevant-max-retrieval-prior",
    default=None,
    type=float,
    help="Max retrieval prior for multi-signal class-0 low-prior signal",
)
@click.option(
    "--irrelevant-min-signal-count",
    default=None,
    type=int,
    help="Minimum number of multi-signal class-0 signals required",
)
@click.option(
    "--benchmark-min-inclusion-fraction",
    default=None,
    type=float,
    help="Benchmark-only partial-eligibility inclusion fraction threshold",
)
@click.option(
    "--benchmark-max-nei-fraction",
    default=None,
    type=float,
    help="Benchmark-only partial-eligibility NEI fraction threshold",
)
@click.option("--max-trials-per-topic", default=None, type=int, help="Cap evaluated viable trials per topic")
@click.option("--max-criteria-per-trial", default=None, type=int, help="Cap evaluated criteria per trial")
@click.option("--topic-timeout-seconds", default=None, type=int, help="Fail a single topic after N seconds (0 disables)")
@click.option(
    "--no-few-shot",
    is_flag=True,
    default=False,
    help="Disable dynamic few-shot bank in eligibility evaluation",
)
@click.option(
    "--few-shot-dir",
    type=click.Path(path_type=Path),
    default=Path("banco_few_shot"),
    help="Directory of few-shot example JSONL files (default: banco_few_shot)",
)
def main(
    input_path: Path,
    output_path: Path,
    top_k: int,
    ctgov_dir: Path | None,
    bm25_index_dir: Path | None,
    bm25_mode: str | None,
    fielded_bm25_index_dir: Path | None,
    bm25_top_k: int | None,
    dense_top_k: int | None,
    dense_rrf_weight: float | None,
    fused_top_k: int | None,
    rerank_top_k: int | None,
    listwise_top_k: int | None,
    no_dense: bool,
    require_benchmark_index_manifest: bool,
    no_benchmark_index_manifest_check: bool,
    concurrency: int | None,
    mode: str | None,
    topic_limit: int,
    topic_ids: str,
    no_listwise: bool,
    no_hyde: bool,
    no_verifier: bool,
    no_self_consistency: bool,
    no_sc: bool,
    no_llm_judge: bool,
    no_self_critique: bool,
    no_questions: bool,
    no_dossiers: bool,
    enable_criterion_triage: bool,
    enable_section_header_policy: bool,
    benchmark_soft_veto: bool,
    enable_hard_excluded_fill: bool,
    no_hard_excluded_fill: bool,
    enable_retrieval_tail_fill: bool,
    include_retrieval_traces: bool,
    benchmark_candidate_selection_policy: str | None,
    benchmark_diverse_keep_top: int | None,
    benchmark_diverse_select_total: int | None,
    benchmark_entity_rerank_policy: str | None,
    benchmark_entity_rerank_weight: float | None,
    benchmark_entity_protect_top: int | None,
    benchmark_criterion_evidence_policy: str | None,
    benchmark_criterion_evidence_weight: float | None,
    no_irrel_heuristic: bool,
    enable_irrel_heuristic: bool,
    enable_multisignal_irrel_heuristic: bool,
    no_multisignal_irrel_heuristic: bool,
    irrelevant_max_retrieval_prior: float | None,
    irrelevant_min_signal_count: int | None,
    benchmark_min_inclusion_fraction: float | None,
    benchmark_max_nei_fraction: float | None,
    max_trials_per_topic: int | None,
    max_criteria_per_trial: int | None,
    topic_timeout_seconds: int | None,
    no_few_shot: bool,
    few_shot_dir: Path,
) -> None:
    """CLI entrypoint."""
    setup_logging()
    s = get_settings()
    if top_k < 1:
        raise click.BadParameter("--top-k must be >= 1")
    s.runner.output_top_k = top_k
    if ctgov_dir is not None:
        s.paths.ctgov_dir = ctgov_dir
    if bm25_index_dir is not None:
        s.retrieval.bm25_index_dir = bm25_index_dir
    if bm25_mode is not None:
        s.retrieval.bm25_mode = bm25_mode
    if fielded_bm25_index_dir is not None:
        s.retrieval.fielded_bm25_index_dir = fielded_bm25_index_dir
    if bm25_top_k is not None:
        s.retrieval.bm25_top_k = max(1, bm25_top_k)
    if dense_top_k is not None:
        if dense_top_k <= 0:
            s.runner.use_dense_retrieval = False
        else:
            s.retrieval.dense_top_k = dense_top_k
    if no_dense:
        s.runner.use_dense_retrieval = False
    if dense_rrf_weight is not None:
        s.retrieval.dense_rrf_weight = max(0.0, dense_rrf_weight)
    if fused_top_k is not None:
        s.retrieval.fused_top_k = max(1, fused_top_k)
    if rerank_top_k is not None:
        s.retrieval.rerank_top_k = max(1, rerank_top_k)
    if listwise_top_k is not None:
        s.retrieval.listwise_top_k = max(1, listwise_top_k)
    if require_benchmark_index_manifest:
        s.runner.benchmark_index_manifest_policy = "require"
    if no_benchmark_index_manifest_check:
        s.runner.benchmark_index_manifest_policy = "off"
    if topic_limit < 0:
        raise click.BadParameter("--topic-limit must be >= 0")
    if mode is not None:
        s.runner.mode = mode
    if no_listwise:
        s.runner.use_listwise = False
    if no_hyde:
        s.runner.use_hyde = False
    if no_verifier:
        s.runner.use_verifier = False
    if no_self_consistency or no_sc:
        s.runner.use_self_consistency = False
    if no_llm_judge:
        s.runner.use_llm_judge = False
    if no_self_critique:
        s.runner.use_self_critique = False
    if no_questions:
        s.runner.use_questions = False
    if no_dossiers:
        s.runner.use_dossiers = False
    if enable_criterion_triage:
        s.runner.use_criterion_triage = True
    if enable_section_header_policy:
        s.runner.use_section_header_policy = True
    if benchmark_soft_veto:
        s.runner.benchmark_soft_veto = True
    if enable_hard_excluded_fill:
        s.runner.use_hard_excluded_fill = True
    if no_hard_excluded_fill:
        s.runner.use_hard_excluded_fill = False
    if enable_retrieval_tail_fill:
        s.runner.use_retrieval_tail_fill = True
    if include_retrieval_traces:
        s.runner.include_retrieval_traces = True
    if benchmark_candidate_selection_policy is not None:
        s.runner.benchmark_candidate_selection_policy = benchmark_candidate_selection_policy
    if benchmark_diverse_keep_top is not None:
        s.runner.benchmark_diverse_keep_top = max(0, benchmark_diverse_keep_top)
    if benchmark_diverse_select_total is not None:
        s.runner.benchmark_diverse_select_total = max(1, benchmark_diverse_select_total)
    if benchmark_entity_rerank_policy is not None:
        s.runner.benchmark_entity_rerank_policy = benchmark_entity_rerank_policy
    if benchmark_entity_rerank_weight is not None:
        s.runner.benchmark_entity_rerank_weight = max(
            0.0,
            min(1.0, benchmark_entity_rerank_weight),
        )
    if benchmark_entity_protect_top is not None:
        s.runner.benchmark_entity_protect_top = max(0, benchmark_entity_protect_top)
    if benchmark_criterion_evidence_policy is not None:
        s.runner.benchmark_criterion_evidence_policy = benchmark_criterion_evidence_policy
    if benchmark_criterion_evidence_weight is not None:
        s.runner.benchmark_criterion_evidence_weight = max(
            0.0,
            min(1.0, benchmark_criterion_evidence_weight),
        )
    if enable_irrel_heuristic:
        s.runner.use_irrelevance_heuristic = True
    if no_irrel_heuristic:
        s.runner.use_irrelevance_heuristic = False
    if enable_multisignal_irrel_heuristic:
        s.runner.use_multisignal_irrelevance_heuristic = True
    if no_multisignal_irrel_heuristic:
        s.runner.use_multisignal_irrelevance_heuristic = False
    if irrelevant_max_retrieval_prior is not None:
        s.runner.irrelevant_max_retrieval_prior = max(
            0.0,
            min(1.0, irrelevant_max_retrieval_prior),
        )
    if irrelevant_min_signal_count is not None:
        s.runner.irrelevant_min_signal_count = max(1, irrelevant_min_signal_count)
    if benchmark_min_inclusion_fraction is not None:
        s.runner.benchmark_min_inclusion_fraction = max(
            0.0,
            min(1.0, benchmark_min_inclusion_fraction),
        )
    if benchmark_max_nei_fraction is not None:
        s.runner.benchmark_max_nei_fraction = max(
            0.0,
            min(1.0, benchmark_max_nei_fraction),
        )
    if no_few_shot:
        s.runner.use_few_shot = False
    if max_trials_per_topic is not None:
        s.runner.max_trials_per_topic = max(0, max_trials_per_topic)
    if max_criteria_per_trial is not None:
        s.runner.max_criteria_per_trial = max(0, max_criteria_per_trial)
    if topic_timeout_seconds is not None:
        s.runner.topic_timeout_seconds = max(0, topic_timeout_seconds)
    conc = concurrency if concurrency is not None else s.runner.concurrency

    s.paths.ensure()
    if s.retrieval.bm25_mode == "fielded":
        active_bm25_index = Path(
            s.retrieval.fielded_bm25_index_dir
            or (s.paths.indices_dir / "bm25_trec2021_fielded")
        )
    else:
        active_bm25_index = Path(s.retrieval.bm25_index_dir or (s.paths.indices_dir / "bm25"))
    benchmark_index_manifest = _enforce_benchmark_index_policy(active_bm25_index)
    benchmark_corpus_manifest = _enforce_benchmark_corpus_policy(
        active_bm25_index,
        Path(s.paths.ctgov_dir),
    )

    loaded_patients = _load_patients(input_path)
    try:
        patients = _select_patients(
            loaded_patients,
            topic_ids=_parse_topic_ids(topic_ids),
            topic_limit=topic_limit,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if not patients:
        raise click.ClickException("No topics selected")
    logger.info(f"Loaded {len(loaded_patients)} patients from {input_path}; selected {len(patients)}")

    runtime = build_agent_runtime(
        use_few_shot=s.runner.use_few_shot,
        few_shot_dir=few_shot_dir,
    )
    sem = asyncio.Semaphore(conc)

    async def _run_all() -> list[dict[str, Any]]:
        return await asyncio.gather(*[_process_one(runtime, p, sem, top_k) for p in patients])

    results = asyncio.run(_run_all())

    output = {
        "metadata": {
            "input": str(input_path),
            "n_topics_loaded": len(loaded_patients),
            "n_topics": len(patients),
            "topic_ids": [p["topic_id"] for p in patients],
            "topic_limit": topic_limit,
            "topic_ids_filter": _parse_topic_ids(topic_ids) or [],
            "top_k": top_k,
            "cache_policy": {
                "mode": "controlled",
                "cache_dir": str(s.paths.cache_dir),
                "valid_outputs_reused": True,
                "empty_llm_outputs_cached": False,
            },
            "prediction_schema": {
                "official_grade_field": "predicted_trec_qrel",
                "deprecated_prediction_aliases": ["trec_qrel"],
            },
            "settings": {
                "runner_mode": s.runner.mode,
                "use_dense_retrieval": s.runner.use_dense_retrieval,
                "use_hyde": s.runner.use_hyde,
                "use_listwise": s.runner.use_listwise,
                "use_verifier": s.runner.use_verifier,
                "use_few_shot": s.runner.use_few_shot,
                "use_self_consistency": s.runner.use_self_consistency,
                "use_llm_judge": s.runner.use_llm_judge,
                "use_self_critique": s.runner.use_self_critique,
                "use_questions": s.runner.use_questions,
                "use_dossiers": s.runner.use_dossiers,
                "use_criterion_triage": s.runner.use_criterion_triage,
                "use_section_header_policy": s.runner.use_section_header_policy,
                "max_retrieval_attempts": s.runner.max_retrieval_attempts,
                "max_critique_iterations": s.runner.max_critique_iterations,
                "topic_timeout_seconds": s.runner.topic_timeout_seconds,
                "output_top_k": s.runner.output_top_k,
                "max_trials_per_topic": s.runner.max_trials_per_topic,
                "max_criteria_per_trial": s.runner.max_criteria_per_trial,
                "benchmark_candidate_selection_policy": (
                    s.runner.benchmark_candidate_selection_policy
                ),
                "benchmark_diverse_keep_top": s.runner.benchmark_diverse_keep_top,
                "benchmark_diverse_select_total": s.runner.benchmark_diverse_select_total,
                "benchmark_entity_rerank_policy": s.runner.benchmark_entity_rerank_policy,
                "benchmark_entity_rerank_weight": s.runner.benchmark_entity_rerank_weight,
                "benchmark_entity_protect_top": s.runner.benchmark_entity_protect_top,
                "benchmark_criterion_evidence_policy": (
                    s.runner.benchmark_criterion_evidence_policy
                ),
                "benchmark_criterion_evidence_weight": (
                    s.runner.benchmark_criterion_evidence_weight
                ),
                "benchmark_soft_veto": s.runner.benchmark_soft_veto,
                "use_hard_excluded_fill": s.runner.use_hard_excluded_fill,
                "use_retrieval_tail_fill": s.runner.use_retrieval_tail_fill,
                "include_retrieval_traces": s.runner.include_retrieval_traces,
                "use_irrelevance_heuristic": s.runner.use_irrelevance_heuristic,
                "use_multisignal_irrelevance_heuristic": (
                    s.runner.use_multisignal_irrelevance_heuristic
                ),
                "min_inclusion_fraction": s.runner.min_inclusion_fraction,
                "max_nei_fraction": s.runner.max_nei_fraction,
                "benchmark_min_inclusion_fraction": s.runner.benchmark_min_inclusion_fraction,
                "benchmark_max_nei_fraction": s.runner.benchmark_max_nei_fraction,
                "irrelevant_min_nei_fraction": s.runner.irrelevant_min_nei_fraction,
                "irrelevant_max_inclusion_met": s.runner.irrelevant_max_inclusion_met,
                "irrelevant_max_retrieval_prior": s.runner.irrelevant_max_retrieval_prior,
                "irrelevant_min_signal_count": s.runner.irrelevant_min_signal_count,
                "few_shot_bank_size": runtime.few_shot_bank_size,
                "benchmark_index_manifest_policy": s.runner.benchmark_index_manifest_policy,
                "benchmark_index_manifest": benchmark_index_manifest,
                "benchmark_corpus_manifest": benchmark_corpus_manifest,
                "ctgov_dir": str(s.paths.ctgov_dir),
                "bm25_mode": s.retrieval.bm25_mode,
                "bm25_index_dir": str(active_bm25_index),
                "fielded_bm25_index_dir": str(
                    s.retrieval.fielded_bm25_index_dir
                    or (s.paths.indices_dir / "bm25_trec2021_fielded")
                ),
                "bm25_top_k": s.retrieval.bm25_top_k,
                "fielded_bm25_per_field_k": s.retrieval.fielded_bm25_per_field_k,
                "fielded_rerank_retrieval_blend": s.retrieval.fielded_rerank_retrieval_blend,
                "dense_top_k": s.retrieval.dense_top_k,
                "bm25_rrf_weight": s.retrieval.bm25_rrf_weight,
                "dense_rrf_weight": s.retrieval.dense_rrf_weight,
                "fused_top_k": s.retrieval.fused_top_k,
                "rerank_top_k": s.retrieval.rerank_top_k,
                "listwise_top_k": s.retrieval.listwise_top_k,
                "corpus_loaded": runtime.corpus_loaded,
                "corpus_size": runtime.corpus_size,
                "mesh_loaded": runtime.mesh_loaded,
                "mesh_concepts": runtime.mesh_concepts,
                "qdrant_mode": runtime.qdrant_mode,
                "qdrant_url": runtime.qdrant_url,
                "qdrant_collection": runtime.qdrant_collection,
                "qdrant_collection_exists": runtime.qdrant_collection_exists,
                "qdrant_points_count": runtime.qdrant_points_count,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "topics": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote predictions: {output_path}")


if __name__ == "__main__":
    main()
