"""Audit qrel/golden-trial coverage through the agent's recorded stages.

The runner must be executed with ``--include-retrieval-traces`` so each topic
contains ``diagnostics.retrieval_traces`` with NCT IDs by pipeline stage.
This script is LLM-free; it only explains whether official qrels reached each
agent stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trial_matcher.ingestion.trec_parser import Qrel, parse_qrels

DEFAULT_STAGES = (
    "bm25_candidates",
    "fused_candidates",
    "reranked_candidates",
    "final_candidates",
    "ranked_trials",
)
DEFAULT_CUTOFFS = (10, 20, 50, 100, 200, 500, 1000)

STAGE_ALIASES = {
    "bm25_candidates": ("bm25",),
    "dense_candidates": ("dense",),
    "fused_candidates": ("fused",),
    "reranked_candidates": ("reranked",),
    "listwise_candidates": ("listwise",),
    "final_candidates": ("final",),
    "ranked_trials": ("ranked",),
    "judged_top10": ("judged",),
}


def qrels_by_topic(qrels: Iterable[Qrel]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for q in qrels:
        out.setdefault(q.topic_id, {})[q.nct_id] = int(q.grade)
    return out


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _ranked_trial_ids(topic: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for row in topic.get("ranked_trials", []) or []:
        nct_id = row.get("nct_id") if isinstance(row, dict) else getattr(row, "nct_id", None)
        if nct_id:
            out.append(str(nct_id))
    return out


def _trace_ids_for_stage(topic: dict[str, Any], traces: dict[str, Any], stage: str) -> list[str] | None:
    if stage in traces:
        return list(traces.get(stage) or [])
    for alias in STAGE_ALIASES.get(stage, ()):
        if alias in traces:
            return list(traces.get(alias) or [])
    if stage == "ranked_trials":
        return _ranked_trial_ids(topic)
    return None


def _unique_ids(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        nct_id = str(raw or "")
        if nct_id and nct_id not in seen:
            out.append(nct_id)
            seen.add(nct_id)
    return out


def stage_metrics(ids: Iterable[str], qrels: dict[str, int], *, cutoffs: Iterable[int]) -> dict[str, Any]:
    ranked_ids = _unique_ids(ids)
    relevant_total = sum(1 for grade in qrels.values() if grade > 0)
    eligible_total = sum(1 for grade in qrels.values() if grade == 2)
    judged_total = len(qrels)
    out: dict[str, Any] = {
        "count": len(ranked_ids),
        "unique_count": len(ranked_ids),
        "judged_total": judged_total,
        "relevant_total": relevant_total,
        "eligible_total": eligible_total,
    }
    for cutoff in cutoffs:
        top = ranked_ids[:cutoff]
        judged_hits = sum(1 for nct_id in top if nct_id in qrels)
        relevant_hits = sum(1 for nct_id in top if qrels.get(nct_id, 0) > 0)
        eligible_hits = sum(1 for nct_id in top if qrels.get(nct_id, 0) == 2)
        out[f"judged_hits@{cutoff}"] = judged_hits
        out[f"relevant_hits@{cutoff}"] = relevant_hits
        out[f"eligible_hits@{cutoff}"] = eligible_hits
        out[f"recall_relevant@{cutoff}"] = _safe_div(relevant_hits, relevant_total)
        out[f"recall_eligible@{cutoff}"] = _safe_div(eligible_hits, eligible_total)
        out[f"judged_fraction@{cutoff}"] = _safe_div(judged_hits, len(top))
    return out


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_topics(topic_rows: list[dict[str, Any]], *, cutoffs: Iterable[int]) -> dict[str, Any]:
    stages = sorted({stage for row in topic_rows for stage in row.get("stages", {})})
    out: dict[str, Any] = {
        "n_topics": len(topic_rows),
        "stages": {},
        "topics_missing_traces": [
            row["topic_id"] for row in topic_rows if row.get("missing_trace_stages")
        ],
    }
    for stage in stages:
        present = [row["stages"][stage] for row in topic_rows if stage in row["stages"]]
        summary: dict[str, Any] = {
            "mean_unique_count": _mean(m["unique_count"] for m in present),
            "topics_with_zero_relevant@20": [
                row["topic_id"]
                for row in topic_rows
                if stage in row["stages"] and row["stages"][stage].get("relevant_hits@20", 0) == 0
            ],
            "topics_with_zero_eligible@20": [
                row["topic_id"]
                for row in topic_rows
                if stage in row["stages"] and row["stages"][stage].get("eligible_hits@20", 0) == 0
            ],
        }
        for cutoff in cutoffs:
            summary[f"mean_recall_relevant@{cutoff}"] = _mean(
                m.get(f"recall_relevant@{cutoff}", 0.0) for m in present
            )
            summary[f"mean_recall_eligible@{cutoff}"] = _mean(
                m.get(f"recall_eligible@{cutoff}", 0.0) for m in present
            )
            summary[f"mean_judged_fraction@{cutoff}"] = _mean(
                m.get(f"judged_fraction@{cutoff}", 0.0) for m in present
            )
        out["stages"][stage] = summary
    trial_not_found_by_topic: dict[str, int] = {}
    for row in topic_rows:
        count = int(row.get("trial_not_found_count") or 0)
        if count:
            trial_not_found_by_topic[row["topic_id"]] = count
    out["trial_not_found_count_total"] = sum(trial_not_found_by_topic.values())
    out["topics_with_trial_not_found"] = trial_not_found_by_topic
    gaps = [row.get("final_to_ranked_gap_at_20", {}) for row in topic_rows]
    out["final_to_ranked_gap_at_20"] = {
        "missing_count_total": sum(int(g.get("missing_count", 0) or 0) for g in gaps),
        "relevant_lost_total": sum(int(g.get("relevant_lost", 0) or 0) for g in gaps),
        "eligible_lost_total": sum(int(g.get("eligible_lost", 0) or 0) for g in gaps),
        "topics_with_relevant_lost": [
            row["topic_id"]
            for row in topic_rows
            if int((row.get("final_to_ranked_gap_at_20") or {}).get("relevant_lost", 0) or 0)
            > 0
        ],
        "topics_with_eligible_lost": [
            row["topic_id"]
            for row in topic_rows
            if int((row.get("final_to_ranked_gap_at_20") or {}).get("eligible_lost", 0) or 0)
            > 0
        ],
    }
    return out


def audit_predictions(
    predictions_path: Path,
    qrels_path: Path,
    *,
    stages: Iterable[str] = DEFAULT_STAGES,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    qrels = qrels_by_topic(parse_qrels(qrels_path))
    cutoffs = tuple(sorted({int(c) for c in cutoffs}))
    stage_list = [s for s in stages if s]

    topic_rows: list[dict[str, Any]] = []
    for topic in predictions.get("topics", []):
        topic_id = str(topic.get("topic_id", ""))
        diagnostics = topic.get("diagnostics") or {}
        traces = diagnostics.get("retrieval_traces") or {}
        topic_qrels = qrels.get(topic_id, {})
        row: dict[str, Any] = {
            "topic_id": topic_id,
            "trial_not_found_count": int(diagnostics.get("trial_not_found_count", 0) or 0),
            "stages": {},
            "missing_trace_stages": [],
        }
        for stage in stage_list:
            ids = _trace_ids_for_stage(topic, traces, stage)
            if ids is None:
                row["missing_trace_stages"].append(stage)
                continue
            row["stages"][stage] = stage_metrics(ids, topic_qrels, cutoffs=cutoffs)
        final_ids = _unique_ids(
            _trace_ids_for_stage(topic, traces, "final_candidates") or []
        )[:20]
        ranked_ids = _unique_ids(
            _trace_ids_for_stage(topic, traces, "ranked_trials") or []
        )[:20]
        ranked_set = set(ranked_ids)
        missing = [nct_id for nct_id in final_ids if nct_id not in ranked_set]
        row["final_to_ranked_gap_at_20"] = {
            "final_candidates_at_20": len(final_ids),
            "ranked_trials_at_20": len(ranked_ids),
            "missing_count": len(missing),
            "relevant_lost": sum(1 for nct_id in missing if topic_qrels.get(nct_id, 0) > 0),
            "eligible_lost": sum(1 for nct_id in missing if topic_qrels.get(nct_id, 0) == 2),
            "missing_relevant_ids": [
                nct_id for nct_id in missing if topic_qrels.get(nct_id, 0) > 0
            ],
            "missing_eligible_ids": [
                nct_id for nct_id in missing if topic_qrels.get(nct_id, 0) == 2
            ],
        }
        topic_rows.append(row)

    return {
        "predictions": str(predictions_path),
        "qrels": str(qrels_path),
        "stages": stage_list,
        "cutoffs": list(cutoffs),
        "metadata": {
            "n_topics": len(predictions.get("topics", [])),
            "settings": (predictions.get("metadata") or {}).get("settings", {}),
        },
        "aggregate": aggregate_topics(topic_rows, cutoffs=cutoffs),
        "topics_detail": topic_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    p.add_argument("--cutoffs", type=int, nargs="+", default=list(DEFAULT_CUTOFFS))
    args = p.parse_args()

    summary = audit_predictions(
        args.predictions,
        args.qrels,
        stages=args.stages,
        cutoffs=args.cutoffs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
