"""Audit retrieval-stage recall against TREC qrels.

This script is intentionally LLM-free. It measures whether relevant trials are
present in the first-stage candidate sets before eligibility evaluation can
exclude them. Use it to decide whether the next quality work belongs in
retrieval/reranking or in eligibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from trial_matcher.config import get_settings
from trial_matcher.ingestion.lazy_corpus import LazyTrialCorpus
from trial_matcher.ingestion.trec_parser import Qrel, TrecTopic, parse_qrels, parse_topics
from trial_matcher.models.agent_state import TrialCandidate
from trial_matcher.retrieval.bm25 import BM25Retriever
from trial_matcher.retrieval.dense import DenseRetriever
from trial_matcher.retrieval.fielded_bm25 import BM25_FIELD_NAMES, FieldedBM25Retriever
from trial_matcher.retrieval.hybrid import reciprocal_rank_fusion
from trial_matcher.retrieval.reranker import CrossEncoderReranker


DEFAULT_CUTOFFS = (10, 20, 50, 100, 200)


@dataclass(frozen=True)
class TopicSelection:
    topics: list[TrecTopic]
    missing_ids: list[str]


def select_topics(
    topics: Iterable[TrecTopic],
    *,
    topic_ids: str = "",
    topic_limit: int = 0,
) -> TopicSelection:
    """Select topics deterministically, applying explicit IDs before limit."""
    topic_list = list(topics)
    if topic_ids.strip():
        requested = [tid.strip() for tid in topic_ids.split(",") if tid.strip()]
        by_id = {t.topic_id: t for t in topic_list}
        missing = [tid for tid in requested if tid not in by_id]
        selected = [by_id[tid] for tid in requested if tid in by_id]
    else:
        missing = []
        selected = topic_list
    if topic_limit > 0:
        selected = selected[:topic_limit]
    return TopicSelection(topics=selected, missing_ids=missing)


def qrels_by_topic(qrels: Iterable[Qrel]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for q in qrels:
        out.setdefault(q.topic_id, {})[q.nct_id] = int(q.grade)
    return out


def _candidate_ids(candidates: Iterable[TrialCandidate]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        if cand.nct_id and cand.nct_id not in seen:
            out.append(cand.nct_id)
            seen.add(cand.nct_id)
    return out


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def stage_metrics(
    candidates: list[TrialCandidate],
    qrels: dict[str, int],
    *,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Compute judged/relevant recall diagnostics for one stage."""
    ids = _candidate_ids(candidates)
    relevant_total = sum(1 for grade in qrels.values() if grade > 0)
    eligible_total = sum(1 for grade in qrels.values() if grade == 2)
    judged_total = len(qrels)

    out: dict[str, Any] = {
        "count": len(candidates),
        "unique_count": len(ids),
        "judged_total": judged_total,
        "relevant_total": relevant_total,
        "eligible_total": eligible_total,
    }
    for cutoff in cutoffs:
        top = ids[:cutoff]
        judged_hits = sum(1 for nct_id in top if nct_id in qrels)
        relevant_hits = sum(1 for nct_id in top if qrels.get(nct_id, 0) > 0)
        eligible_hits = sum(1 for nct_id in top if qrels.get(nct_id, 0) == 2)
        excludes_hits = sum(1 for nct_id in top if qrels.get(nct_id, 0) == 1)
        irrelevant_hits = sum(1 for nct_id in top if qrels.get(nct_id, -1) == 0)
        unjudged = len(top) - judged_hits
        out[f"judged_hits@{cutoff}"] = judged_hits
        out[f"relevant_hits@{cutoff}"] = relevant_hits
        out[f"eligible_hits@{cutoff}"] = eligible_hits
        out[f"excludes_hits@{cutoff}"] = excludes_hits
        out[f"irrelevant_hits@{cutoff}"] = irrelevant_hits
        out[f"unjudged@{cutoff}"] = unjudged
        out[f"recall_relevant@{cutoff}"] = _safe_div(relevant_hits, relevant_total)
        out[f"recall_eligible@{cutoff}"] = _safe_div(eligible_hits, eligible_total)
        out[f"judged_fraction@{cutoff}"] = _safe_div(judged_hits, len(top))
    return out


def overlap_metrics(a: list[TrialCandidate], b: list[TrialCandidate], *, cutoff: int) -> dict[str, Any]:
    a_ids = set(_candidate_ids(a)[:cutoff])
    b_ids = set(_candidate_ids(b)[:cutoff])
    union = a_ids | b_ids
    inter = a_ids & b_ids
    return {
        "cutoff": cutoff,
        "intersection": len(inter),
        "union": len(union),
        "jaccard": _safe_div(len(inter), len(union)),
    }


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_topics(topic_rows: list[dict[str, Any]], *, cutoffs: Iterable[int]) -> dict[str, Any]:
    stages = sorted({stage for row in topic_rows for stage in row.get("stages", {})})
    out: dict[str, Any] = {"n_topics": len(topic_rows), "stages": {}}
    for stage in stages:
        present = [row["stages"][stage] for row in topic_rows if stage in row["stages"]]
        if not present:
            continue
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
    overlap_values = []
    for row in topic_rows:
        overlap = (row.get("overlap") or {}).get("bm25_dense@100")
        if overlap is not None:
            overlap_values.append(overlap.get("jaccard", 0.0))
    out["mean_bm25_dense_jaccard@100"] = _mean(overlap_values)
    out["missing_reasons"] = {}
    for stage in stages:
        totals: dict[str, int] = {}
        for row in topic_rows:
            reasons = (row.get("missing_reasons") or {}).get(stage) or {}
            for key, value in reasons.items():
                totals[key] = totals.get(key, 0) + int(value or 0)
        if totals:
            out["missing_reasons"][stage] = totals
    return out


def _coverage_summary(qrels: dict[str, dict[str, int]], ids: set[str] | None) -> dict[str, Any] | None:
    if ids is None:
        return None
    all_judged = {nct_id for topic_qrels in qrels.values() for nct_id in topic_qrels}
    relevant = {
        nct_id
        for topic_qrels in qrels.values()
        for nct_id, grade in topic_qrels.items()
        if grade > 0
    }
    eligible = {
        nct_id
        for topic_qrels in qrels.values()
        for nct_id, grade in topic_qrels.items()
        if grade == 2
    }

    def one(label: str, values: set[str]) -> dict[str, Any]:
        covered = len(values & ids)
        return {
            "label": label,
            "total": len(values),
            "covered": covered,
            "missing": len(values) - covered,
            "coverage": _safe_div(covered, len(values)),
        }

    return {
        "all_judged": one("all_judged", all_judged),
        "relevant_1or2": one("relevant_1or2", relevant),
        "eligible_2": one("eligible_2", eligible),
    }


def missing_reason_counts(
    *,
    topic_qrels: dict[str, int],
    candidates: list[TrialCandidate],
    corpus_ids: set[str] | None,
    index_ids: set[str] | None,
) -> dict[str, int]:
    """Explain why relevant qrels are absent from a retrieval stage."""
    retrieved = set(_candidate_ids(candidates))
    out = {
        "relevant_total": 0,
        "retrieved": 0,
        "not_in_corpus": 0,
        "not_in_index": 0,
        "not_retrieved": 0,
    }
    for nct_id, grade in topic_qrels.items():
        if grade <= 0:
            continue
        out["relevant_total"] += 1
        if nct_id in retrieved:
            out["retrieved"] += 1
        elif corpus_ids is not None and nct_id not in corpus_ids:
            out["not_in_corpus"] += 1
        elif index_ids is not None and nct_id not in index_ids:
            out["not_in_index"] += 1
        else:
            out["not_retrieved"] += 1
    return out


def _trial_text_getter(corpus: LazyTrialCorpus):
    def get_text(nct_id: str) -> str:
        trial = corpus.get(nct_id)
        if trial is None:
            return ""
        return f"{trial.title}\n{trial.brief_summary}\n{' '.join(trial.conditions)}"

    return get_text


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    topics_path = Path(args.topics)
    qrels_path = Path(args.qrels)
    topics = parse_topics(topics_path)
    selection = select_topics(
        topics,
        topic_ids=args.topic_ids,
        topic_limit=args.topic_limit,
    )
    if selection.missing_ids:
        missing = ", ".join(selection.missing_ids)
        raise ValueError(f"Requested topic IDs not found: {missing}")

    qrels = qrels_by_topic(parse_qrels(qrels_path))
    cutoffs = tuple(sorted({int(c) for c in args.cutoffs}))
    if args.bm25_mode == "fielded":
        active_index_dir = args.fielded_bm25_index_dir or args.bm25_index_dir
        settings.retrieval.bm25_mode = "fielded"
        if active_index_dir:
            settings.retrieval.fielded_bm25_index_dir = Path(active_index_dir)
        settings.retrieval.fielded_bm25_per_field_k = int(args.fielded_bm25_per_field_k)
        bm25 = FieldedBM25Retriever(active_index_dir) if active_index_dir else FieldedBM25Retriever()
    else:
        settings.retrieval.bm25_mode = "single"
        if args.bm25_index_dir:
            settings.retrieval.bm25_index_dir = Path(args.bm25_index_dir)
        bm25 = BM25Retriever(args.bm25_index_dir) if args.bm25_index_dir else BM25Retriever()
    index_ids = set(bm25.nct_ids)
    audit_corpus = LazyTrialCorpus(args.corpus_dir) if args.corpus_dir else None
    corpus_ids = set(audit_corpus) if audit_corpus is not None else None
    dense = None if args.no_dense else DenseRetriever()
    reranker = CrossEncoderReranker() if args.with_rerank else None
    rerank_corpus_dir = Path(args.corpus_dir or settings.paths.ctgov_dir)
    corpus = LazyTrialCorpus(rerank_corpus_dir) if args.with_rerank else None
    get_text = _trial_text_getter(corpus) if corpus is not None else None

    topic_rows: list[dict[str, Any]] = []
    for topic in selection.topics:
        print(f"[retrieval-audit] topic={topic.topic_id}", flush=True)
        topic_qrels = qrels.get(topic.topic_id, {})
        field_runs: dict[str, list[TrialCandidate]] = {}
        if args.bm25_mode == "fielded" and isinstance(bm25, FieldedBM25Retriever):
            field_runs = bm25.retrieve_by_field(topic.text, k=args.fielded_bm25_per_field_k)
            bm25_candidates = bm25.retrieve(
                topic.text,
                k=args.bm25_k,
                per_field_k=args.fielded_bm25_per_field_k,
            )
        else:
            bm25_candidates = bm25.retrieve(topic.text, k=args.bm25_k)
        dense_candidates: list[TrialCandidate] = []
        if dense is not None:
            dense_candidates = dense.retrieve(topic.text, k=args.dense_k)
        runs = [bm25_candidates]
        if dense_candidates:
            runs.append(dense_candidates)
        weights = [args.bm25_weight]
        if dense_candidates:
            weights.append(args.dense_weight)
        fused = reciprocal_rank_fusion(runs, weights=weights, top_k=args.fused_k)

        stages: dict[str, Any] = {
            "bm25": stage_metrics(bm25_candidates, topic_qrels, cutoffs=cutoffs),
            "fused": stage_metrics(fused, topic_qrels, cutoffs=cutoffs),
        }
        for field, run in field_runs.items():
            stages[f"bm25_field_{field}"] = stage_metrics(run, topic_qrels, cutoffs=cutoffs)
        if dense_candidates:
            stages["dense"] = stage_metrics(dense_candidates, topic_qrels, cutoffs=cutoffs)

        if reranker is not None and get_text is not None:
            reranked = reranker.rerank(topic.text, fused, get_text=get_text, top_k=args.rerank_k)
            stages["reranked"] = stage_metrics(reranked, topic_qrels, cutoffs=cutoffs)

        row = {
            "topic_id": topic.topic_id,
            "stages": stages,
            "missing_reasons": {
                "bm25": missing_reason_counts(
                    topic_qrels=topic_qrels,
                    candidates=bm25_candidates,
                    corpus_ids=corpus_ids,
                    index_ids=index_ids,
                ),
                "fused": missing_reason_counts(
                    topic_qrels=topic_qrels,
                    candidates=fused,
                    corpus_ids=corpus_ids,
                    index_ids=index_ids,
                ),
            },
            "overlap": {
                "bm25_dense@100": overlap_metrics(
                    bm25_candidates,
                    dense_candidates,
                    cutoff=min(100, args.bm25_k, args.dense_k),
                )
                if dense_candidates
                else None
            },
        }
        topic_rows.append(row)

    return {
        "topics": str(topics_path),
        "qrels": str(qrels_path),
        "topic_ids": [t.topic_id for t in selection.topics],
        "params": {
            "bm25_k": args.bm25_k,
            "dense_k": args.dense_k,
            "fused_k": args.fused_k,
            "bm25_weight": args.bm25_weight,
            "dense_weight": args.dense_weight if not args.no_dense else 0.0,
            "with_rerank": bool(args.with_rerank),
            "rerank_k": args.rerank_k if args.with_rerank else None,
            "cutoffs": list(cutoffs),
            "qdrant_mode": settings.qdrant.mode,
            "qdrant_collection": settings.qdrant.collection,
            "bm25_index_dir": str(args.bm25_index_dir) if args.bm25_index_dir else None,
            "bm25_mode": args.bm25_mode,
            "fielded_bm25_index_dir": str(args.fielded_bm25_index_dir) if args.fielded_bm25_index_dir else None,
            "fielded_bm25_per_field_k": args.fielded_bm25_per_field_k,
            "field_names": list(BM25_FIELD_NAMES) if args.bm25_mode == "fielded" else [],
            "corpus_dir": str(args.corpus_dir) if args.corpus_dir else None,
        },
        "coverage": {
            "corpus": _coverage_summary(qrels, corpus_ids),
            "bm25_index": _coverage_summary(qrels, index_ids),
        },
        "aggregate": aggregate_topics(topic_rows, cutoffs=cutoffs),
        "topics_detail": topic_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topics", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--topic-limit", type=int, default=0)
    p.add_argument("--topic-ids", default="")
    p.add_argument("--bm25-k", type=int, default=200)
    p.add_argument("--bm25-mode", choices=["single", "fielded"], default="single")
    p.add_argument("--bm25-index-dir", type=Path, default=None)
    p.add_argument("--fielded-bm25-index-dir", type=Path, default=None)
    p.add_argument("--fielded-bm25-per-field-k", type=int, default=1000)
    p.add_argument("--corpus-dir", type=Path, default=None, help="Optional corpus dir for qrel coverage diagnostics")
    p.add_argument("--dense-k", type=int, default=200)
    p.add_argument("--fused-k", type=int, default=200)
    p.add_argument("--bm25-weight", type=float, default=None)
    p.add_argument("--dense-weight", type=float, default=None)
    p.add_argument("--with-rerank", action="store_true")
    p.add_argument("--rerank-k", type=int, default=50)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--cutoffs", type=int, nargs="+", default=list(DEFAULT_CUTOFFS))
    args = p.parse_args()
    settings = get_settings()
    if args.bm25_weight is None:
        args.bm25_weight = settings.retrieval.bm25_rrf_weight
    if args.dense_weight is None:
        args.dense_weight = settings.retrieval.dense_rrf_weight
    if args.dense_k <= 0:
        args.no_dense = True

    summary = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
