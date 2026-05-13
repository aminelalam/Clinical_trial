"""Baseline: BM25S only — no dense, no rerank, no eligibility cascade.

The point is to have a published comparison number for the memoria.

Usage:
    python eval/baselines/bm25_only.py --topics data/trec_ct/topics_2021.xml \\
        --output predictions_bm25_2021.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from trial_matcher.ingestion.trec_parser import parse_topics
from trial_matcher.retrieval.bm25 import BM25Retriever


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    bm25 = BM25Retriever()
    bm25.load()

    topics = parse_topics(args.topics)
    out: dict = {"metadata": {"system": "bm25_only", "top_k": args.top_k}, "topics": []}
    for t in topics:
        cands = bm25.retrieve(t.text, k=args.top_k)
        out["topics"].append({
            "topic_id": t.topic_id,
            "ranked_trials": [
                {
                    "nct_id": c.nct_id,
                    "rank": i + 1,
                    "score": float(c.score),
                    "label": "eligible",  # baseline does not classify; treat as TREC qrel 2 always
                    "trec_qrel": 2,
                    "rationale": "",
                    "n_inclusion_met": 0,
                    "n_inclusion_total": 0,
                    "n_exclusion_met": 0,
                    "n_exclusion_total": 0,
                    "fraction_nei": 0.0,
                }
                for i, c in enumerate(cands)
            ],
            "questions": [],
            "dossiers": [],
        })

    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
