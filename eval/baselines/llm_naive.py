"""Baseline: BM25 retrieval + 1-shot LLM eligibility (no cascade, no verifier).

Used to demonstrate that the cascade and verifier add value over a naive
LLM-judging pipeline. Output format matches the runner so the same
eval scripts work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from trial_matcher.ingestion.lazy_corpus import LazyTrialCorpus
from trial_matcher.ingestion.trec_parser import parse_topics
from trial_matcher.llm.client import UnifiedLLM
from trial_matcher.retrieval.bm25 import BM25Retriever


async def naive_eval(llm: UnifiedLLM, patient_text: str, trial_text: str) -> str:
    prompt = (
        "Decide whether a patient with the note below is likely eligible for the trial below. "
        "Reply with one of: eligible | excludes | irrelevant. Add a 1-line rationale.\n\n"
        f"Patient: {patient_text[:1500]}\n\n"
        f"Trial: {trial_text[:2500]}\n\n"
        "Output: <label>: <rationale>"
    )
    return await llm.acomplete(prompt, model="mini", temperature=0.0, max_tokens=120)


async def run(topics_path: Path, output_path: Path, top_k: int) -> None:
    from trial_matcher.config import get_settings

    s = get_settings()
    bm25 = BM25Retriever()
    bm25.load()
    llm = UnifiedLLM()
    corpus = LazyTrialCorpus(Path(s.paths.ctgov_dir))
    topics = parse_topics(topics_path)

    out: dict = {"metadata": {"system": "llm_naive", "top_k": top_k}, "topics": []}
    for t in topics:
        cands = bm25.retrieve(t.text, k=top_k)
        ranked = []
        for i, c in enumerate(cands):
            trial = corpus.get(c.nct_id)
            if trial is not None:
                text = f"{trial.title or ''}\n{trial.brief_summary or ''}\n{(trial.eligibility.inclusion_text or '')[:2000]}"
            else:
                text = (c.snippet or c.title or "")
            try:
                decision_line = await naive_eval(llm, t.text, text)
            except Exception:
                decision_line = "irrelevant: llm failed"
            label_token = decision_line.split(":", 1)[0].strip().lower()
            label_map = {"eligible": 2, "excludes": 1, "irrelevant": 0}
            qrel = label_map.get(label_token, 0)
            ranked.append(
                {
                    "nct_id": c.nct_id,
                    "rank": i + 1,
                    "score": float(c.score),
                    "label": label_token if label_token in label_map else "irrelevant",
                    "trec_qrel": qrel,
                    "rationale": decision_line,
                    "n_inclusion_met": 0,
                    "n_inclusion_total": 0,
                    "n_exclusion_met": 0,
                    "n_exclusion_total": 0,
                    "fraction_nei": 0.0,
                }
            )
        out["topics"].append(
            {"topic_id": t.topic_id, "ranked_trials": ranked, "questions": [], "dossiers": []}
        )

    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    asyncio.run(run(args.topics, args.output, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
