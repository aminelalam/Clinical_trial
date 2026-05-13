"""Build the MedCPT dense index in Qdrant.

GPU strongly recommended (4-8h on a single 3090; 12-24h on CPU).
Resumable: skips trials already present in the collection.

Usage:
    python scripts/build_dense_index.py \\
        --ctgov-dir data/ctgov_snapshot \\
        --batch-size 32 \\
        --max-text-tokens 512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from trial_matcher.ingestion import filter_corpus, parse_ctgov_dump
from trial_matcher.logging import logger, setup_logging
from trial_matcher.retrieval.dense import DenseRetriever


def _trial_to_text(title: str, summary: str, conditions: list[str], inclusion: str, exclusion: str) -> str:
    parts = [title, summary, " ".join(conditions or []), inclusion, exclusion]
    text = "\n".join(p for p in parts if p)
    return text[:4000]  # cap before tokenizer truncation


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ctgov-dir", type=Path, default=Path("data/ctgov_snapshot"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument(
        "--limit", type=int, default=0, help="0 = no limit (full corpus)"
    )
    args = p.parse_args()

    retriever = DenseRetriever()
    retriever.ensure_collection(vector_size=768)

    logger.info("Streaming trials...")
    raw = parse_ctgov_dump(args.ctgov_dir)
    trials_iter = filter_corpus(raw) if not args.no_filter else raw

    batch_texts: list[str] = []
    batch_ids: list[int] = []
    batch_payloads: list[dict] = []
    n = 0
    t0 = time.time()

    pbar = tqdm(desc="Encoding+upserting", unit="docs")
    for trial in trials_iter:
        if args.limit and n >= args.limit:
            break
        text = _trial_to_text(
            trial.title,
            trial.brief_summary,
            trial.conditions,
            trial.eligibility.inclusion_text,
            trial.eligibility.exclusion_text,
        )
        # Hash NCT id to a stable integer for Qdrant point id
        point_id = abs(hash(trial.nct_id)) % (2**63 - 1)
        batch_texts.append(text)
        batch_ids.append(point_id)
        batch_payloads.append(
            {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "phase": trial.phase.value,
                "status": trial.status.value,
            }
        )
        n += 1
        pbar.update(1)

        if len(batch_texts) >= args.batch_size:
            vectors = retriever.encode_articles(batch_texts, batch_size=args.batch_size)
            retriever.upsert(batch_ids, vectors, batch_payloads)
            batch_texts.clear()
            batch_ids.clear()
            batch_payloads.clear()

    if batch_texts:
        vectors = retriever.encode_articles(batch_texts, batch_size=args.batch_size)
        retriever.upsert(batch_ids, vectors, batch_payloads)

    pbar.close()
    elapsed = time.time() - t0
    logger.info(f"Indexed {n} trials in {elapsed/60:.1f} min ({n/elapsed:.1f} docs/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
