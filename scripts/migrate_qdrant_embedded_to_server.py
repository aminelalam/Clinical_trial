"""Copy an existing embedded Qdrant collection to a Qdrant server.

This reuses the already-computed MedCPT vectors from ``data/indices/qdrant``
and upserts them into the configured Qdrant server. It is much faster and
cheaper than rebuilding the dense index from CT.gov with the article encoder.

Typical local flow:
    docker compose up -d qdrant
    python scripts/migrate_qdrant_embedded_to_server.py --source-path data/indices/qdrant
    python scripts/qdrant_status.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, HnswConfigDiff, PointStruct, VectorParams
from tqdm import tqdm

from trial_matcher.config import get_settings
from trial_matcher.logging import logger, setup_logging


def _extract_vector(vector: Any) -> list[float]:
    if isinstance(vector, dict):
        if not vector:
            return []
        vector = next(iter(vector.values()))
    return list(vector or [])


def _infer_vector_size(client: QdrantClient, collection: str) -> int:
    records, _ = client.scroll(
        collection_name=collection,
        limit=1,
        with_payload=False,
        with_vectors=True,
    )
    if not records:
        raise RuntimeError(f"Source collection {collection!r} is empty")
    vector = _extract_vector(records[0].vector)
    if not vector:
        raise RuntimeError(f"Source collection {collection!r} has no vectors")
    return len(vector)


def _ensure_target_collection(
    client: QdrantClient,
    *,
    collection: str,
    vector_size: int,
    recreate: bool,
    on_disk_vectors: bool,
    on_disk_hnsw: bool,
    on_disk_payload: bool,
) -> None:
    if client.collection_exists(collection):
        if not recreate:
            return
        logger.warning(f"Deleting target collection {collection!r} before migration")
        client.delete_collection(collection_name=collection)

    logger.info(
        f"Creating target collection {collection!r} "
        f"(vector_size={vector_size}, on_disk_vectors={on_disk_vectors}, "
        f"on_disk_hnsw={on_disk_hnsw}, on_disk_payload={on_disk_payload})"
    )
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            on_disk=on_disk_vectors,
        ),
        hnsw_config=HnswConfigDiff(on_disk=on_disk_hnsw),
        on_disk_payload=on_disk_payload,
    )


def main() -> int:
    setup_logging()
    s = get_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, default=Path("data/indices/qdrant"))
    parser.add_argument("--target-url", default=s.qdrant.url)
    parser.add_argument("--target-api-key", default=s.qdrant.api_key)
    parser.add_argument("--collection", default=s.qdrant.collection)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 = full collection")
    parser.add_argument("--recreate-target", action="store_true")
    parser.add_argument("--in-memory-vectors", action="store_true")
    parser.add_argument("--in-memory-hnsw", action="store_true")
    parser.add_argument("--in-memory-payload", action="store_true")
    args = parser.parse_args()

    if not args.source_path.exists():
        raise FileNotFoundError(args.source_path)

    source = QdrantClient(path=str(args.source_path), timeout=s.qdrant.timeout_seconds)
    target = QdrantClient(
        url=args.target_url,
        api_key=args.target_api_key or None,
        timeout=s.qdrant.timeout_seconds,
    )

    if not source.collection_exists(args.collection):
        raise RuntimeError(f"Source collection {args.collection!r} does not exist")

    source_count = int(source.count(args.collection, exact=True).count)
    target_count = (
        int(target.count(args.collection, exact=True).count)
        if target.collection_exists(args.collection)
        else 0
    )
    if args.limit == 0 and target_count >= source_count and not args.recreate_target:
        logger.info(
            f"Target already has {target_count} points; source has {source_count}. "
            "Nothing to migrate."
        )
        return 0

    vector_size = _infer_vector_size(source, args.collection)
    _ensure_target_collection(
        target,
        collection=args.collection,
        vector_size=vector_size,
        recreate=args.recreate_target,
        on_disk_vectors=not args.in_memory_vectors,
        on_disk_hnsw=not args.in_memory_hnsw,
        on_disk_payload=not args.in_memory_payload,
    )

    total = min(source_count, args.limit) if args.limit else source_count
    logger.info(
        f"Migrating {total} points from embedded {args.source_path} "
        f"to {args.target_url} collection={args.collection!r}"
    )
    started = time.time()
    migrated = 0
    offset: int | str | None = None
    pbar = tqdm(total=total, desc="Migrating Qdrant points", unit="points")

    while migrated < total:
        limit = min(args.batch_size, total - migrated)
        records, offset = source.scroll(
            collection_name=args.collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        points = [
            PointStruct(
                id=record.id,
                vector=_extract_vector(record.vector),
                payload=dict(record.payload or {}),
            )
            for record in records
        ]
        target.upsert(collection_name=args.collection, points=points, wait=True)
        migrated += len(points)
        pbar.update(len(points))
        if offset is None:
            break

    pbar.close()
    elapsed = time.time() - started
    final_count = int(target.count(args.collection, exact=True).count)
    logger.info(
        f"Migrated {migrated} points in {elapsed/60:.1f} min. "
        f"Target now has {final_count} points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
