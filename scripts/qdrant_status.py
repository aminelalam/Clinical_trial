"""Print Qdrant server/embedded collection status without loading ML models."""

from __future__ import annotations

import json

from trial_matcher.config import get_settings
from trial_matcher.retrieval.dense import DenseRetriever


def main() -> int:
    s = get_settings()
    retriever = DenseRetriever()
    try:
        status = retriever.collection_status()
    except Exception as e:
        status = {
            "mode": retriever.qdrant_mode(),
            "collection": s.qdrant.collection,
            "exists": False,
            "points_count": 0,
            "reachable": False,
            "error": repr(e),
        }
    else:
        status["reachable"] = True
    status.update(
        {
            "url": s.qdrant.url,
            "api_key_configured": bool(s.qdrant.api_key),
            "path": s.qdrant.path or None,
            "timeout_seconds": s.qdrant.timeout_seconds,
        }
    )
    if status.get("status") is not None:
        status["status"] = str(status["status"])
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
