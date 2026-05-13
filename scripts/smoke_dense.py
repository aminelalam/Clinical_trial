"""Quick smoke test for the dense MedCPT index.

Uses the configured Qdrant mode. For normal runs this should be server mode:
    QDRANT_MODE=server QDRANT_URL=http://localhost:6333 python scripts/smoke_dense.py
"""

from __future__ import annotations

import os
import sys
import time

print(
    "[start] "
    f"cwd={os.getcwd()} "
    f"qdrant_mode={os.environ.get('QDRANT_MODE')} "
    f"qdrant_url={os.environ.get('QDRANT_URL')} "
    f"qdrant_path={os.environ.get('QDRANT_PATH')}",
    flush=True,
)
sys.stdout.flush()

t0 = time.time()
from trial_matcher.retrieval.dense import DenseRetriever  # noqa: E402
from trial_matcher.config import get_settings  # noqa: E402

s = get_settings()
print(
    f"[{time.time()-t0:.1f}s] DenseRetriever imported "
    f"settings_mode={s.qdrant.mode} settings_url={s.qdrant.url}",
    flush=True,
)

r = DenseRetriever()
print(f"[{time.time()-t0:.1f}s] retriever instantiated", flush=True)

r.ensure_collection(vector_size=768)
print(f"[{time.time()-t0:.1f}s] collection ensured", flush=True)

queries = [
    "47-year-old woman with HER2 positive metastatic breast cancer ECOG 1",
    "62-year-old man with stage IV NSCLC harboring an EGFR exon 19 deletion",
]
for q in queries:
    t1 = time.time()
    hits = r.retrieve(q, k=5)
    print(f"[{time.time()-t0:.1f}s] query='{q[:50]}...' got {len(hits)} hits in {time.time()-t1:.2f}s", flush=True)
    for h in hits:
        print(f"  rank={h.rank:>2}  {h.nct_id}  score={h.score:.3f}  title={(h.title or '')[:80]}", flush=True)

print(f"[done {time.time()-t0:.1f}s]", flush=True)
