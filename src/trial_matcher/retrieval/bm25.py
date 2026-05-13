"""BM25S-backed lexical retriever.

BM25S (https://github.com/xhluca/bm25s) is ~500x faster than rank-bm25 and
matches Elasticsearch on BEIR. No Java required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config import get_settings
from ..logging import logger
from ..models.agent_state import TrialCandidate

if TYPE_CHECKING:  # pragma: no cover
    import bm25s


class BM25Retriever:
    """Wrapper around bm25s.BM25 with persisted state and Trial primary_text indexing."""

    def __init__(self, index_dir: Path | str | None = None):
        settings = get_settings()
        configured_dir = settings.retrieval.bm25_index_dir
        self.index_dir = Path(index_dir or configured_dir or settings.paths.indices_dir / "bm25")
        self._bm25: bm25s.BM25 | None = None
        self._nct_ids: list[str] = []
        self._k1 = settings.retrieval.bm25_k1
        self._b = settings.retrieval.bm25_b

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, corpus_texts: list[str], nct_ids: list[str]) -> None:
        """Build the BM25 index from parallel lists of documents and NCT ids."""
        import bm25s
        from bm25s import tokenize

        if len(corpus_texts) != len(nct_ids):
            raise ValueError("corpus_texts and nct_ids must be the same length")

        logger.info(f"Tokenizing {len(corpus_texts)} documents for BM25S")
        tokens = tokenize(corpus_texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25(k1=self._k1, b=self._b)
        self._bm25.index(tokens)
        self._nct_ids = list(nct_ids)
        logger.info(f"BM25S index built: {len(nct_ids)} docs, k1={self._k1}, b={self._b}")

    def save(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("Cannot save: index not built")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(self.index_dir))
        (self.index_dir / "nct_ids.txt").write_text(
            "\n".join(self._nct_ids), encoding="utf-8"
        )
        logger.info(f"BM25 index saved to {self.index_dir}")

    def load(self) -> None:
        import bm25s

        nct_path = self.index_dir / "nct_ids.txt"
        if not nct_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_dir}. Run build_bm25_index.py first."
            )
        self._bm25 = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        self._nct_ids = nct_path.read_text(encoding="utf-8").splitlines()
        logger.info(f"BM25 index loaded: {len(self._nct_ids)} docs")

    @property
    def nct_ids(self) -> list[str]:
        """Return indexed NCT IDs, loading the persisted index metadata if needed."""
        if not self._nct_ids:
            self.load()
        return list(self._nct_ids)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[TrialCandidate]:
        """Return the top-k candidates for ``query``."""
        if self._bm25 is None:
            self.load()
        from bm25s import tokenize

        settings = get_settings()
        k = k or settings.retrieval.bm25_top_k

        q_tokens = tokenize([query], stopwords="en", show_progress=False)
        # bm25s.retrieve returns (results, scores) where results is shape (n_queries, k)
        results, scores = self._bm25.retrieve(q_tokens, k=k)  # type: ignore[union-attr]

        candidates: list[TrialCandidate] = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0])):
            nct_id = self._nct_ids[int(idx)]
            candidates.append(
                TrialCandidate(
                    nct_id=nct_id,
                    score=float(score),
                    source="bm25",
                    rank=rank + 1,
                )
            )
        return candidates
