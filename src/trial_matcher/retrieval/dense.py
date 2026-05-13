"""Dense retrieval with MedCPT (NCBI) and Qdrant.

MedCPT consists of:
  - Query encoder (ncbi/MedCPT-Query-Encoder)
  - Article encoder (ncbi/MedCPT-Article-Encoder)
Both produce 768-d vectors in the same space. We index trials with the
article encoder and embed patient queries with the query encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..logging import logger
from ..models.agent_state import TrialCandidate

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient


class DenseRetriever:
    """MedCPT + Qdrant. Lazy-loads models on first use to keep import cheap."""

    def __init__(self, collection: str | None = None):
        s = get_settings()
        self.collection = collection or s.qdrant.collection
        self._query_tokenizer = None
        self._query_model = None
        self._article_tokenizer = None
        self._article_model = None
        self._client: QdrantClient | None = None
        self._client_mode: str | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _load_query_encoder(self) -> None:
        if self._query_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        s = get_settings()
        logger.info(f"Loading MedCPT-Query-Encoder: {s.retrieval.medcpt_query_encoder}")
        self._query_tokenizer = AutoTokenizer.from_pretrained(s.retrieval.medcpt_query_encoder)
        self._query_model = AutoModel.from_pretrained(s.retrieval.medcpt_query_encoder)
        self._query_model.eval()
        if torch.cuda.is_available():
            self._query_model = self._query_model.cuda()

    def _load_article_encoder(self) -> None:
        if self._article_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        s = get_settings()
        logger.info(f"Loading MedCPT-Article-Encoder: {s.retrieval.medcpt_article_encoder}")
        self._article_tokenizer = AutoTokenizer.from_pretrained(
            s.retrieval.medcpt_article_encoder
        )
        self._article_model = AutoModel.from_pretrained(s.retrieval.medcpt_article_encoder)
        self._article_model.eval()
        if torch.cuda.is_available():
            self._article_model = self._article_model.cuda()

    def _client_(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            s = get_settings()
            mode = self.qdrant_mode()
            self._client_mode = mode
            if mode == "embedded":
                if not s.qdrant.path:
                    raise RuntimeError("QDRANT_MODE=embedded requires QDRANT_PATH")
                Path(s.qdrant.path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Using embedded Qdrant at {s.qdrant.path}")
                self._client = QdrantClient(path=s.qdrant.path, timeout=s.qdrant.timeout_seconds)
            else:
                logger.info(f"Using Qdrant server at {s.qdrant.url}")
                self._client = QdrantClient(
                    url=s.qdrant.url,
                    api_key=s.qdrant.api_key or None,
                    timeout=s.qdrant.timeout_seconds,
                )
        return self._client

    @staticmethod
    def qdrant_mode() -> str:
        """Resolve the active Qdrant mode from settings.

        ``server`` intentionally ignores ``QDRANT_PATH``. This prevents an old
        embedded development setting from silently overriding the production
        server or Azure/Qdrant Cloud endpoint.
        """
        s = get_settings()
        if s.qdrant.mode == "auto":
            return "embedded" if s.qdrant.path else "server"
        return s.qdrant.mode

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_query(self, text: str) -> list[float]:
        """Encode a single query into a 768-d vector."""
        self._load_query_encoder()
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._load_query_encoder()
        with torch.no_grad():
            enc = self._query_tokenizer(  # type: ignore[union-attr]
                texts, truncation=True, padding=True, return_tensors="pt", max_length=64
            )
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            out = self._query_model(**enc).last_hidden_state[:, 0, :]  # type: ignore[union-attr]
        return out.cpu().tolist()

    def encode_articles(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        import torch

        self._load_article_encoder()
        results: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._article_tokenizer(  # type: ignore[union-attr]
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                out = self._article_model(**enc).last_hidden_state[:, 0, :]  # type: ignore[union-attr]
                results.extend(out.cpu().tolist())
        return results

    # ------------------------------------------------------------------
    # Qdrant ops
    # ------------------------------------------------------------------
    def ensure_collection(self, vector_size: int = 768) -> None:
        from qdrant_client.http.models import Distance, HnswConfigDiff, VectorParams

        s = get_settings()
        client = self._client_()
        if not client.collection_exists(self.collection):
            logger.info(f"Creating Qdrant collection '{self.collection}'")
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    on_disk=s.qdrant.on_disk_vectors,
                ),
                hnsw_config=HnswConfigDiff(on_disk=s.qdrant.on_disk_hnsw),
                on_disk_payload=s.qdrant.on_disk_payload,
            )

    def collection_status(self) -> dict[str, Any]:
        """Return lightweight Qdrant collection diagnostics without loading ML models."""
        client = self._client_()
        out: dict[str, Any] = {
            "mode": self._client_mode or self.qdrant_mode(),
            "collection": self.collection,
            "exists": False,
            "points_count": 0,
        }
        if not client.collection_exists(self.collection):
            return out
        info = client.get_collection(self.collection)
        out["exists"] = True
        out["status"] = getattr(info, "status", None)
        out["points_count"] = int(getattr(info, "points_count", 0) or 0)
        out["vectors_count"] = int(getattr(info, "vectors_count", 0) or 0)
        return out

    def upsert(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.http.models import PointStruct

        client = self._client_()
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name=self.collection, points=points)

    def retrieve(
        self,
        query_text: str,
        k: int | None = None,
        filters: dict | None = None,
    ) -> list[TrialCandidate]:
        s = get_settings()
        k = k or s.retrieval.dense_top_k
        client = self._client_()
        vec = self.encode_query(query_text)

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        qfilter = None
        if filters:
            must = []
            for key, value in filters.items():
                if isinstance(value, list):
                    must.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    must.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qfilter = Filter(must=must) if must else None

        # qdrant-client >= 1.13 removed .search(); use .query_points() instead.
        # The response is QueryResponse(points=[ScoredPoint, ...]).
        response = client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        )
        hits = response.points
        return [
            TrialCandidate(
                nct_id=str(h.payload.get("nct_id")) if h.payload else "",
                score=float(h.score),
                source="dense",
                rank=i + 1,
                title=h.payload.get("title") if h.payload else None,
            )
            for i, h in enumerate(hits)
        ]
