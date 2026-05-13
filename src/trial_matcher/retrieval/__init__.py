"""Retrieval stack: BM25 (BM25S), dense (MedCPT + Qdrant), hybrid (RRF), and rerankers."""

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fielded_bm25 import FieldedBM25Retriever
from .filters import apply_hard_filters, viable_count
from .hybrid import reciprocal_rank_fusion
from .hyde import HyDEGenerator
from .multiquery import MultiQueryGenerator  # EXPERIMENTAL — not wired into agent graph
from .reranker import CrossEncoderReranker

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "FieldedBM25Retriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "HyDEGenerator",
    "MultiQueryGenerator",
    "apply_hard_filters",
    "viable_count",
]
