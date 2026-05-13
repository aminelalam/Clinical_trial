"""Tool definitions exposed to the agent.

These are concrete, domain-specific tools (not generic 'search'). They are
defined once and reused by ``nodes.py`` so the agent's contract surface stays
small and testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import get_settings
from ..eligibility.cascade import EligibilityCascade
from ..models.agent_state import TrialCandidate
from ..models.criterion import Criterion
from ..models.eligibility import CriterionEval
from ..models.patient import PatientProfile
from ..models.trial import Trial
from ..nlp.criterion_extractor import CriterionExtractor
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever
from ..retrieval.fielded_bm25 import FieldedBM25Retriever
from ..retrieval.filters import apply_hard_filters
from ..retrieval.hyde import HyDEGenerator
from ..retrieval.listwise import ListwiseReranker
from ..retrieval.reranker import CrossEncoderReranker


class AgentTools:
    """Bundle of tools threaded through the agent.

    Holds the long-lived models (BM25 index, MedCPT, rerankers, LLM clients)
    so the agent doesn't reload them per patient.
    """

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        fielded_bm25: FieldedBM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        listwise: ListwiseReranker | None = None,
        hyde: HyDEGenerator | None = None,
        cascade: EligibilityCascade | None = None,
        criterion_extractor: CriterionExtractor | None = None,
        trials_by_id: Mapping[str, Trial] | None = None,
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.fielded_bm25 = fielded_bm25 or FieldedBM25Retriever()
        self.dense = dense or DenseRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.listwise = listwise or ListwiseReranker()
        self.hyde = hyde or HyDEGenerator()
        self.cascade = cascade or EligibilityCascade()
        self.criterion_extractor = criterion_extractor or CriterionExtractor()
        self.trials_by_id: Mapping[str, Trial] = trials_by_id or {}

    # ------------------------------------------------------------------
    # Tool: lexical search (BM25 with optional MeSH expansion)
    # ------------------------------------------------------------------
    def search_trials_lexical(
        self,
        query: str,
        mesh_terms: list[str] | None = None,
        top_k: int = 200,
        plan: Any | None = None,
        patient: PatientProfile | None = None,
    ) -> list[TrialCandidate]:
        settings = get_settings()
        if settings.retrieval.bm25_mode == "fielded":
            return self.fielded_bm25.retrieve(
                query,
                mesh_terms=mesh_terms,
                plan=plan,
                patient=patient,
                k=top_k,
            )
        if mesh_terms:
            query = query + " " + " ".join(mesh_terms)
        return self.bm25.retrieve(query, k=top_k)

    # ------------------------------------------------------------------
    # Tool: dense search (MedCPT)
    # ------------------------------------------------------------------
    async def search_trials_dense(
        self,
        patient_text: str,
        top_k: int = 200,
        use_hyde: bool = True,
        filters: dict | None = None,
    ) -> list[TrialCandidate]:
        if use_hyde:
            vec = await self.hyde.compose_query_embedding(patient_text, self.dense)
            from qdrant_client.http.models import (
                FieldCondition,
                Filter,
                MatchAny,
                MatchValue,
            )

            client = self.dense._client_()
            qfilter = None
            if filters:
                must = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        must.append(FieldCondition(key=key, match=MatchAny(any=value)))
                    else:
                        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
                qfilter = Filter(must=must) if must else None
            # qdrant-client >= 1.13 removed .search(); use .query_points().
            response = client.query_points(
                collection_name=self.dense.collection,
                query=vec,
                limit=top_k,
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
        return self.dense.retrieve(patient_text, k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Tool: pointwise rerank
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: list[TrialCandidate],
        top_k: int = 50,
    ) -> list[TrialCandidate]:
        def get_text(nct: str) -> str:
            t = self.trials_by_id.get(nct)
            if t is None:
                return ""
            return "\n".join(
                p
                for p in [
                    t.title,
                    t.official_title or "",
                    " ".join(t.conditions),
                    " ".join(t.keywords),
                    " ".join(t.interventions),
                    t.brief_summary,
                    t.detailed_description or "",
                    t.eligibility.inclusion_text,
                ]
                if p
            )

        return self.reranker.rerank(query, candidates, get_text=get_text, top_k=top_k)

    # ------------------------------------------------------------------
    # Tool: listwise rerank
    # ------------------------------------------------------------------
    def listwise_rerank(
        self,
        query: str,
        candidates: list[TrialCandidate],
        top_k: int = 30,
    ) -> list[TrialCandidate]:
        def get_text(nct: str) -> str:
            t = self.trials_by_id.get(nct)
            if t is None:
                return ""
            return "\n".join(
                p
                for p in [
                    t.title,
                    t.official_title or "",
                    " ".join(t.conditions),
                    " ".join(t.interventions),
                    t.brief_summary,
                ]
                if p
            )

        return self.listwise.rerank(query, candidates, get_text=get_text, top_k=top_k)

    # ------------------------------------------------------------------
    # Tool: hard filters
    # ------------------------------------------------------------------
    def apply_filters(
        self,
        candidates: list[TrialCandidate],
        patient: PatientProfile,
        allowed_statuses: list[str] | None = None,
        filter_status: bool = True,
    ) -> list[TrialCandidate]:
        return apply_hard_filters(
            candidates,
            self.trials_by_id,
            patient,
            allowed_statuses=allowed_statuses,
            filter_status=filter_status,
        )

    # ------------------------------------------------------------------
    # Tool: extract criteria from a trial
    # ------------------------------------------------------------------
    async def extract_criteria(
        self,
        trial: Trial,
        max_criteria: int = 0,
        patient: PatientProfile | None = None,
    ) -> list[Criterion]:
        return await self.criterion_extractor.extract(
            trial,
            max_criteria=max_criteria,
            patient=patient,
            use_triage=get_settings().runner.use_criterion_triage,
            use_section_header_policy=get_settings().runner.use_section_header_policy,
        )

    async def extract_criteria_with_diagnostics(
        self,
        trial: Trial,
        max_criteria: int = 0,
        patient: PatientProfile | None = None,
    ):
        return await self.criterion_extractor.extract_with_diagnostics(
            trial,
            max_criteria=max_criteria,
            patient=patient,
            use_triage=get_settings().runner.use_criterion_triage,
            use_section_header_policy=get_settings().runner.use_section_header_policy,
        )

    # ------------------------------------------------------------------
    # Tool: evaluate eligibility for a list of criteria
    # ------------------------------------------------------------------
    async def evaluate_eligibility(
        self,
        criteria: list[Criterion],
        patient: PatientProfile,
        concurrency: int = 8,
    ) -> list[CriterionEval]:
        return await self.cascade.evaluate_many(criteria, patient, concurrency=concurrency)

    # ------------------------------------------------------------------
    # Tool: get a Trial object by id
    # ------------------------------------------------------------------
    def get_trial(self, nct_id: str) -> Trial | None:
        return self.trials_by_id.get(nct_id)
