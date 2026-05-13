"""AgentState — the LangGraph shared state. Contract between every node.

Two coexisting representations:

- ``AgentStateDict`` is a ``TypedDict`` used as the graph's runtime state.
  Nodes return partial dict updates; LangGraph merges them per-channel using
  the reducers declared via ``Annotated``. This is the only shape that
  supports paralel fan-out (retrieve_lexical || retrieve_dense) and stable
  loop-control without spurious ``INVALID_CONCURRENT_GRAPH_UPDATE`` errors.

- ``AgentState`` is the original Pydantic model. We keep it because the
  runner serialises the final dict back into it (model_validate) for output
  rendering, and unit tests still rely on the model for fixtures and helper
  methods. The TypedDict mirrors its fields one-for-one.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from .criterion import Criterion
from .critique import Critique
from .dossier import TrialDossier
from .eligibility import CriterionEval, TrialEval
from .patient import PatientProfile
from .question import ClinicalQuestion
from .ranking import JudgedTrial, RankedTrial
from .search_plan import SearchPlan


class TrialCandidate(BaseModel):
    """Lightweight record passed between retrieval nodes."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str
    score: float = 0.0
    source: str = Field(default="", description="bm25|dense|fused|reranked|listwise")
    rank: int = 0

    # Cached pieces of the trial (avoid re-loading from disk repeatedly)
    title: str | None = None
    snippet: str | None = None
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)

    # Filter state
    hard_excluded: bool = False
    excluded_reason: str | None = None


class NodeTiming(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node: str
    seconds: float


def _add_int(left: int, right: int) -> int:
    """Reducer for scalar counters incremented from multiple nodes.

    LangGraph passes the previous channel value as ``left`` and the new
    update as ``right``. Treating both as additive keeps a true running
    total even when the same channel is touched by parallel nodes.
    """
    return int(left or 0) + int(right or 0)


class AgentStateDict(TypedDict, total=False):
    """LangGraph-native runtime state.

    Nodes return ``dict[str, Any]`` updates that LangGraph merges using the
    reducers below. Lists with ``operator.add`` accumulate; scalar counters
    use ``_add_int`` so increments don't get lost in parallel branches.
    """

    # Inputs (set once, never mutated by nodes)
    patient_raw: str
    run_id: str

    # Stage 1 — patient understanding
    patient_profile: PatientProfile | None

    # Stage 2 — planning
    search_plan: SearchPlan | None

    # Stage 3 — retrieval (each candidate slot is owned by exactly one node,
    # so LastValue suffices; the lists are written, not appended).
    bm25_candidates: list[TrialCandidate]
    dense_candidates: list[TrialCandidate]
    fused_candidates: list[TrialCandidate]
    reranked_candidates: list[TrialCandidate]
    listwise_candidates: list[TrialCandidate]
    final_candidates: list[TrialCandidate]

    # Stage 4 — eligibility
    extracted_criteria: dict[str, list[Criterion]]
    criterion_selection_diagnostics: dict[str, dict[str, Any]]
    candidate_selection_diagnostics: dict[str, Any]
    entity_negation_diagnostics: dict[str, Any]
    criterion_evidence_diagnostics: dict[str, Any]
    irrelevance_diagnostics: dict[str, dict[str, Any]]
    criterion_evals: dict[str, list[CriterionEval]]
    trial_evals: dict[str, TrialEval]

    # Stage 5 — ranking
    ranked_trials: list[RankedTrial]
    judged_top10: list[JudgedTrial]
    critique: Critique | None
    hard_excluded_fill_count: int
    retrieval_tail_fill_count: int
    hard_excluded_fill_reasons: dict[str, int]
    hard_excluded_fill_skipped_corpus_miss: int

    # Stage 6 — output
    questions: list[ClinicalQuestion]
    dossiers: list[TrialDossier]

    # Agent control — counters use additive reducer so concurrent branches
    # cannot lose increments.
    retrieval_attempts: Annotated[int, _add_int]
    max_retrieval_attempts: int
    needs_re_retrieval: bool
    re_retrieval_triggered: bool
    relaxed_plan_used: bool
    critique_iterations: Annotated[int, _add_int]
    max_critique_iterations: int

    # Accumulating diagnostics — every node appends, never overwrites.
    critique_notes: Annotated[list[str], operator.add]
    node_timings: Annotated[list[NodeTiming], operator.add]
    errors: Annotated[list[str], operator.add]

    # Counters (additive)
    llm_calls: Annotated[int, _add_int]
    cache_hits: Annotated[int, _add_int]


def initial_state(
    patient_raw: str,
    run_id: str,
    *,
    max_retrieval_attempts: int = 2,
    max_critique_iterations: int = 1,
) -> AgentStateDict:
    """Build a fresh state dict with all required slots populated.

    LangGraph will not synthesise default values for TypedDict channels, so
    we explicitly seed every list/dict/scalar to its zero value.
    """
    return AgentStateDict(  # type: ignore[typeddict-item]
        patient_raw=patient_raw,
        run_id=run_id,
        patient_profile=None,
        search_plan=None,
        bm25_candidates=[],
        dense_candidates=[],
        fused_candidates=[],
        reranked_candidates=[],
        listwise_candidates=[],
        final_candidates=[],
        extracted_criteria={},
        criterion_selection_diagnostics={},
        candidate_selection_diagnostics={},
        entity_negation_diagnostics={},
        criterion_evidence_diagnostics={},
        irrelevance_diagnostics={},
        criterion_evals={},
        trial_evals={},
        ranked_trials=[],
        judged_top10=[],
        critique=None,
        hard_excluded_fill_count=0,
        retrieval_tail_fill_count=0,
        hard_excluded_fill_reasons={},
        hard_excluded_fill_skipped_corpus_miss=0,
        questions=[],
        dossiers=[],
        retrieval_attempts=0,
        max_retrieval_attempts=max_retrieval_attempts,
        needs_re_retrieval=False,
        re_retrieval_triggered=False,
        relaxed_plan_used=False,
        critique_iterations=0,
        max_critique_iterations=max_critique_iterations,
        critique_notes=[],
        node_timings=[],
        errors=[],
        llm_calls=0,
        cache_hits=0,
    )


class AgentState(BaseModel):
    """Pydantic mirror of ``AgentStateDict`` used for serialisation only.

    Kept for backward compatibility with the runner output, the API
    response, and existing unit-test fixtures. Nodes no longer use this
    class directly: see ``AgentStateDict`` for the LangGraph runtime.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Inputs
    patient_raw: str
    run_id: str

    # Stage 1 — patient understanding
    patient_profile: PatientProfile | None = None

    # Stage 2 — planning (agentic)
    search_plan: SearchPlan | None = None

    # Stage 3 — retrieval
    bm25_candidates: list[TrialCandidate] = Field(default_factory=list)
    dense_candidates: list[TrialCandidate] = Field(default_factory=list)
    fused_candidates: list[TrialCandidate] = Field(default_factory=list)
    reranked_candidates: list[TrialCandidate] = Field(default_factory=list)
    listwise_candidates: list[TrialCandidate] = Field(default_factory=list)
    final_candidates: list[TrialCandidate] = Field(default_factory=list)

    # Stage 4 — eligibility
    extracted_criteria: dict[str, list[Any]] = Field(default_factory=dict)
    criterion_selection_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    candidate_selection_diagnostics: dict[str, Any] = Field(default_factory=dict)
    entity_negation_diagnostics: dict[str, Any] = Field(default_factory=dict)
    criterion_evidence_diagnostics: dict[str, Any] = Field(default_factory=dict)
    irrelevance_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    criterion_evals: dict[str, list[CriterionEval]] = Field(default_factory=dict)
    trial_evals: dict[str, TrialEval] = Field(default_factory=dict)

    # Stage 5 — ranking
    ranked_trials: list[RankedTrial] = Field(default_factory=list)
    judged_top10: list[JudgedTrial] = Field(default_factory=list)
    critique: Critique | None = None
    hard_excluded_fill_count: int = 0
    retrieval_tail_fill_count: int = 0
    hard_excluded_fill_reasons: dict[str, int] = Field(default_factory=dict)
    hard_excluded_fill_skipped_corpus_miss: int = 0

    # Stage 6 — output
    questions: list[ClinicalQuestion] = Field(default_factory=list)
    dossiers: list[TrialDossier] = Field(default_factory=list)

    # Agent control
    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 2
    needs_re_retrieval: bool = False
    re_retrieval_triggered: bool = False
    relaxed_plan_used: bool = False
    critique_iterations: int = 0
    max_critique_iterations: int = 1
    critique_notes: list[str] = Field(default_factory=list)

    # Telemetry
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_timings: list[NodeTiming] = Field(default_factory=list)
    llm_calls: int = 0
    cache_hits: int = 0
    errors: list[str] = Field(default_factory=list)

    def add_timing(self, node: str, seconds: float) -> None:
        self.node_timings.append(NodeTiming(node=node, seconds=seconds))
