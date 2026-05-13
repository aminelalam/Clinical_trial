"""Shared runtime construction for CLI and FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .agent.graph import build_agent
from .agent.tools import AgentTools
from .config import get_settings
from .eligibility.cascade import EligibilityCascade
from .ingestion.lazy_corpus import LazyTrialCorpus
from .ingestion.mesh_loader import load_mesh_index
from .llm.few_shot import FewShotBank
from .logging import logger
from .models.trial import Trial
from .nlp.mesh_normalizer import MeSHNormalizer
from .retrieval.dense import DenseRetriever


@dataclass
class AgentRuntime:
    agent: Any
    tools: AgentTools
    trials_by_id: Mapping[str, Trial]
    mesh_normalizer: MeSHNormalizer | None
    corpus_loaded: bool
    corpus_size: int
    mesh_loaded: bool
    mesh_concepts: int
    few_shot_bank_size: int
    qdrant_mode: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_collection_exists: bool
    qdrant_points_count: int


def build_agent_runtime(
    *,
    use_few_shot: bool = True,
    few_shot_dir: Path = Path("banco_few_shot"),
) -> AgentRuntime:
    """Build the long-lived agent runtime once and reuse it.

    This keeps the CLI and FastAPI service aligned: both use the same lazy
    corpus, same cascade, same few-shot bank, and same retrieval tooling.
    """
    s = get_settings()

    bank = None
    if use_few_shot and few_shot_dir.exists():
        bank = FewShotBank.from_jsonl_dir(few_shot_dir)
        if bank.examples:
            logger.info(f"Loaded few-shot bank: {len(bank.examples)} examples from {few_shot_dir}")
            bank.index()  # Uses lightweight all-MiniLM-L6-v2 by default
        else:
            bank = None
            logger.warning(f"few-shot dir {few_shot_dir} present but empty; running without")
    else:
        logger.info("Few-shot bank disabled (flag or missing dir)")

    cascade = EligibilityCascade(
        use_verifier=s.runner.use_verifier,
        use_self_consistency=s.runner.use_self_consistency,
        few_shot_bank=bank,
    )

    ctgov_dir = Path(s.paths.ctgov_dir)
    if ctgov_dir.exists():
        trials_by_id: Mapping[str, Trial] = LazyTrialCorpus(ctgov_dir)
        corpus_loaded = True
        logger.info(f"Trial corpus: {len(trials_by_id)} trials indexed")
    else:
        trials_by_id = {}
        corpus_loaded = False
        logger.warning(f"CTGov directory not found: {ctgov_dir}; hard filters will mark corpus misses")

    # MeSH normalizer — used by the planning node to expand the patient's
    # diagnosis with canonical synonyms. Failure is non-fatal: the rest of
    # the pipeline runs without query expansion.
    mesh_normalizer: MeSHNormalizer | None = None
    mesh_concepts = 0
    mesh_loaded = False
    mesh_dir = Path(s.paths.mesh_dir)
    if mesh_dir.exists() and any(mesh_dir.glob("desc*.xml")):
        try:
            mesh_index = load_mesh_index(mesh_dir)
            mesh_normalizer = MeSHNormalizer(mesh_index)
            mesh_concepts = len(mesh_index.concepts)
            mesh_loaded = mesh_concepts > 0
            logger.info(
                f"MeSH normalizer ready: {mesh_concepts} concepts, "
                f"{len(mesh_index.by_synonym)} synonyms indexed"
            )
        except Exception as e:
            logger.warning(f"MeSH index could not be loaded: {e!r}; continuing without normalisation")
    else:
        logger.info(f"MeSH directory not found or empty: {mesh_dir}; skipping normalisation")

    dense = DenseRetriever()
    qdrant_status: dict[str, Any] = {
        "mode": dense.qdrant_mode(),
        "exists": False,
        "points_count": 0,
    }
    try:
        qdrant_status = dense.collection_status()
        logger.info(
            "Qdrant: "
            f"mode={qdrant_status.get('mode')} "
            f"collection={s.qdrant.collection} "
            f"exists={qdrant_status.get('exists')} "
            f"points={qdrant_status.get('points_count', 0)}"
        )
    except Exception as e:
        logger.warning(f"Qdrant preflight failed: {e!r}")

    tools = AgentTools(cascade=cascade, dense=dense, trials_by_id=trials_by_id)
    agent = build_agent(tools=tools, mesh_normalizer=mesh_normalizer)
    return AgentRuntime(
        agent=agent,
        tools=tools,
        trials_by_id=trials_by_id,
        mesh_normalizer=mesh_normalizer,
        corpus_loaded=corpus_loaded,
        corpus_size=len(trials_by_id),
        mesh_loaded=mesh_loaded,
        mesh_concepts=mesh_concepts,
        few_shot_bank_size=len(bank.examples) if bank else 0,
        qdrant_mode=str(qdrant_status.get("mode") or dense.qdrant_mode()),
        qdrant_url=s.qdrant.url,
        qdrant_collection=s.qdrant.collection,
        qdrant_collection_exists=bool(qdrant_status.get("exists", False)),
        qdrant_points_count=int(qdrant_status.get("points_count", 0) or 0),
    )
