"""Centralized configuration. All env vars are namespaced TRIAL_MATCHER__<SECTION>__<KEY>."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["azure", "groq", "ollama", "openai"]
QdrantMode = Literal["server", "embedded", "auto"]
RunnerMode = Literal["benchmark", "clinical_active"]
BenchmarkManifestPolicy = Literal["off", "warn", "require"]
BM25Mode = Literal["single", "fielded"]
BenchmarkCandidateSelectionPolicy = Literal["top_score", "diverse_top10"]
BenchmarkEntityRerankPolicy = Literal["off", "audit", "rerank_final"]
BenchmarkCriterionEvidencePolicy = Literal["off", "score_adjust"]


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAL_MATCHER__LLM__", extra="ignore")

    default_provider: ProviderName = "azure"

    # Azure OpenAI (read directly from standard names too)
    azure_api_key: str = Field(default="", validation_alias="AZURE_OPENAI_API_KEY")
    azure_endpoint: str = Field(default="", validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = Field(
        default="2024-08-01-preview", validation_alias="AZURE_OPENAI_API_VERSION"
    )
    azure_deployment_mini: str = Field(
        default="gpt-4o-mini", validation_alias="AZURE_OPENAI_DEPLOYMENT_MINI"
    )
    azure_deployment_large: str = Field(
        default="gpt-4o", validation_alias="AZURE_OPENAI_DEPLOYMENT_LARGE"
    )

    # Groq fallback
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:32b-instruct", validation_alias="OLLAMA_MODEL")

    # Plain OpenAI
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Inference behavior
    temperature_extract: float = 0.0
    temperature_judge: float = 0.2
    temperature_sc: float = 0.5
    sc_k_samples: int = 3
    sc_confidence_threshold: float = 0.7
    max_retries: int = 3
    timeout_seconds: int = 60
    enable_provider_fallbacks: bool = False
    enable_ollama_fallback: bool = False
    mini_is_reasoning: bool | None = None
    large_is_reasoning: bool | None = None
    structured_reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh"
    ] = "minimal"


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAL_MATCHER__RETRIEVAL__", extra="ignore")

    bm25_index_dir: Path | None = None
    bm25_mode: BM25Mode = "single"
    fielded_bm25_index_dir: Path | None = None
    fielded_bm25_per_field_k: int = 1000
    fielded_rerank_retrieval_blend: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 200
    dense_top_k: int = 200
    rrf_k: int = 60
    bm25_rrf_weight: float = 1.0
    dense_rrf_weight: float = 1.0
    fused_top_k: int = 100
    rerank_top_k: int = 50
    listwise_top_k: int = 30
    final_top_k: int = 30

    medcpt_query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    medcpt_article_encoder: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_cross_encoder: str = "ncbi/MedCPT-Cross-Encoder"
    listwise_model: str = "castorini/rank_zephyr_7b_v1_full"


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mode: QdrantMode = Field(default="server", validation_alias="QDRANT_MODE")
    url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")
    collection: str = Field(default="ctgov_trials_medcpt", validation_alias="QDRANT_COLLECTION")
    path: str = Field(default="", validation_alias="QDRANT_PATH")
    timeout_seconds: int = Field(default=120, validation_alias="QDRANT_TIMEOUT_SECONDS")
    on_disk_vectors: bool = Field(default=True, validation_alias="QDRANT_ON_DISK_VECTORS")
    on_disk_hnsw: bool = Field(default=True, validation_alias="QDRANT_ON_DISK_HNSW")
    on_disk_payload: bool = Field(default=True, validation_alias="QDRANT_ON_DISK_PAYLOAD")


class PathsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAL_MATCHER__PATHS__", extra="ignore")

    data_dir: Path = Path("data")
    ctgov_dir: Path = Path("data/ctgov_snapshot")
    trec_dir: Path = Path("data/trec_ct")
    mesh_dir: Path = Path("data/mesh")
    indices_dir: Path = Path("data/indices")
    cache_dir: Path = Path(".cache")

    def ensure(self) -> None:
        for p in (
            self.data_dir,
            self.ctgov_dir,
            self.trec_dir,
            self.mesh_dir,
            self.indices_dir,
            self.cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


class RunnerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAL_MATCHER__RUNNER__", extra="ignore")

    mode: RunnerMode = "benchmark"
    concurrency: int = 8
    use_dense_retrieval: bool = True
    use_hyde: bool = True
    use_listwise: bool = True
    use_verifier: bool = True
    use_few_shot: bool = True
    use_self_consistency: bool = True
    use_llm_judge: bool = True
    use_self_critique: bool = True
    use_questions: bool = True
    use_dossiers: bool = True
    use_criterion_triage: bool = False
    use_section_header_policy: bool = False
    max_retrieval_attempts: int = 2
    max_critique_iterations: int = 1  # cap for the rerank-after-critique loop
    topic_timeout_seconds: int = 0  # 0 = no per-topic timeout
    output_top_k: int = 20
    max_trials_per_topic: int = 0  # 0 = no cap
    max_criteria_per_trial: int = 0  # 0 = no cap
    benchmark_candidate_selection_policy: BenchmarkCandidateSelectionPolicy = "top_score"
    benchmark_diverse_keep_top: int = 9
    benchmark_diverse_select_total: int = 10
    benchmark_entity_rerank_policy: BenchmarkEntityRerankPolicy = "off"
    benchmark_entity_rerank_weight: float = 0.09
    benchmark_entity_protect_top: int = 3
    benchmark_criterion_evidence_policy: BenchmarkCriterionEvidencePolicy = "off"
    benchmark_criterion_evidence_weight: float = 0.50
    benchmark_soft_veto: bool = False
    benchmark_index_manifest_policy: BenchmarkManifestPolicy = "warn"
    use_hard_excluded_fill: bool = True
    use_retrieval_tail_fill: bool = False
    include_retrieval_traces: bool = False
    # Aggregator thresholds — configurable for calibration (B7).
    min_inclusion_fraction: float = 0.6
    max_nei_fraction: float = 0.6
    benchmark_min_inclusion_fraction: float = 0.1
    benchmark_max_nei_fraction: float = 1.0
    use_irrelevance_heuristic: bool = False
    use_multisignal_irrelevance_heuristic: bool = False
    irrelevant_min_nei_fraction: float = 0.8
    irrelevant_max_inclusion_met: int = 0
    irrelevant_max_retrieval_prior: float = 0.5
    irrelevant_min_signal_count: int = 3


class TelemetrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAL_MATCHER__TELEMETRY__", extra="ignore")

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


class Settings(BaseSettings):
    """Root settings object — imported wherever config is needed."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    Calls ``load_dotenv`` first so that the ``.env`` values are available
    as real environment variables — necessary because pydantic-settings
    sub-models (``LLMSettings``, etc.) are constructed independently via
    ``default_factory`` and do NOT inherit ``env_file`` from the parent.
    Existing process environment variables win over ``.env`` values so
    reproducible experiment scripts can override local defaults.
    """
    from dotenv import load_dotenv

    load_dotenv(override=False)
    return Settings()
