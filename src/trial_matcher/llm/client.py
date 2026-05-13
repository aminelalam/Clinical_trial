"""Unified LLM client over Azure OpenAI / OpenAI / Groq / Ollama.

Design goals:
- Same call surface regardless of provider.
- Automatic retry with exponential backoff on transient failures (rate limits, 5xx).
- Optional fallback chain: disabled by default for reproducible evals; if
  explicitly enabled, try configured secondary providers after the primary.
- Disk-cached completions keyed by (provider, model, messages, temperature).
- Tracks usage stats in a thread-local counter for telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import ProviderName, get_settings
from ..logging import logger

# Models that do NOT support the 'temperature' parameter (reasoning models).
# When detected, we omit temperature from the API call.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5",)


def _is_reasoning_model(deployment: str) -> bool:
    """Return True if the deployment name matches a known reasoning model."""
    dl = deployment.lower()
    return any(dl.startswith(p) for p in _REASONING_MODEL_PREFIXES)


@dataclass
class LLMResponse:
    text: str
    provider: ProviderName
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    reasoning_tokens: int = 0
    raw_model: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class EmptyLLMResponseError(RuntimeError):
    """Raised when a provider returns no visible assistant content."""

    def __init__(self, response: LLMResponse):
        self.response = response
        super().__init__(
            "Empty LLM response "
            f"provider={response.provider} model={response.model} "
            f"deployment={response.deployment or response.model} "
            f"finish_reason={response.finish_reason!r} "
            f"completion_tokens={response.completion_tokens} "
            f"reasoning_tokens={response.reasoning_tokens} "
            f"api_version={response.api_version or 'n/a'}"
        )


class LLMContentFilterError(RuntimeError):
    """Raised when the configured provider blocks a prompt by safety policy."""


def _is_content_filter_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "content_filter" in text or "responsibleaipolicyviolation" in text


class _Counters:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class UnifiedLLM:
    """Provider-agnostic chat-completion client.

    Usage:
        llm = UnifiedLLM()
        response = await llm.acomplete("hello", model="mini")
    """

    def __init__(
        self,
        primary: ProviderName | None = None,
        fallbacks: list[ProviderName] | None = None,
    ):
        s = get_settings()
        self.primary: ProviderName = primary or s.llm.default_provider
        self.fallbacks: list[ProviderName] = (
            fallbacks
            if fallbacks is not None
            else (
                self._configured_fallbacks(self.primary)
                if s.llm.enable_provider_fallbacks
                else []
            )
        )
        self.counters = _Counters()
        self._azure_client = None
        self._openai_client = None
        self._groq_client = None
        self._http = None

    @staticmethod
    def _default_fallbacks(primary: ProviderName) -> list[ProviderName]:
        chain: dict[ProviderName, list[ProviderName]] = {
            "azure": ["groq", "ollama"],
            "openai": ["groq", "ollama"],
            "groq": ["azure", "ollama"],
            "ollama": ["groq", "azure"],
        }
        return chain.get(primary, [])

    @staticmethod
    def _provider_configured(provider: ProviderName) -> bool:
        s = get_settings()
        if provider == "azure":
            return bool(s.llm.azure_api_key and s.llm.azure_endpoint)
        if provider == "openai":
            return bool(s.llm.openai_api_key)
        if provider == "groq":
            return bool(s.llm.groq_api_key)
        if provider == "ollama":
            return bool(
                s.llm.enable_ollama_fallback
                and s.llm.ollama_base_url
                and s.llm.ollama_model
            )
        return False

    @classmethod
    def _configured_fallbacks(cls, primary: ProviderName) -> list[ProviderName]:
        return [
            provider
            for provider in cls._default_fallbacks(primary)
            if cls._provider_configured(provider)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: Literal["mini", "large"] | str = "mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict | None = None,
        cache: bool = True,
    ) -> str:
        """Run a chat completion. Returns the assistant text content."""
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        resp = await self.achat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            cache=cache,
        )
        return resp.text

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: Literal["mini", "large"] | str = "mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict | None = None,
        cache: bool = True,
    ) -> LLMResponse:
        if not cache:
            return await self._achat_uncached(
                messages, model, temperature, max_tokens, response_format
            )

        # Cache lookup keyed by (provider, model, messages, temp, max_tokens, response_format).
        from ..cache import cache_key, get_cache

        s = get_settings()
        key_parts = (
            self.primary,
            model,
            self._cache_model_identity(self.primary, model),
            s.llm.azure_api_version if self.primary == "azure" else "",
            s.llm.structured_reasoning_effort,
            messages,
            round(temperature, 4),
            max_tokens,
            response_format,
        )
        ckey = cache_key("llm", *key_parts)
        cdb = get_cache("llm")
        if ckey in cdb:
            self.counters.cache_hits += 1
            payload = cdb[ckey]
            if not str(payload.get("text", "")).strip():
                try:
                    del cdb[ckey]
                except Exception:  # pragma: no cover
                    pass
            else:
                return LLMResponse(**payload, cached=True)

        resp = await self._achat_uncached(
            messages, model, temperature, max_tokens, response_format
        )
        # Don't cache empty responses — reasoning models may produce empty
        # text on the first attempt but succeed on retry.
        if resp.text.strip():
            try:
                cdb.set(
                    ckey,
                    {
                        "text": resp.text,
                        "provider": resp.provider,
                        "model": resp.model,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "total_tokens": resp.total_tokens,
                        "finish_reason": resp.finish_reason,
                        "reasoning_tokens": resp.reasoning_tokens,
                        "raw_model": resp.raw_model,
                        "deployment": resp.deployment,
                        "api_version": resp.api_version,
                        "raw": {},
                    },
                )
            except Exception:  # pragma: no cover
                pass
        return resp

    # ------------------------------------------------------------------
    # Backend dispatch with fallback chain
    # ------------------------------------------------------------------
    async def _achat_uncached(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        chain: list[ProviderName] = [self.primary, *self.fallbacks]
        last_exc: Exception | None = None
        for provider in chain:
            try:
                self.counters.calls += 1
                resp = await self._call_provider(
                    provider, messages, model, temperature, max_tokens, response_format
                )
                self.counters.prompt_tokens += resp.prompt_tokens
                self.counters.completion_tokens += resp.completion_tokens
                return resp
            except EmptyLLMResponseError:
                raise
            except Exception as e:  # pragma: no cover - depends on network
                if _is_content_filter_error(e):
                    raise LLMContentFilterError(
                        f"Provider {provider} blocked the prompt by content policy: {e}"
                    ) from e
                last_exc = e
                if provider == chain[-1]:
                    logger.warning(
                        f"Provider {provider} failed ({type(e).__name__}: {e}); no fallback left"
                    )
                else:
                    logger.warning(
                        f"Provider {provider} failed ({type(e).__name__}: {e}); trying next"
                    )
        if last_exc:
            raise last_exc
        raise RuntimeError("No LLM provider configured")

    async def _call_provider(
        self,
        provider: ProviderName,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        if provider == "azure":
            return await self._azure(messages, model, temperature, max_tokens, response_format)
        if provider == "openai":
            return await self._openai(messages, model, temperature, max_tokens, response_format)
        if provider == "groq":
            return await self._groq(messages, model, temperature, max_tokens, response_format)
        if provider == "ollama":
            return await self._ollama(messages, model, temperature, max_tokens)
        raise ValueError(f"Unknown provider: {provider}")

    def is_reasoning_model(self, model: Literal["mini", "large"] | str = "mini") -> bool:
        """Return whether the primary provider/model should be treated as reasoning."""
        return self._is_reasoning_for(self.primary, model, self._cache_model_identity(self.primary, model))

    def _cache_model_identity(self, provider: ProviderName, model: str) -> str:
        s = get_settings()
        if provider == "azure":
            if model == "large":
                return s.llm.azure_deployment_large
            if model == "mini":
                return s.llm.azure_deployment_mini
            return model
        if provider == "openai":
            return "gpt-4o" if model == "large" else "gpt-4o-mini" if model == "mini" else model
        if provider == "groq":
            return s.llm.groq_model
        if provider == "ollama":
            return s.llm.ollama_model
        return model

    def _is_reasoning_for(self, provider: ProviderName, model: str, actual: str) -> bool:
        s = get_settings()
        if provider == "azure":
            configured: bool | None
            if model == "mini":
                configured = s.llm.mini_is_reasoning
            elif model == "large":
                configured = s.llm.large_is_reasoning
            else:
                configured = None
            if configured is not None:
                return configured
        return _is_reasoning_model(actual)

    @staticmethod
    def _reasoning_tokens(usage: Any) -> int:
        details = getattr(usage, "completion_tokens_details", None)
        return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0

    @staticmethod
    def _ensure_non_empty(resp: LLMResponse) -> LLMResponse:
        if not resp.text.strip():
            raise EmptyLLMResponseError(resp)
        return resp

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------
    async def _azure(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        s = get_settings()
        if not s.llm.azure_api_key or not s.llm.azure_endpoint:
            raise RuntimeError("Azure OpenAI not configured (AZURE_OPENAI_API_KEY/ENDPOINT)")
        if self._azure_client is None:
            from openai import AsyncAzureOpenAI

            self._azure_client = AsyncAzureOpenAI(
                api_key=s.llm.azure_api_key,
                azure_endpoint=s.llm.azure_endpoint,
                api_version=s.llm.azure_api_version,
                timeout=s.llm.timeout_seconds,
                max_retries=0,
            )
        deployment = (
            s.llm.azure_deployment_large if model == "large" else s.llm.azure_deployment_mini
        )
        is_reasoning = self._is_reasoning_for("azure", model, deployment)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(s.llm.max_retries),
            wait=wait_exponential(multiplier=1.5, min=1.0, max=20.0),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                kwargs: dict[str, Any] = {
                    "model": deployment,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                # Reasoning models (o1/o3/o4/gpt-5*) reject temperature.
                if not is_reasoning:
                    kwargs["temperature"] = temperature
                elif s.llm.structured_reasoning_effort:
                    kwargs["reasoning_effort"] = s.llm.structured_reasoning_effort
                if response_format:
                    kwargs["response_format"] = response_format
                resp = await self._azure_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        out = LLMResponse(
            text=choice.message.content or "",
            provider="azure",
            model=deployment,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning_tokens=self._reasoning_tokens(usage),
            raw_model=getattr(resp, "model", None),
            deployment=deployment,
            api_version=s.llm.azure_api_version,
        )
        return self._ensure_non_empty(out)

    # ------------------------------------------------------------------
    # OpenAI (plain)
    # ------------------------------------------------------------------
    async def _openai(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        s = get_settings()
        if not s.llm.openai_api_key:
            raise RuntimeError("OpenAI not configured")
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=s.llm.openai_api_key,
                timeout=s.llm.timeout_seconds,
                max_retries=0,
            )
        actual = "gpt-4o" if model == "large" else "gpt-4o-mini" if model == "mini" else model
        is_reasoning = self._is_reasoning_for("openai", model, actual)
        kwargs: dict[str, Any] = {
            "model": actual,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        # Reasoning models (o1/o3/o4/gpt-5*) reject temperature.
        if not is_reasoning:
            kwargs["temperature"] = temperature
        elif s.llm.structured_reasoning_effort:
            kwargs["reasoning_effort"] = s.llm.structured_reasoning_effort
        if response_format:
            kwargs["response_format"] = response_format
        resp = await self._openai_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        out = LLMResponse(
            text=choice.message.content or "",
            provider="openai",
            model=actual,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning_tokens=self._reasoning_tokens(usage),
            raw_model=getattr(resp, "model", None),
            deployment=actual,
        )
        return self._ensure_non_empty(out)

    # ------------------------------------------------------------------
    # Groq (free Llama 3.3 70B)
    # ------------------------------------------------------------------
    async def _groq(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        s = get_settings()
        if not s.llm.groq_api_key:
            raise RuntimeError("Groq not configured (GROQ_API_KEY missing)")
        if self._groq_client is None:
            from groq import AsyncGroq

            self._groq_client = AsyncGroq(api_key=s.llm.groq_api_key, timeout=s.llm.timeout_seconds)
        kwargs: dict[str, Any] = {
            "model": s.llm.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = await self._groq_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        out = LLMResponse(
            text=choice.message.content or "",
            provider="groq",
            model=s.llm.groq_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", None),
            raw_model=getattr(resp, "model", None),
            deployment=s.llm.groq_model,
        )
        return self._ensure_non_empty(out)

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------
    async def _ollama(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import httpx

        s = get_settings()
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=s.llm.timeout_seconds)
        actual = s.llm.ollama_model
        url = f"{s.llm.ollama_base_url}/api/chat"
        payload = {
            "model": actual,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        out = LLMResponse(
            text=text,
            provider="ollama",
            model=actual,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            total_tokens=0,
            deployment=actual,
        )
        return self._ensure_non_empty(out)
