"""LLM client routing policy."""

from __future__ import annotations


def test_provider_fallbacks_disabled_by_default():
    from trial_matcher.llm.client import UnifiedLLM

    llm = UnifiedLLM(primary="azure")

    assert llm.fallbacks == []


def test_provider_fallbacks_filter_unconfigured_providers(monkeypatch):
    from trial_matcher.config import get_settings
    from trial_matcher.llm.client import UnifiedLLM

    monkeypatch.setenv("TRIAL_MATCHER__LLM__ENABLE_PROVIDER_FALLBACKS", "true")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    llm = UnifiedLLM(primary="azure")

    assert "groq" not in llm.fallbacks
    assert "ollama" not in llm.fallbacks
