"""Structured LLM helper strategy."""

from __future__ import annotations


class _ReasoningLLM:
    def is_reasoning_model(self, model: str) -> bool:
        return True


class _PlainLLM:
    def is_reasoning_model(self, model: str) -> bool:
        return False


def test_reasoning_model_skips_json_mode_on_first_attempt():
    from trial_matcher.llm.structured import _build_attempt

    attempt = _build_attempt(
        llm=_ReasoningLLM(),
        model="mini",
        base_prompt="short prompt",
        last_err=None,
        attempt=0,
        max_tokens=3000,
    )

    assert attempt.response_format is None


def test_plain_model_uses_json_mode_even_for_long_prompt():
    from trial_matcher.llm.structured import _build_attempt

    attempt = _build_attempt(
        llm=_PlainLLM(),
        model="mini",
        base_prompt="x" * 6000,
        last_err=None,
        attempt=0,
        max_tokens=3000,
    )

    assert attempt.response_format == {"type": "json_object"}


def test_retry_attempt_raises_token_budget_for_plain_model():
    from trial_matcher.llm.structured import _build_attempt

    attempt = _build_attempt(
        llm=_PlainLLM(),
        model="mini",
        base_prompt="short prompt",
        last_err=ValueError("truncated JSON"),
        attempt=1,
        max_tokens=1500,
    )

    assert attempt.response_format is None
    assert attempt.max_tokens == 3000


def test_extract_json_repairs_invalid_backslash_escape():
    from trial_matcher.llm.structured import _extract_json

    data = _extract_json(r'{"evidence_quote": "ECOG \> 1 noted", "decision": "NEI"}')

    assert data["evidence_quote"] == r"ECOG \> 1 noted"


def test_extract_json_repairs_raw_control_character_inside_string():
    from trial_matcher.llm.structured import _extract_json

    data = _extract_json('{"question_text": "line one\nline two", "priority": "medium"}')

    assert data["question_text"] == "line one\nline two"


def test_criterion_classify_has_reasoning_safe_initial_budget():
    from trial_matcher.nlp.criterion_extractor import _CRITERION_CLASSIFY_MIN_TOKENS

    assert _CRITERION_CLASSIFY_MIN_TOKENS >= 4800
