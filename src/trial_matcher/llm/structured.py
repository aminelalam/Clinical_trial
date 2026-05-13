"""Pydantic-validated structured output helper."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..cache import cache_key, get_cache
from ..config import get_settings
from ..logging import logger
from .client import EmptyLLMResponseError, LLMResponse, UnifiedLLM

T = TypeVar("T", bound=BaseModel)


_JSON_INSTRUCTION = (
    "\n\nReturn ONLY one RFC 8259 valid JSON object that strictly matches the "
    "schema described above. Use double-quoted strings, comma separators, and "
    "escape any embedded quotes inside string values. Do not include prose, "
    "code fences, comments, or commentary."
)


class StructuredOutputError(RuntimeError):
    """Raised after all structured-output attempts fail."""


@dataclass
class _Attempt:
    response_format: dict | None
    max_tokens: int
    prompt: str


async def structured_complete(
    llm: UnifiedLLM,
    *,
    prompt: str,
    response_model: type[T],
    system: str | None = None,
    model: str = "mini",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 1,
    task_name: str | None = None,
) -> T:
    """Generate a JSON response and parse it as ``response_model``.

    First attempts provider JSON mode. Retries switch to prompt-only JSON
    instructions and, for reasoning models, a larger visible+reasoning token
    budget. The structured cache is populated only after Pydantic validation.
    """
    s = get_settings()
    task = task_name or response_model.__name__
    model_schema = response_model.model_json_schema()
    primary = getattr(llm, "primary", "unknown")
    identity_fn = getattr(llm, "_cache_model_identity", None)
    model_identity = identity_fn(primary, model) if callable(identity_fn) else model
    cache_parts = (
        "structured_v4_task_segmented_json_instruction",
        primary,
        model,
        model_identity,
        s.llm.azure_api_version if primary == "azure" else "",
        s.llm.structured_reasoning_effort,
        task,
        prompt,
        system,
        temperature,
        max_tokens,
        max_retries,
        model_schema,
    )
    ckey = cache_key("llm_structured", *cache_parts)
    cdb = get_cache("llm_structured")
    if ckey in cdb:
        try:
            return response_model.model_validate(cdb[ckey])
        except Exception:
            try:
                del cdb[ckey]
            except Exception:  # pragma: no cover
                pass

    last_err: Exception | None = None
    last_resp: LLMResponse | None = None

    for attempt in range(max_retries + 1):
        attempted = _build_attempt(
            llm=llm,
            model=model,
            base_prompt=prompt,
            last_err=last_err,
            attempt=attempt,
            max_tokens=max_tokens,
        )
        try:
            resp = await _achat_for_structured(
                llm,
                attempted.prompt,
                system=system,
                model=model,
                temperature=temperature,
                max_tokens=attempted.max_tokens,
                response_format=attempted.response_format,
            )
            last_resp = resp
            data = _extract_json(resp.text)
            parsed = response_model.model_validate(data)
            cdb.set(ckey, parsed.model_dump(mode="json"))
            return parsed
        except (EmptyLLMResponseError, ValidationError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            if isinstance(e, EmptyLLMResponseError):
                last_resp = e.response
            logger.warning(
                "Structured output failed "
                f"(task={task}, attempt {attempt + 1}/{max_retries + 1}, "
                f"mode={'json_object' if attempted.response_format else 'plain_json'}, "
                f"max_tokens={attempted.max_tokens}): {e}"
            )
    if last_err is None:  # pragma: no cover
        raise RuntimeError("structured_complete: unreachable")
    details = ""
    if last_resp is not None:
        details = (
            f" provider={last_resp.provider} model={last_resp.model}"
            f" deployment={last_resp.deployment or last_resp.model}"
            f" finish_reason={last_resp.finish_reason!r}"
            f" completion_tokens={last_resp.completion_tokens}"
            f" reasoning_tokens={last_resp.reasoning_tokens}"
        )
    raise StructuredOutputError(
        f"Structured output failed for {response_model.__name__} task={task} after "
        f"{max_retries + 1} attempts:{details}; last_error={last_err}"
    ) from last_err


def _build_attempt(
    *,
    llm: UnifiedLLM,
    model: str,
    base_prompt: str,
    last_err: Exception | None,
    attempt: int,
    max_tokens: int,
) -> _Attempt:
    prompt = base_prompt + _JSON_INSTRUCTION
    if last_err is not None:
        prompt += (
            f"\n\nYour previous output failed validation with: {last_err}.\n"
            "Return a corrected RFC 8259 JSON object. Check for missing commas, "
            "unescaped quotes, trailing prose, and invalid enum values. Output JSON only."
        )
    token_budget = max_tokens
    is_reasoning_fn = getattr(llm, "is_reasoning_model", None)
    is_reasoning = is_reasoning_fn(model) if callable(is_reasoning_fn) else False
    response_format = (
        None
        if (attempt > 0 or is_reasoning)
        else {"type": "json_object"}
    )
    if attempt > 0:
        token_budget = max(max_tokens * 2, max_tokens + 512)
    return _Attempt(response_format=response_format, max_tokens=token_budget, prompt=prompt)


async def _achat_for_structured(
    llm: UnifiedLLM,
    prompt: str,
    *,
    system: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
) -> LLMResponse:
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    resp: Any = await llm.achat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        cache=False,
    )
    if isinstance(resp, LLMResponse):
        return resp
    text = str(resp or "")
    if not text.strip():
        raise EmptyLLMResponseError(
            LLMResponse(text=text, provider=getattr(llm, "primary", "azure"), model=model)
        )
    return LLMResponse(text=text, provider=getattr(llm, "primary", "azure"), model=model)


def _extract_json(text: str) -> dict:
    """Try hard to extract a JSON object from possibly noisy LLM output."""
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith(("json", "JSON")):
            text = text[4:].lstrip()
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    # Try direct parse
    try:
        return _loads_json_lenient(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return _loads_json_lenient(text[start : end + 1])
    raise ValueError(f"Could not extract JSON from output: {text[:200]!r}")


_INVALID_JSON_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters that appear inside JSON strings."""
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ord(ch) < 32:
            if ch == "\n":
                out.append(r"\n")
            elif ch == "\r":
                out.append(r"\r")
            elif ch == "\t":
                out.append(r"\t")
            else:
                out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)


def _loads_json_lenient(text: str) -> dict:
    """Load JSON, repairing narrow classes of common LLM JSON mistakes.

    The repair is deliberately narrow. JSON permits only a small set of escape
    sequences, while clinical strings sometimes contain copied units such as
    ``\\_``/``\\>`` or raw newlines inside a string value. We repair only those
    cases and keep otherwise strict parsing.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        candidates = [
            _INVALID_JSON_BACKSLASH.sub(r"\\\\", text),
            _escape_control_chars_in_strings(text),
        ]
        candidates.append(_escape_control_chars_in_strings(candidates[0]))

        seen = {text}
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise first_error
