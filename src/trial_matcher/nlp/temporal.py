"""Temporal constraint extraction from criterion text.

Recognizes patterns like:
- "within (the) (last|past|previous) X (days|weeks|months|years)"
- "X (days|weeks|months|years) (prior|ago|before)"
- "current(ly)", "ongoing", "ever", "no history of"
- ">= X days since"

Normalizes to a TemporalConstraint with `relation` and `days` (when applicable).
"""

from __future__ import annotations

import re
from typing import NamedTuple

from ..models.criterion import TemporalConstraint

UNIT_TO_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


class _Match(NamedTuple):
    pattern: re.Pattern
    relation: str
    has_quantity: bool


_PATTERNS: list[_Match] = [
    _Match(
        re.compile(
            r"within\s+(?:the\s+)?(?:past|previous|last\s+)?(\d+)\s+(day|week|month|year)s?",
            re.IGNORECASE,
        ),
        "within",
        True,
    ),
    _Match(
        re.compile(r"(\d+)\s+(day|week|month|year)s?\s+(?:prior|ago|before)", re.IGNORECASE),
        "within",
        True,
    ),
    _Match(
        re.compile(
            r"in\s+the\s+past\s+(\d+)\s+(day|week|month|year)s?", re.IGNORECASE
        ),
        "within",
        True,
    ),
    _Match(
        re.compile(
            r"(?:>=|≥|at\s+least)\s*(\d+)\s+(day|week|month|year)s?\s+(?:since|after)",
            re.IGNORECASE,
        ),
        "after",
        True,
    ),
    _Match(re.compile(r"current(?:ly)?\b", re.IGNORECASE), "current", False),
    _Match(re.compile(r"\bongoing\b", re.IGNORECASE), "ongoing", False),
    _Match(re.compile(r"\b(?:no\s+)?history\s+of\b", re.IGNORECASE), "ever", False),
    _Match(re.compile(r"\bnever\b", re.IGNORECASE), "never", False),
]


class TemporalExtractor:
    """Stateless extractor — instantiated for symmetry with NegationAnnotator."""

    def extract(self, text: str) -> TemporalConstraint | None:
        if not text:
            return None
        for m in _PATTERNS:
            match = m.pattern.search(text)
            if not match:
                continue
            if m.has_quantity:
                n = int(match.group(1))
                unit = match.group(2).lower()
                days = n * UNIT_TO_DAYS.get(unit, 0)
                return TemporalConstraint(relation=m.relation, days=days, raw=match.group(0))
            return TemporalConstraint(relation=m.relation, days=None, raw=match.group(0))
        return None

    def annotate(self, text: str) -> str:
        """Return text with [TEMPORAL: ...] markers around recognized phrases."""
        out = text
        for m in _PATTERNS:
            out = m.pattern.sub(lambda mm: f"[TEMPORAL: {mm.group(0)}]", out)
        return out


_default_extractor = TemporalExtractor()


def extract_temporal(text: str) -> TemporalConstraint | None:
    return _default_extractor.extract(text)


def annotate_temporal(text: str) -> str:
    return _default_extractor.annotate(text)
