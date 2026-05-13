"""Negation detection for eligibility criteria.

Uses scispaCy + NegEx as the heavy hammer; falls back to a small rule list when
scispaCy isn't available so the rest of the system still works in dev environments
without the model installed.
"""

from __future__ import annotations

import re
from typing import Any

# Lightweight rule-based fallback patterns
_NEG_PATTERNS = [
    r"\b(?:no|none|never|without|absent|denies|negative for|free of|excluding)\b",
    r"\b(?:no\s+history\s+of|no\s+evidence\s+of|not\s+(?:had|present|known))\b",
    r"\bcannot\b",
    r"\bunable\s+to\b",
]
_NEG_REGEX = re.compile("|".join(_NEG_PATTERNS), re.IGNORECASE)


class NegationAnnotator:
    """Annotates clinical text with explicit [NEGATED:...] markers around negated entities.

    Tries scispaCy + NegEx first; falls back to regex bracketing when unavailable.
    """

    def __init__(self) -> None:
        self._nlp: Any | None = None
        self._unavailable = False
        self._init_attempted = False

    def _init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import spacy
            from negspacy.negation import Negex  # noqa: F401  (registered as a pipe)

            try:
                nlp = spacy.load("en_core_sci_sm")
            except OSError:
                # Model not installed; mark unavailable, fallback regex still works
                self._unavailable = True
                return
            if "negex" not in nlp.pipe_names:
                nlp.add_pipe("negex", last=True)
            self._nlp = nlp
        except Exception:
            self._unavailable = True

    def annotate(self, text: str) -> str:
        """Return text with explicit negation tags inserted around negated entities."""
        self._init()
        if self._unavailable or self._nlp is None:
            return self._regex_annotate(text)
        try:
            doc = self._nlp(text)
        except Exception:
            return self._regex_annotate(text)
        out = text
        for ent in reversed(list(doc.ents)):
            if getattr(ent._, "negex", False):
                out = out[: ent.start_char] + f"[NEGATED: {ent.text}]" + out[ent.end_char :]
        return out

    @staticmethod
    def _regex_annotate(text: str) -> str:
        """Fallback: just mark sentences containing negation cues."""
        sentences = re.split(r"(?<=[.;])\s+", text)
        marked = []
        for s in sentences:
            if _NEG_REGEX.search(s):
                marked.append(f"[CONTAINS_NEGATION] {s}")
            else:
                marked.append(s)
        return " ".join(marked)

    def has_negation(self, text: str) -> bool:
        """Cheap test: does the text contain any negation cue?"""
        return bool(_NEG_REGEX.search(text))


# Module-level convenience
_default_annotator = NegationAnnotator()


def annotate_negations(text: str) -> str:
    return _default_annotator.annotate(text)


def has_negation(text: str) -> bool:
    return _default_annotator.has_negation(text)
