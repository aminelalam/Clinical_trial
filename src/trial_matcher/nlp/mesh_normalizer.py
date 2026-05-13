"""MeSH normalization for patient terms.

Wraps a MeSHIndex with cached lookups and a small noun-phrase extractor for
when the patient text is messy. Used by the planning agent to expand BM25
queries with canonical synonyms.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..ingestion.mesh_loader import MeSHIndex, MeshConcept


class MeSHNormalizer:
    """Lookup wrapper around MeSHIndex with prefix-based fallback."""

    def __init__(self, index: MeSHIndex):
        self.index = index

    @lru_cache(maxsize=4096)
    def normalize(self, term: str) -> tuple[MeshConcept, ...]:
        """Return canonical MeSH concepts for a free-text clinical term.

        Tries exact match first, then progressively shorter prefixes. Returns
        an empty tuple when no match is found.
        """
        if not term:
            return tuple()
        normalized = re.sub(r"[\(\)]", " ", term).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)

        matches = self.index.lookup(normalized)
        if matches:
            return tuple(matches)

        # Try removing common qualifiers
        for qualifier in [
            "metastatic ",
            "advanced ",
            "recurrent ",
            "primary ",
            "stage iv ",
            "stage 4 ",
        ]:
            if normalized.startswith(qualifier):
                stripped = normalized[len(qualifier) :]
                m = self.index.lookup(stripped)
                if m:
                    return tuple(m)

        # Single-word fallback: try the longest word
        words = sorted(normalized.split(), key=len, reverse=True)
        for w in words[:3]:
            m = self.index.lookup(w)
            if m:
                return tuple(m)

        return tuple()

    def expand_synonyms(self, term: str, max_synonyms: int = 8) -> list[str]:
        """Return up to ``max_synonyms`` canonical synonyms for ``term``."""
        concepts = self.normalize(term)
        if not concepts:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for c in concepts:
            for s in c.synonyms:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(s)
                if len(out) >= max_synonyms:
                    return out
        return out
