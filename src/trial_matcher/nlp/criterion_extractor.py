"""Extract a structured list of Criterion objects from a Trial.

Pre-processes the raw eligibility text with NegEx and temporal extraction so
the LLM has explicit markers to work from. This is the single biggest lever
for T2 (Micro-F1) per the upgrade doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm.client import UnifiedLLM
from ..llm.prompts import CRITERION_CLASSIFY_V1, CRITERION_EXTRACT_V1
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.criterion import Criterion, CriterionType, Polarity, Predicate
from ..models.patient import PatientProfile
from ..models.trial import Trial
from .negation import annotate_negations, has_negation
from .criterion_triage import (
    CriterionTriageResult,
    infer_criterion_type,
    is_section_header,
    section_header_action,
    select_criteria,
)
from .temporal import annotate_temporal


class _ExtractedCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore")
    criteria: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class CriterionExtractionResult:
    criteria: list[Criterion]
    diagnostics: dict[str, Any]


_CRITERION_CLASSIFY_MIN_TOKENS = 4800


class CriterionExtractor:
    """Extract structured criteria from a Trial.eligibility section."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def extract(
        self,
        trial: Trial,
        max_criteria: int = 0,
        patient: PatientProfile | None = None,
        use_triage: bool = False,
        use_section_header_policy: bool = True,
    ) -> list[Criterion]:
        result = await self.extract_with_diagnostics(
            trial,
            max_criteria=max_criteria,
            patient=patient,
            use_triage=use_triage,
            use_section_header_policy=use_section_header_policy,
        )
        return result.criteria

    async def extract_with_diagnostics(
        self,
        trial: Trial,
        max_criteria: int = 0,
        patient: PatientProfile | None = None,
        use_triage: bool = False,
        use_section_header_policy: bool = True,
    ) -> CriterionExtractionResult:
        text = trial.eligibility.raw_text
        if not text:
            return CriterionExtractionResult(
                criteria=[],
                diagnostics={
                    "source": "empty",
                    "triage_enabled": max_criteria > 0,
                    "total_seen": 0,
                    "selected": 0,
                    "dropped": 0,
                },
            )

        segmented, segmentation_diagnostics = self._fallback_split_with_diagnostics(
            trial,
            use_section_header_policy=use_section_header_policy,
        )
        if segmented:
            selected_result = (
                select_criteria(segmented, max_criteria, patient=patient)
                if use_triage
                else _legacy_select_criteria(segmented, max_criteria)
            )
            classified = await self._classify_segmented(trial, selected_result.selected)
            return CriterionExtractionResult(
                criteria=classified,
                diagnostics={
                    "source": "segmented",
                    "selection_policy": "clinical_triage" if use_triage else "legacy_balanced",
                    **segmentation_diagnostics,
                    **selected_result.diagnostics,
                    "classified_by_type": _counts_by_type(classified),
                },
            )

        raw = await self._extract_from_raw_text(
            trial,
            text,
            use_section_header_policy=use_section_header_policy,
        )
        selected_result = (
            select_criteria(raw, max_criteria, patient=patient)
            if use_triage
            else _legacy_select_criteria(raw, max_criteria)
        )
        return CriterionExtractionResult(
            criteria=selected_result.selected,
            diagnostics={
                "source": "raw_llm",
                "selection_policy": "clinical_triage" if use_triage else "legacy_balanced",
                **selected_result.diagnostics,
                "classified_by_type": _counts_by_type(selected_result.selected),
            },
        )

    async def _classify_segmented(self, trial: Trial, segmented: list[Criterion]) -> list[Criterion]:
        """Classify deterministic segments instead of asking the LLM to segment raw text.

        This keeps reasoning-model prompts bounded: segmentation is mechanical and
        local, while the LLM only assigns type/predicate metadata for criteria we
        are actually going to evaluate.
        """
        by_id = {c.id: c for c in segmented}
        criteria_block = "\n".join(
            f"{c.id} | {c.polarity.value} | {c.annotated_text or c.raw_text}"
            for c in segmented
        )
        prompt = CRITERION_CLASSIFY_V1.format(
            nct_id=trial.nct_id,
            trial_title=trial.title,
            criteria_block=criteria_block,
        )
        max_tokens = max(_CRITERION_CLASSIFY_MIN_TOKENS, min(9000, 1000 + 350 * len(segmented)))

        try:
            extracted = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_ExtractedCriteria,
                model="mini",
                temperature=0.0,
                max_tokens=max_tokens,
                max_retries=1,
                task_name="criterion_classify",
            )
        except Exception as e:
            logger.warning(f"Criterion extraction failed for {trial.nct_id}: {e!r}; "
                           "using deterministic segmented fallback")
            return segmented

        out: list[Criterion] = []
        seen: set[str] = set()
        for raw in extracted.criteria:
            try:
                cid = str(raw.get("id") or "")
                source = by_id.get(cid)
                if source is not None:
                    raw = {
                        **raw,
                        "id": source.id,
                        "polarity": source.polarity.value,
                        "raw_text": raw.get("raw_text") or source.raw_text,
                        "type": raw.get("type") or source.type.value,
                        "annotated_text": source.annotated_text,
                        "has_negation": raw.get("has_negation", source.has_negation),
                        "triage_score": source.triage_score,
                        "triage_reasons": source.triage_reasons,
                    }
                coerced = self._coerce(raw)
                if coerced.id in by_id and coerced.id not in seen:
                    out.append(coerced)
                    seen.add(coerced.id)
            except Exception as e:
                logger.debug(f"Skipping malformed criterion: {e}")
        for c in segmented:
            if c.id not in seen:
                out.append(c)
        return sorted(out, key=lambda c: list(by_id).index(c.id))

    async def _extract_from_raw_text(
        self,
        trial: Trial,
        text: str,
        use_section_header_policy: bool = True,
    ) -> list[Criterion]:
        annotated = annotate_temporal(annotate_negations(text))
        prompt = CRITERION_EXTRACT_V1.format(
            nct_id=trial.nct_id, trial_title=trial.title, annotated_text=annotated
        )

        try:
            extracted = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_ExtractedCriteria,
                model="mini",
                temperature=0.0,
                max_tokens=6000,
                max_retries=1,
                task_name="criterion_extract_raw",
            )
        except Exception as e:
            logger.warning(f"Criterion extraction failed for {trial.nct_id}: {e!r}; "
                           "falling back to regex-split fallback")
            return self._fallback_split(
                trial,
                use_section_header_policy=use_section_header_policy,
            )

        out: list[Criterion] = []
        for raw in extracted.criteria:
            try:
                out.append(self._coerce(raw))
            except Exception as e:
                logger.debug(f"Skipping malformed criterion: {e}")
        return out

    @staticmethod
    def _coerce(raw: dict[str, Any]) -> Criterion:
        polarity = Polarity(raw.get("polarity", "inclusion"))
        type_ = CriterionType(raw.get("type", "other")) if raw.get("type") else CriterionType.OTHER
        pred_raw = raw.get("predicate")
        pred = None
        if isinstance(pred_raw, dict) and pred_raw.get("op") and pred_raw.get("feature"):
            try:
                pred = Predicate.model_validate(pred_raw)
            except Exception:
                pred = None
        rt = (raw.get("raw_text") or "").strip()
        return Criterion(
            id=str(raw.get("id") or "x_0"),
            polarity=polarity,
            raw_text=rt,
            type=type_,
            predicate=pred,
            is_mandatory=bool(raw.get("is_mandatory", True)),
            has_negation=bool(raw.get("has_negation", has_negation(rt))),
            annotated_text=annotate_temporal(annotate_negations(rt)) if rt else None,
            extraction_confidence=float(raw.get("extraction_confidence", 1.0)),
            triage_score=float(raw.get("triage_score", 0.0) or 0.0),
            triage_reasons=list(raw.get("triage_reasons", []) or []),
        )

    # ------------------------------------------------------------------
    # Fallback when the LLM call fails — best-effort regex split
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_split(
        trial: Trial,
        use_section_header_policy: bool = True,
    ) -> list[Criterion]:
        return CriterionExtractor._fallback_split_with_diagnostics(
            trial,
            use_section_header_policy=use_section_header_policy,
        )[0]

    @staticmethod
    def _fallback_split_with_diagnostics(
        trial: Trial,
        use_section_header_policy: bool = True,
    ) -> tuple[list[Criterion], dict[str, int]]:
        out: list[Criterion] = []
        headers_merged = 0
        headers_dropped = 0
        for kind, text in [
            (Polarity.INCLUSION, trial.eligibility.inclusion_text),
            (Polarity.EXCLUSION, trial.eligibility.exclusion_text),
        ]:
            if not text:
                continue
            chunks, diagnostics = _split_bullets_with_diagnostics(
                text,
                use_section_header_policy=use_section_header_policy,
            )
            headers_merged += diagnostics["section_headers_merged"]
            headers_dropped += diagnostics["section_headers_dropped"]
            for i, line in enumerate(chunks, start=1):
                if not line:
                    continue
                criterion_type = infer_criterion_type(line)
                if criterion_type == CriterionType.SECTION_HEADER:
                    if use_section_header_policy:
                        headers_dropped += 1
                        continue
                    criterion_type = CriterionType.OTHER
                prefix = "i" if kind == Polarity.INCLUSION else "e"
                out.append(
                    Criterion(
                        id=f"{prefix}_{i}",
                        polarity=kind,
                        raw_text=line.strip(),
                        type=criterion_type,
                        annotated_text=annotate_temporal(annotate_negations(line)),
                        has_negation=has_negation(line),
                    )
                )
        return out, {
            "section_headers_merged": headers_merged,
            "section_headers_dropped": headers_dropped,
        }


def _split_bullets_with_diagnostics(
    text: str,
    use_section_header_policy: bool = True,
) -> tuple[list[str], dict[str, int]]:
    import re

    if not use_section_header_policy:
        return _legacy_regex_chunks(text), {
            "section_headers_merged": 0,
            "section_headers_dropped": 0,
        }

    chunks = _line_chunks(text)
    if len(chunks) <= 1:
        # Try sentence-based split as last resort
        chunks = re.split(r"(?<=[.;])\s+(?=[A-Z])", text)
    merged: list[str] = []
    pending_headers: list[str] = []
    headers_merged = 0
    headers_dropped = 0
    for raw in chunks:
        chunk = (raw or "").strip()
        if not chunk:
            continue
        header_action = section_header_action(chunk)
        if header_action == "drop":
            headers_dropped += 1
            continue
        if header_action == "merge":
            pending_headers.append(chunk.rstrip(":").strip())
            continue
        if pending_headers:
            headers_merged += len(pending_headers)
            prefix = " / ".join(pending_headers)
            chunk = f"{prefix}: {chunk}"
            pending_headers = []
        merged.append(chunk)

    headers_dropped += len(pending_headers)
    return merged, {
        "section_headers_merged": headers_merged,
        "section_headers_dropped": headers_dropped,
    }


def _legacy_regex_chunks(text: str) -> list[str]:
    import re

    chunks = re.split(r"\n\s*(?:[-*\u2022\u00b7]|\d+\.)\s*|\n\s*\n", text)
    if len(chunks) <= 1:
        chunks = re.split(r"(?<=[.;])\s+(?=[A-Z])", text)
    return [c.strip() for c in chunks if c and c.strip()]


def _line_chunks(text: str) -> list[str]:
    """Split CT.gov eligibility text while preserving wrapped criterion lines."""
    import re

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
            current = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        bullet = re.match(r"^(?:[-*•·]|\d+\.)\s*(.+)$", line)
        if bullet:
            flush()
            current = [bullet.group(1).strip()]
            continue

        if is_section_header(line):
            flush()
            chunks.append(line)
            continue

        if current:
            current.append(line)
        else:
            current = [line]

    flush()
    return [chunk for chunk in chunks if chunk]


def _balanced_limit(criteria: list[Criterion], max_count: int) -> list[Criterion]:
    """Backward-compatible wrapper around the original balanced cap."""
    return _legacy_select_criteria(criteria, max_count).selected


def _legacy_select_criteria(criteria: list[Criterion], max_count: int) -> CriterionTriageResult:
    """Original polarity-balanced cap, kept as the default benchmark policy.

    It intentionally ignores type scores. The P5 triage experiment improved
    some ranking metrics but regressed trial-level eligibility on 75 topics, so
    the clinically scored selector remains opt-in.
    """
    if max_count <= 0 or len(criteria) <= max_count:
        selected = list(criteria)
    else:
        selected = []
        selected_ids: set[str] = set()

        def add(items: list[Criterion], limit: int) -> None:
            for item in items:
                if len(selected) >= max_count or limit <= 0:
                    return
                if item.id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.id)
                limit -= 1

        inclusions = [c for c in criteria if c.polarity == Polarity.INCLUSION]
        exclusions = [c for c in criteria if c.polarity == Polarity.EXCLUSION]
        inc_quota = (max_count + 1) // 2 if exclusions else max_count
        exc_quota = max_count - inc_quota

        add(inclusions, inc_quota)
        add(exclusions, exc_quota)
        add(criteria, max_count - len(selected))

        selected_order = {c.id: i for i, c in enumerate(criteria)}
        selected = sorted(selected, key=lambda c: selected_order[c.id])

    selected_ids = {c.id for c in selected}
    dropped = [c for c in criteria if c.id not in selected_ids]
    return CriterionTriageResult(
        selected=selected,
        diagnostics={
            "triage_enabled": False,
            "total_seen": len(criteria),
            "selected": len(selected),
            "dropped": len(dropped),
            "selected_by_type": _counts_by_type(selected),
            "dropped_by_type": _counts_by_type(dropped),
            "selected_by_polarity": _counts_by_polarity(selected),
            "dropped_by_polarity": _counts_by_polarity(dropped),
            "mean_selected_score": 0.0,
            "top_dropped": [],
        },
    )


def _counts_by_type(criteria: list[Criterion]) -> dict[str, int]:
    out: dict[str, int] = {}
    for criterion in criteria:
        out[criterion.type.value] = out.get(criterion.type.value, 0) + 1
    return out


def _counts_by_polarity(criteria: list[Criterion]) -> dict[str, int]:
    out: dict[str, int] = {}
    for criterion in criteria:
        out[criterion.polarity.value] = out.get(criterion.polarity.value, 0) + 1
    return out
