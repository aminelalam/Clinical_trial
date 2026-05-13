"""Clinical triage for bounded eligibility criteria.

CT.gov eligibility sections often contain dozens of bullets. In smoke and
budgeted benchmark runs we cannot evaluate every criterion, so the cap must
select clinically discriminative criteria rather than a prefix of the section.

This module is deliberately deterministic and auditable. It runs before the LLM
criterion classifier, so it relies on lexical clinical cues and the already
extracted patient profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..models.criterion import Criterion, CriterionType, Polarity
from ..models.patient import PatientProfile


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+-]{1,}", re.I)

_ADMIN_PATTERNS = (
    r"\binformed consent\b",
    r"\bwritten consent\b",
    r"\bsign(?:ed|ing)? consent\b",
    r"\bwilling(?:ness)?\b",
    r"\bable to comply\b",
    r"\bcomply with\b",
    r"\bstudy procedures\b",
    r"\bfollow[- ]?up\b",
    r"\bavailable for\b",
)

_HEADER_REFERENCE_PATTERNS = (
    r"^see\s+(?:disease|patient|prior|treatment|therapy|radiotherapy|surgery|eligibility)"
    r"\s+(?:characteristics|criteria|section|requirements?)$",
    r"^(?:disease|patient|prior|treatment|therapy|radiotherapy|surgery)\s+characteristics$",
)

_DROP_ONLY_HEADER_PATTERNS = (
    r"^see\s+",
    r"^(?:inclusion|exclusion)(?:\s+criteria)?$",
    r"^(?:or|other)$",
    r"^part\s+\d+[a-z]?$",
    r"^arm\s+\d+[a-z]?$",
    r"^for all subjects$",
)

_HEADER_HINTS = {
    "disease characteristics",
    "patient characteristics",
    "prior therapy",
    "prior therapies",
    "prior/concurrent therapy",
    "prior/concurrent therapies",
    "radiotherapy",
    "radiation therapy",
    "surgery",
    "for control participants",
    "for experimental participants",
    "for treatment participants",
    "for non-operated patients",
    "laboratory criteria",
    "organ function",
    "exclusion criteria",
    "inclusion criteria",
}

_TYPE_PATTERNS: tuple[tuple[CriterionType, tuple[str, ...]], ...] = (
    (
        CriterionType.SECTION_HEADER,
        (
            r"^\s*(?:disease|patient|prior|treatment|therapy|radiotherapy|surgery)"
            r"\s+(?:characteristics|criteria|section|requirements?)\s*:?\s*$",
            r"^\s*for\s+(?:control|experimental|treatment|non-operated)\s+participants?\s*:?\s*$",
            r"^\s*see\s+.+(?:characteristics|criteria|section|requirements?)\s*$",
        ),
    ),
    (
        CriterionType.AGE,
        (
            r"\bage\b",
            r"\byears?\s+(?:old|of age)\b",
            r"\b(?:older|younger) than\b",
        ),
    ),
    (
        CriterionType.SEX,
        (
            r"\bmale\b",
            r"\bfemale\b",
            r"\bmen\b",
            r"\bwomen\b",
            r"\bwoman\b",
            r"\bman\b",
            r"\bsex\b",
        ),
    ),
    (
        CriterionType.PERFORMANCE_STATUS,
        (
            r"\becog\b",
            r"\bkarnofsky\b",
            r"\bperformance status\b",
            r"\bps\s*[0-5]\b",
        ),
    ),
    (
        CriterionType.BIOMARKER,
        (
            r"\begfr\b",
            r"\balk\b",
            r"\bros1\b",
            r"\bbraf\b",
            r"\bkras\b",
            r"\bnras\b",
            r"\bher2\b",
            r"\berbb2\b",
            r"\bbrca[12]?\b",
            r"\bpalb2\b",
            r"\bhrd\b",
            r"\bmsi\b",
            r"\bmmr\b",
            r"\bpd-?l1\b",
            r"\btmb\b",
            r"\bmutation\b",
            r"\bmutated\b",
            r"\bamplification\b",
            r"\boverexpress",
            r"\bpositive\b",
            r"\bnegative\b",
            r"\breceptor\b",
        ),
    ),
    (
        CriterionType.PRIOR_TREATMENT,
        (
            r"\bprior\b",
            r"\bprevious\b",
            r"\bpretreated\b",
            r"\brefractory\b",
            r"\brelaps",
            r"\bprogress(?:ed|ion)? after\b",
            r"\breceived\b",
            r"\btreated with\b",
            r"\bchemotherap",
            r"\bimmunotherap",
            r"\bradiation\b",
            r"\bradiotherap",
            r"\bsurgery\b",
            r"\bcheckpoint\b",
            r"\bplatinum\b",
            r"\bline of therapy\b",
        ),
    ),
    (
        CriterionType.LAB,
        (
            r"\banc\b",
            r"\bneutrophil",
            r"\bplatelet",
            r"\bhemoglobin\b",
            r"\bcreatinine\b",
            r"\bbilirubin\b",
            r"\bast\b",
            r"\balt\b",
            r"\binr\b",
            r"\begfr\b",
            r"\brenal function\b",
            r"\bhepatic function\b",
            r"\borgan function\b",
            r"\blaboratory\b",
        ),
    ),
    (
        CriterionType.PREGNANCY,
        (
            r"\bpregnan",
            r"\bbreast[- ]?feeding\b",
            r"\blactating\b",
            r"\bchildbearing potential\b",
            r"\bcontraception\b",
        ),
    ),
    (
        CriterionType.COMORBIDITY,
        (
            r"\binfection\b",
            r"\bhiv\b",
            r"\bhepatitis\b",
            r"\bcardiac\b",
            r"\bheart\b",
            r"\bcns\b",
            r"\bbrain metast",
            r"\bautoimmune\b",
            r"\buncontrolled\b",
            r"\bseizure\b",
            r"\bdiabetes\b",
            r"\bcomorbid",
            r"\bconcurrent illness\b",
        ),
    ),
    (
        CriterionType.DIAGNOSIS,
        (
            r"\bhistolog",
            r"\bcytolog",
            r"\bdiagnos",
            r"\bconfirmed\b",
            r"\bdisease\b",
            r"\bcancer\b",
            r"\bcarcinoma\b",
            r"\bmelanoma\b",
            r"\blymphoma\b",
            r"\bleukemia\b",
            r"\bmyeloma\b",
            r"\bsarcoma\b",
            r"\bstage\b",
            r"\bmetastatic\b",
            r"\bunresectable\b",
            r"\badvanced\b",
            r"\btumou?r\b",
            r"\bmalignan",
        ),
    ),
    (
        CriterionType.CONSENT,
        (
            r"\bconsent\b",
            r"\bwilling\b",
            r"\bcomply\b",
        ),
    ),
)

_BASE_TYPE_SCORE = {
    CriterionType.BIOMARKER: 8.0,
    CriterionType.DIAGNOSIS: 7.0,
    CriterionType.PRIOR_TREATMENT: 6.5,
    CriterionType.PERFORMANCE_STATUS: 5.8,
    CriterionType.LAB: 5.4,
    CriterionType.AGE: 5.0,
    CriterionType.SEX: 4.8,
    CriterionType.COMORBIDITY: 4.7,
    CriterionType.PREGNANCY: 4.4,
    CriterionType.CONSENT: 1.0,
    CriterionType.SECTION_HEADER: 0.0,
    CriterionType.OTHER: 2.0,
}

_PATIENT_STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "from",
    "that",
    "this",
    "than",
    "then",
    "have",
    "has",
    "had",
    "with",
    "without",
    "within",
    "current",
    "active",
    "prior",
    "previous",
    "treatment",
    "treated",
    "therapy",
    "therapies",
    "surgery",
    "surgical",
    "symptom",
    "symptoms",
    "with",
    "year",
    "old",
    "years",
    "male",
    "female",
    "patient",
    "history",
    "stage",
    "cancer",
    "disease",
    "tumor",
    "tumour",
}


@dataclass(frozen=True)
class CriterionTriageResult:
    selected: list[Criterion]
    diagnostics: dict[str, Any]


def infer_criterion_type(text: str) -> CriterionType:
    """Infer a rough criterion type from raw text before LLM classification."""
    if is_section_header(text):
        return CriterionType.SECTION_HEADER
    low = f" {text.lower()} "
    for ctype, patterns in _TYPE_PATTERNS:
        if any(re.search(pattern, low) for pattern in patterns):
            return ctype
    return CriterionType.OTHER


def is_section_header(text: str) -> bool:
    """Return True when a segment is a CT.gov heading, not an evaluable criterion."""
    clean = _normalise_header_text(text)
    if not clean:
        return False
    low = clean.lower().rstrip(":").strip()
    if low in _HEADER_HINTS:
        return True
    if any(re.search(pattern, low) for pattern in _HEADER_REFERENCE_PATTERNS):
        return True
    if not clean.endswith(":"):
        return False
    words = _WORD_RE.findall(low)
    if len(words) > 8:
        return False
    if re.search(r"(>=|<=|>|<|=|\bmust\b|\brequired\b|\brequires\b|\bno\b|\bnot\b|\bwithout\b)", low):
        return False
    return True


def section_header_action(text: str) -> str | None:
    """Return how a detected CT.gov section header should be handled.

    ``merge`` means the header adds useful context to the following bullet
    ("Radiotherapy:", "Adequate liver function:"). ``drop`` means the header is
    structural navigation only ("See Disease Characteristics", "OR").
    """
    if not is_section_header(text):
        return None
    low = _normalise_header_text(text).lower().rstrip(":").strip()
    if any(re.search(pattern, low) for pattern in _DROP_ONLY_HEADER_PATTERNS):
        return "drop"
    return "merge"


def _normalise_header_text(text: str) -> str:
    return re.sub(r"^\s*(?:[-*•·]|\d+\.)\s*", "", text.strip())


def _is_administrative(text: str) -> bool:
    low = text.lower()
    return any(re.search(pattern, low) for pattern in _ADMIN_PATTERNS)


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    out = {m.group(0).lower() for m in _WORD_RE.finditer(text)}
    return {t for t in out if len(t) >= 3 and t not in _PATIENT_STOPWORDS}


def _patient_terms(patient: PatientProfile | None) -> set[str]:
    if patient is None:
        return set()
    terms: set[str] = set()
    terms |= _tokens(patient.primary_diagnosis)
    terms |= _tokens(patient.primary_diagnosis_stage)
    for diagnosis in patient.secondary_diagnoses:
        terms |= _tokens(diagnosis)
    for biomarker in patient.biomarkers:
        terms |= _tokens(biomarker.name)
        terms |= _tokens(biomarker.status)
        terms |= _tokens(biomarker.value)
    for treatment in patient.prior_treatments:
        terms |= _tokens(treatment.name)
        terms |= _tokens(treatment.category)
    for comorbidity in patient.comorbidities:
        terms |= _tokens(comorbidity.name)
    for lab in patient.labs:
        terms |= _tokens(lab.name)
    return terms


def _score_criterion(
    criterion: Criterion,
    *,
    patient_terms: set[str],
) -> tuple[float, list[str], CriterionType]:
    text = criterion.raw_text or ""
    inferred = criterion.type if criterion.type != CriterionType.OTHER else infer_criterion_type(text)
    score = _BASE_TYPE_SCORE.get(inferred, 2.0)
    reasons = [f"type:{inferred.value}"]

    if criterion.polarity == Polarity.EXCLUSION:
        score += 0.4
        reasons.append("exclusion")
        if inferred in {
            CriterionType.COMORBIDITY,
            CriterionType.PREGNANCY,
            CriterionType.PRIOR_TREATMENT,
            CriterionType.LAB,
            CriterionType.PERFORMANCE_STATUS,
        }:
            score += 0.8
            reasons.append("high_signal_exclusion")

    if criterion.is_mandatory:
        score += 0.3
        reasons.append("mandatory")
    if criterion.predicate is not None:
        score += 0.4
        reasons.append("typed_predicate")

    low = text.lower()
    matched_terms = sorted(patient_terms & _tokens(low))
    if matched_terms:
        boost = min(3.0, 0.7 * len(matched_terms))
        score += boost
        reasons.append(f"patient_match:{','.join(matched_terms[:4])}")

    if re.search(r"\b(stage|metastatic|unresectable|advanced|refractory|progression)\b", low):
        score += 0.8
        reasons.append("disease_severity")
    if re.search(r"\b(mutated|mutation|positive|negative|amplification|expression|receptor)\b", low):
        score += 0.7
        reasons.append("molecular_status")
    if re.search(r"\b(ecog|karnofsky|performance status)\b", low):
        score += 0.5
        reasons.append("functional_status")

    if _is_administrative(low):
        score -= 3.5
        reasons.append("administrative_penalty")
    if re.search(r"\badequate\b.*\b(function|organ|laboratory)\b", low):
        score -= 0.6
        reasons.append("broad_organ_function")

    return max(score, 0.0), reasons, inferred


def _counts_by(items: Iterable[Criterion], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        value = getattr(item, attr)
        key = getattr(value, "value", str(value))
        out[key] = out.get(key, 0) + 1
    return out


def select_criteria(
    criteria: list[Criterion],
    max_count: int,
    *,
    patient: PatientProfile | None = None,
) -> CriterionTriageResult:
    """Select a clinically balanced subset and return audit diagnostics."""
    if not criteria:
        return CriterionTriageResult(
            selected=[],
            diagnostics={
                "triage_enabled": max_count > 0,
                "total_seen": 0,
                "selected": 0,
                "dropped": 0,
            },
        )

    terms = _patient_terms(patient)
    scored: list[tuple[Criterion, float, list[str], int]] = []
    for idx, criterion in enumerate(criteria):
        score, reasons, inferred = _score_criterion(criterion, patient_terms=terms)
        enriched = criterion.model_copy(
            update={
                "type": inferred,
                "triage_score": round(score, 3),
                "triage_reasons": reasons,
            }
        )
        scored.append((enriched, score, reasons, idx))

    if max_count <= 0 or len(scored) <= max_count:
        selected_scored = scored
    else:
        inclusions = [row for row in scored if row[0].polarity == Polarity.INCLUSION]
        exclusions = [row for row in scored if row[0].polarity == Polarity.EXCLUSION]
        inc_quota = (max_count + 1) // 2 if exclusions else max_count
        exc_quota = max_count - inc_quota

        def by_score(rows: list[tuple[Criterion, float, list[str], int]]):
            return sorted(rows, key=lambda row: (-row[1], row[3]))

        selected_ids: set[str] = set()
        selected_scored: list[tuple[Criterion, float, list[str], int]] = []

        def add(rows, limit: int) -> None:
            for row in rows:
                if len(selected_scored) >= max_count or limit <= 0:
                    return
                if row[0].id in selected_ids:
                    continue
                selected_scored.append(row)
                selected_ids.add(row[0].id)
                limit -= 1

        add(by_score(inclusions), inc_quota)
        add(by_score(exclusions), exc_quota)
        add(by_score(scored), max_count - len(selected_scored))

    selected_scored = sorted(selected_scored, key=lambda row: row[3])
    selected_ids = {row[0].id for row in selected_scored}
    dropped_scored = [row for row in scored if row[0].id not in selected_ids]
    selected = [row[0] for row in selected_scored]
    dropped = [row[0] for row in dropped_scored]
    selected_scores = [row[1] for row in selected_scored]

    diagnostics = {
        "triage_enabled": max_count > 0,
        "total_seen": len(criteria),
        "selected": len(selected),
        "dropped": len(dropped),
        "selected_by_type": _counts_by(selected, "type"),
        "dropped_by_type": _counts_by(dropped, "type"),
        "selected_by_polarity": _counts_by(selected, "polarity"),
        "dropped_by_polarity": _counts_by(dropped, "polarity"),
        "mean_selected_score": round(sum(selected_scores) / len(selected_scores), 3)
        if selected_scores
        else 0.0,
        "top_dropped": [
            {
                "id": row[0].id,
                "type": row[0].type.value,
                "polarity": row[0].polarity.value,
                "score": round(row[1], 3),
                "reasons": row[2],
                "text": row[0].raw_text[:180],
            }
            for row in sorted(dropped_scored, key=lambda row: (-row[1], row[3]))[:5]
        ],
    }
    return CriterionTriageResult(selected=selected, diagnostics=diagnostics)
