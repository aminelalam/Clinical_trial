"""Parse ClinicalTrials.gov v2 API JSON into Trial objects.

The v2 schema is structured under nested modules:
    protocolSection.identificationModule
    protocolSection.descriptionModule
    protocolSection.conditionsModule
    protocolSection.designModule
    protocolSection.eligibilityModule
    protocolSection.contactsLocationsModule
    protocolSection.statusModule
    protocolSection.sponsorCollaboratorsModule

This parser is deliberately tolerant: missing modules degrade gracefully.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..models.trial import (
    AgeRange,
    Contact,
    Eligibility,
    Location,
    Phase,
    RecruitmentStatus,
    Sex,
    Trial,
)

_AGE_RE = re.compile(r"(\d+)\s*(year|month|week|day|hour|minute)", re.IGNORECASE)


def _age_to_days(raw: str | None) -> int | None:
    if not raw:
        return None
    m = _AGE_RE.search(raw)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    factor = {"year": 365, "month": 30, "week": 7, "day": 1, "hour": 0, "minute": 0}[unit]
    return n * factor


def _parse_phase(value: list[str] | str | None) -> Phase:
    if not value:
        return Phase.NA
    if isinstance(value, list):
        value = value[0] if value else "NA"
    v = (value or "NA").upper().replace("PHASE ", "PHASE").replace("/", "_")
    try:
        return Phase(v)
    except ValueError:
        for member in Phase:
            if member.value in v or v in member.value:
                return member
    return Phase.NA


def _parse_status(value: str | None) -> RecruitmentStatus:
    if not value:
        return RecruitmentStatus.UNKNOWN
    v = value.upper().replace(" ", "_")
    try:
        return RecruitmentStatus(v)
    except ValueError:
        return RecruitmentStatus.UNKNOWN


def _parse_sex(value: str | None) -> Sex:
    if not value:
        return Sex.ALL
    v = value.upper()
    return Sex(v) if v in {s.value for s in Sex} else Sex.ALL


def _parse_date(value: str | dict | None) -> date | None:
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("date") or value.get("Date")
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _split_eligibility(text: str) -> tuple[str, str]:
    """Heuristic split of the criteria text into inclusion/exclusion sections."""
    if not text:
        return "", ""
    t = text
    # Common section markers
    incl_match = re.search(r"inclusion\s*criteria\s*:?", t, re.IGNORECASE)
    excl_match = re.search(r"exclusion\s*criteria\s*:?", t, re.IGNORECASE)
    if incl_match and excl_match and incl_match.start() < excl_match.start():
        return (
            t[incl_match.end() : excl_match.start()].strip(),
            t[excl_match.end() :].strip(),
        )
    if incl_match and not excl_match:
        return t[incl_match.end() :].strip(), ""
    if excl_match and not incl_match:
        return t[: excl_match.start()].strip(), t[excl_match.end() :].strip()
    # Fallback: assume entire text is inclusion-flavored
    return t.strip(), ""


def parse_ctgov_study(study: dict[str, Any]) -> Trial | None:
    """Parse a single v2 API study record. Returns None if no NCT id is present."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {}) or {}
    desc = proto.get("descriptionModule", {}) or {}
    cond = proto.get("conditionsModule", {}) or {}
    design = proto.get("designModule", {}) or {}
    elig = proto.get("eligibilityModule", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    contacts_locs = proto.get("contactsLocationsModule", {}) or {}
    status = proto.get("statusModule", {}) or {}
    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}

    nct_id = ident.get("nctId") or ident.get("NCTId")
    if not nct_id:
        return None

    raw_criteria = elig.get("eligibilityCriteria") or ""
    incl_text, excl_text = _split_eligibility(raw_criteria)

    locations = []
    for loc in (contacts_locs.get("locations") or [])[:50]:
        locations.append(
            Location(
                facility=loc.get("facility"),
                city=loc.get("city"),
                state=loc.get("state"),
                country=loc.get("country"),
                status=loc.get("status"),
            )
        )

    contacts = []
    for c in (contacts_locs.get("centralContacts") or [])[:5]:
        contacts.append(
            Contact(
                name=c.get("name"),
                role=c.get("role"),
                email=c.get("email"),
                phone=c.get("phone"),
            )
        )

    return Trial(
        nct_id=nct_id,
        title=ident.get("briefTitle") or "",
        official_title=ident.get("officialTitle"),
        brief_summary=desc.get("briefSummary") or "",
        detailed_description=desc.get("detailedDescription"),
        conditions=cond.get("conditions") or [],
        keywords=cond.get("keywords") or [],
        interventions=[
            str(i.get("name"))
            for i in (arms.get("interventions") or [])
            if isinstance(i, dict) and i.get("name")
        ],
        phase=_parse_phase(design.get("phases")),
        status=_parse_status(status.get("overallStatus")),
        study_type=design.get("studyType"),
        interventional=(design.get("studyType") or "").upper() == "INTERVENTIONAL",
        eligibility=Eligibility(
            raw_text=raw_criteria,
            inclusion_text=incl_text,
            exclusion_text=excl_text,
            age_range=AgeRange(
                min_days=_age_to_days(elig.get("minimumAge")),
                max_days=_age_to_days(elig.get("maximumAge")),
                raw_min=elig.get("minimumAge"),
                raw_max=elig.get("maximumAge"),
            ),
            sex=_parse_sex(elig.get("sex")),
            accepts_healthy_volunteers=elig.get("healthyVolunteers"),
        ),
        locations=locations,
        contacts=contacts,
        sponsor=(sponsor.get("leadSponsor") or {}).get("name"),
        last_update_date=_parse_date(status.get("lastUpdatePostDateStruct")),
        enrollment=((design.get("enrollmentInfo") or {}).get("count")),
    )


def parse_ctgov_dump(dump_dir: Path | str) -> Iterator[Trial]:
    """Yield Trial objects from a directory of page_*.json files written by the downloader."""
    dump_dir = Path(dump_dir)
    files = sorted(dump_dir.glob("page_*.json"))
    if not files:
        return
    for f in files:
        try:
            studies = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for s in studies:
            trial = parse_ctgov_study(s)
            if trial is not None:
                yield trial
