"""Parse legacy ClinicalTrials.gov XML files used by TREC Clinical Trials.

The official TREC 2021/2022 Clinical Trials corpus is a 2021-04-27 snapshot
distributed as ClinicalTrials.gov XML, not the modern v2 JSON shape used by
``ctgov_parser``. This module keeps the conversion tolerant so older records
with missing modules remain retrievable instead of disappearing from benchmark
indexes.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

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
from .ctgov_parser import _age_to_days, _split_eligibility


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _text(root: ET.Element, path: str) -> str:
    value = root.findtext(path)
    return _clean(value)


def _texts(root: ET.Element, path: str) -> list[str]:
    return [_clean(e.text) for e in root.findall(path) if _clean(e.text)]


def _parse_phase(value: str | None) -> Phase:
    if not value:
        return Phase.NA
    v = value.upper().replace("PHASE ", "PHASE").replace("/", "_").replace("-", "_")
    if "EARLY" in v and "1" in v:
        return Phase.EARLY_PHASE_1
    if "PHASE1_PHASE2" in v or ("PHASE1" in v and "PHASE2" in v):
        return Phase.PHASE_1_2
    if "PHASE2_PHASE3" in v or ("PHASE2" in v and "PHASE3" in v):
        return Phase.PHASE_2_3
    for phase in Phase:
        if phase.value in v:
            return phase
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
    if v == "BOTH":
        return Sex.ALL
    return Sex(v) if v in {s.value for s in Sex} else Sex.ALL


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"yes", "true", "1"}:
        return True
    if v in {"no", "false", "0"}:
        return False
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\d+", value)
    return int(m.group(0)) if m else None


def _iter_locations(root: ET.Element) -> list[Location]:
    locations: list[Location] = []
    for loc in root.findall("location")[:50]:
        locations.append(
            Location(
                facility=_text(loc, "facility/name") or None,
                city=_text(loc, "facility/address/city") or None,
                state=_text(loc, "facility/address/state") or None,
                country=_text(loc, "facility/address/country") or None,
                status=_text(loc, "status") or None,
            )
        )
    return locations


def _iter_contacts(root: ET.Element) -> list[Contact]:
    contacts: list[Contact] = []
    for c in root.findall("overall_contact")[:5]:
        contacts.append(
            Contact(
                name=_text(c, "last_name") or None,
                role=None,
                email=_text(c, "email") or None,
                phone=_text(c, "phone") or None,
            )
        )
    return contacts


def parse_ctgov_xml_file(path: Path | str) -> Trial | None:
    """Parse one ClinicalTrials.gov legacy XML file into a ``Trial``."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    return parse_ctgov_xml_root(root)


def parse_ctgov_xml_root(root: ET.Element) -> Trial | None:
    """Parse a ClinicalTrials.gov ``clinical_study`` XML root."""
    nct_id = _text(root, "id_info/nct_id") or _text(root, ".//nct_id")
    if not nct_id:
        return None

    raw_criteria = _text(root, "eligibility/criteria/textblock")
    incl_text, excl_text = _split_eligibility(raw_criteria)
    study_type = _text(root, "study_type") or None

    interventions = []
    for intervention in root.findall("intervention"):
        name = _text(intervention, "intervention_name")
        if name:
            interventions.append(name)

    keywords = _texts(root, "keyword")
    keywords.extend(_texts(root, "condition_browse/mesh_term"))
    keywords.extend(_texts(root, "intervention_browse/mesh_term"))

    return Trial(
        nct_id=nct_id,
        title=_text(root, "brief_title"),
        official_title=_text(root, "official_title") or None,
        brief_summary=_text(root, "brief_summary/textblock"),
        detailed_description=_text(root, "detailed_description/textblock") or None,
        conditions=_texts(root, "condition"),
        keywords=keywords,
        interventions=interventions,
        phase=_parse_phase(_text(root, "phase")),
        status=_parse_status(_text(root, "overall_status")),
        study_type=study_type,
        interventional=(study_type or "").upper() == "INTERVENTIONAL",
        eligibility=Eligibility(
            raw_text=raw_criteria,
            inclusion_text=incl_text,
            exclusion_text=excl_text,
            age_range=AgeRange(
                min_days=_age_to_days(_text(root, "eligibility/minimum_age")),
                max_days=_age_to_days(_text(root, "eligibility/maximum_age")),
                raw_min=_text(root, "eligibility/minimum_age") or None,
                raw_max=_text(root, "eligibility/maximum_age") or None,
            ),
            sex=_parse_sex(_text(root, "eligibility/gender")),
            accepts_healthy_volunteers=_parse_bool(_text(root, "eligibility/healthy_volunteers")),
        ),
        locations=_iter_locations(root),
        contacts=_iter_contacts(root),
        sponsor=_text(root, "sponsors/lead_sponsor/agency") or None,
        last_update_date=_parse_date(
            _text(root, "last_update_posted")
            or _text(root, "last_update_submitted")
            or _text(root, "verification_date")
        ),
        enrollment=_parse_int(_text(root, "enrollment")),
    )


def parse_ctgov_xml_dump(dump_dir: Path | str) -> Iterator[Trial]:
    """Yield ``Trial`` objects from a directory tree of ``*.xml`` files."""
    for path in sorted(Path(dump_dir).rglob("*.xml")):
        trial = parse_ctgov_xml_file(path)
        if trial is not None:
            yield trial


def find_ctgov_xml_nct_id(path: Path | str) -> str | None:
    """Read only enough XML to find the NCT id for lazy corpus indexing."""
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "nct_id" and elem.text:
                return _clean(elem.text)
    except ET.ParseError:
        return None
    return None
