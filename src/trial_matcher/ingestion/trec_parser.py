"""Parse TREC Clinical Trials topics (XML) and qrels (TSV).

Topics format (2021/2022):
    <topics task="2021 TREC Clinical Trials Track">
      <topic number="1">47-year-old woman with ...</topic>
      ...
    </topics>

Qrels format (TREC standard):
    topic_id Q0 nct_id grade
where grade ∈ {0, 1, 2}: 0=irrelevant, 1=excludes, 2=eligible.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict


class TrecTopic(BaseModel):
    """A single TREC CT topic (synthetic patient note)."""

    model_config = ConfigDict(extra="ignore")

    topic_id: str
    text: str


class Qrel(NamedTuple):
    """A single qrel line."""

    topic_id: str
    nct_id: str
    grade: int  # 0, 1, or 2


def parse_topics(path: Path | str) -> list[TrecTopic]:
    """Parse a TREC CT topics XML into a list of TrecTopic objects."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    root = ET.fromstring(p.read_text(encoding="utf-8"))
    topics: list[TrecTopic] = []
    for el in root.findall(".//topic"):
        tid = el.get("number") or el.get("id") or ""
        text = (el.text or "").strip()
        if tid and text:
            topics.append(TrecTopic(topic_id=tid, text=text))
    return topics


def parse_qrels(path: Path | str) -> list[Qrel]:
    """Parse a TREC qrels file. Tolerates whitespace and trailing newlines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out: list[Qrel] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        topic_id, _q0, nct_id, grade = parts[:4]
        try:
            out.append(Qrel(topic_id=topic_id, nct_id=nct_id, grade=int(grade)))
        except ValueError:
            continue
    return out


def qrels_to_trec_eval_dict(qrels: Iterable[Qrel]) -> dict[str, dict[str, int]]:
    """Convert qrels to the dict-of-dicts format pytrec_eval expects."""
    out: dict[str, dict[str, int]] = {}
    for q in qrels:
        out.setdefault(q.topic_id, {})[q.nct_id] = q.grade
    return out
