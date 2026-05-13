"""Parse MeSH descriptor + supplemental XML into a synonym → concept index.

Used by the MeSH normalizer to expand patient terms with canonical synonyms
(e.g. "non-small cell lung cancer" → D002289 + ["NSCLC", "Non-Small Cell Lung Carcinoma", ...]).

The descriptor XML is large (~400MB). We stream it with ``iterparse`` to keep
memory bounded, and cache the parsed index to a pickle file so subsequent
process starts are nearly instant.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict

from ..config import get_settings
from ..logging import logger

_INDEX_CACHE_VERSION = 1


class MeshConcept(BaseModel):
    """A canonical MeSH concept with its known synonym surfaces."""

    model_config = ConfigDict(extra="ignore")

    concept_id: str  # e.g. "D002289"
    name: str  # canonical term
    synonyms: list[str]
    tree_numbers: list[str] = []


class MeSHIndex(BaseModel):
    """In-memory MeSH lookup index."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    concepts: dict[str, MeshConcept]  # concept_id → MeshConcept
    by_synonym: dict[str, list[str]]  # lowercased synonym → [concept_id, ...]

    def lookup(self, term: str) -> list[MeshConcept]:
        """Return all MeSH concepts whose synonyms contain the given (case-insensitive) term."""
        ids = self.by_synonym.get(term.lower().strip(), [])
        return [self.concepts[cid] for cid in ids if cid in self.concepts]


def _iter_descriptors(xml_path: Path) -> Iterator[tuple[str, str, list[str], list[str]]]:
    """Yield (descriptor_id, name, synonyms, tree_numbers) tuples."""
    if not xml_path.exists():
        return
    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if elem.tag != "DescriptorRecord":
            continue
        did_el = elem.find("DescriptorUI")
        name_el = elem.find("DescriptorName/String")
        descriptor_id = (did_el.text or "").strip() if did_el is not None else ""
        name = (name_el.text or "").strip() if name_el is not None else ""

        synonyms: list[str] = []
        if name:
            synonyms.append(name)
        for s in elem.findall(".//Term/String"):
            t = (s.text or "").strip()
            if t:
                synonyms.append(t)

        tree_numbers = [
            (t.text or "").strip()
            for t in elem.findall(".//TreeNumber")
            if t.text
        ]

        if descriptor_id and name:
            yield descriptor_id, name, list(dict.fromkeys(synonyms)), tree_numbers
        elem.clear()


def _iter_supplemental(xml_path: Path) -> Iterator[tuple[str, str, list[str]]]:
    if not xml_path.exists():
        return
    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if elem.tag != "SupplementalRecord":
            continue
        sid_el = elem.find("SupplementalRecordUI")
        name_el = elem.find("SupplementalRecordName/String")
        sid = (sid_el.text or "").strip() if sid_el is not None else ""
        name = (name_el.text or "").strip() if name_el is not None else ""
        synonyms: list[str] = [name] if name else []
        for s in elem.findall(".//Term/String"):
            t = (s.text or "").strip()
            if t:
                synonyms.append(t)
        if sid and name:
            yield sid, name, list(dict.fromkeys(synonyms))
        elem.clear()


def _files_signature(files: list[Path]) -> list[tuple[str, int, int]]:
    """A stable signature of input XML files for cache invalidation."""
    return [
        (f.name, f.stat().st_size, f.stat().st_mtime_ns)
        for f in files
    ]


def _index_cache_path(mesh_dir: Path) -> Path:
    s = get_settings()
    digest = hashlib.sha256(str(mesh_dir.resolve()).lower().encode("utf-8")).hexdigest()[:16]
    return Path(s.paths.cache_dir) / "mesh_index" / f"mesh_index_{digest}.pkl"


def _try_load_cached_index(mesh_dir: Path, files: list[Path]) -> MeSHIndex | None:
    cache_path = _index_cache_path(mesh_dir)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != _INDEX_CACHE_VERSION:
            return None
        if payload.get("files") != _files_signature(files):
            return None
        index = payload.get("index")
        if not isinstance(index, MeSHIndex):
            return None
    except Exception as e:
        logger.warning(f"Ignoring invalid MeSH index cache {cache_path}: {e!r}")
        return None
    logger.info(
        f"MeSH index loaded from cache: {len(index.concepts)} concepts, "
        f"{len(index.by_synonym)} synonyms ({cache_path})"
    )
    return index


def _write_cached_index(mesh_dir: Path, files: list[Path], index: MeSHIndex) -> None:
    cache_path = _index_cache_path(mesh_dir)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(
                {
                    "version": _INDEX_CACHE_VERSION,
                    "files": _files_signature(files),
                    "index": index,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"MeSH index cached to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not write MeSH index cache {cache_path}: {e!r}")


def load_mesh_index(mesh_dir: Path | str, use_cache: bool = True) -> MeSHIndex:
    """Stream-parse the MeSH XML files in ``mesh_dir`` into a MeSHIndex.

    Looks for files matching ``desc*.xml`` and ``supp*.xml``. Reads/writes a
    pickle cache under ``cache_dir/mesh_index/`` so the second process start
    skips the ~30s XML parse entirely.
    """
    mesh_dir = Path(mesh_dir)
    desc_files = sorted(mesh_dir.glob("desc*.xml"))
    supp_files = sorted(mesh_dir.glob("supp*.xml"))
    all_files = desc_files + supp_files

    if use_cache and all_files:
        cached = _try_load_cached_index(mesh_dir, all_files)
        if cached is not None:
            return cached

    concepts: dict[str, MeshConcept] = {}
    by_synonym: dict[str, list[str]] = {}

    for f in desc_files:
        for cid, name, syns, tree_numbers in _iter_descriptors(f):
            concepts[cid] = MeshConcept(
                concept_id=cid, name=name, synonyms=syns, tree_numbers=tree_numbers
            )
            for s in syns:
                by_synonym.setdefault(s.lower(), []).append(cid)

    for f in supp_files:
        for cid, name, syns in _iter_supplemental(f):
            concepts[cid] = MeshConcept(concept_id=cid, name=name, synonyms=syns)
            for s in syns:
                by_synonym.setdefault(s.lower(), []).append(cid)

    index = MeSHIndex(concepts=concepts, by_synonym=by_synonym)
    if use_cache and all_files:
        _write_cached_index(mesh_dir, all_files, index)
    return index
