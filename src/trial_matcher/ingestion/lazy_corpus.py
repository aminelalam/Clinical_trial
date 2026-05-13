"""Lazy-loading trial corpus that loads only needed trials from disk.

Loading all 321k trials into memory at once would use several GB. Instead, we
build a small index (nct_id → page_file + position) during init, then load
individual trials on demand and cache them in an LRU dict.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from ..config import get_settings
from ..ingestion.ctgov_parser import parse_ctgov_study
from ..ingestion.ctgov_xml_parser import find_ctgov_xml_nct_id, parse_ctgov_xml_file
from ..logging import logger
from ..models.trial import Trial

_INDEX_CACHE_VERSION = 2
_IndexEntry = tuple[Path, int, str]


class LazyTrialCorpus(Mapping[str, Trial]):
    """Memory-efficient trial lookup for CT.gov v2 JSON pages or legacy XML."""

    def __init__(self, dump_dir: Path | str, max_cache: int = 2000):
        self._dump_dir = Path(dump_dir)
        # nct_id -> (file, position_in_list, format)
        self._index: dict[str, _IndexEntry] = {}
        self._cache: dict[str, Trial] = {}
        self._max_cache = max_cache
        self._build_index()

    def _index_cache_path(self) -> Path:
        s = get_settings()
        resolved = str(self._dump_dir.resolve()).lower()
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
        return Path(s.paths.cache_dir) / "corpus_index" / f"lazy_trial_corpus_{digest}.json"

    @staticmethod
    def _files_signature(files: list[Path]) -> list[dict[str, Any]]:
        return [
            {
                "name": str(f),
                "size": f.stat().st_size,
                "mtime_ns": f.stat().st_mtime_ns,
            }
            for f in files
        ]

    def _load_cached_index(self, files: list[Path]) -> bool:
        cache_path = self._index_cache_path()
        if not cache_path.exists():
            return False
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != _INDEX_CACHE_VERSION:
                return False
            if payload.get("files") != self._files_signature(files):
                return False
            raw_index = payload.get("index")
            if not isinstance(raw_index, dict):
                return False
            parsed: dict[str, _IndexEntry] = {}
            for nct_id, entry in raw_index.items():
                if not isinstance(entry, list | tuple) or len(entry) not in {2, 3}:
                    continue
                kind = str(entry[2]) if len(entry) == 3 else "json"
                parsed[str(nct_id)] = (self._dump_dir / str(entry[0]), int(entry[1]), kind)
            self._index = parsed
        except Exception as e:
            logger.warning(f"Ignoring invalid trial index cache {cache_path}: {e!r}")
            return False
        logger.info(f"Trial index loaded from cache: {len(self._index)} trials indexed")
        return True

    def _write_cached_index(self, files: list[Path]) -> None:
        cache_path = self._index_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": _INDEX_CACHE_VERSION,
                "dump_dir": str(self._dump_dir.resolve()),
                "files": self._files_signature(files),
                "index": {
                    nct_id: [str(page_file.relative_to(self._dump_dir)), pos, kind]
                    for nct_id, (page_file, pos, kind) in self._index.items()
                },
            }
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not write trial index cache {cache_path}: {e!r}")

    def _build_index(self) -> None:
        """Scan page JSON or XML files and record each NCT ID location."""
        json_files = sorted(self._dump_dir.glob("page_*.json"))
        xml_files = [] if json_files else sorted(self._dump_dir.rglob("*.xml"))
        files = json_files or xml_files
        if self._load_cached_index(files):
            return
        logger.info(f"Building trial index from {len(files)} corpus files...")
        if json_files:
            for f in json_files:
                try:
                    studies = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                for idx, s in enumerate(studies):
                    proto = s.get("protocolSection", {})
                    ident = proto.get("identificationModule", {}) or {}
                    nct_id = ident.get("nctId") or ident.get("NCTId")
                    if nct_id:
                        self._index[nct_id] = (f, idx, "json")
        else:
            for f in xml_files:
                nct_id = find_ctgov_xml_nct_id(f)
                if nct_id:
                    self._index[nct_id] = (f, -1, "xml")
        logger.info(f"Trial index built: {len(self._index)} trials indexed")
        self._write_cached_index(files)

    def __getitem__(self, nct_id: str) -> Trial:
        if nct_id in self._cache:
            return self._cache[nct_id]
        if nct_id not in self._index:
            raise KeyError(nct_id)
        page_file, pos, kind = self._index[nct_id]
        if kind == "xml":
            trial = parse_ctgov_xml_file(page_file)
        else:
            studies = json.loads(page_file.read_text(encoding="utf-8"))
            trial = parse_ctgov_study(studies[pos])
        if trial is None:
            raise KeyError(f"Could not parse trial {nct_id}")
        # Evict oldest if cache is full
        if len(self._cache) >= self._max_cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[nct_id] = trial
        return trial

    def get(self, nct_id: str, default: Trial | None = None) -> Trial | None:
        try:
            return self[nct_id]
        except KeyError:
            return default

    def __contains__(self, nct_id: object) -> bool:
        return nct_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)
