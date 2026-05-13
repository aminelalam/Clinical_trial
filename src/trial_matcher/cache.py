"""Disk-backed cache for LLM calls, embeddings, and any deterministic computation.

Keys are SHA-256 of (namespace + canonical JSON of inputs). Values are JSON-serializable.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

import diskcache

from .config import get_settings

T = TypeVar("T")

_caches: dict[str, diskcache.Cache] = {}
_lock = threading.Lock()


def _canonical(value: Any) -> str:
    """Canonical JSON serialization — sorted keys, no whitespace, stable across runs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def cache_key(namespace: str, *parts: Any) -> str:
    """Compute a stable cache key from a namespace and arbitrary input parts."""
    payload = _canonical([namespace, *parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache(namespace: str = "default") -> diskcache.Cache:
    """Return a diskcache.Cache for the given namespace, creating directory if needed."""
    with _lock:
        if namespace not in _caches:
            base = get_settings().paths.cache_dir / namespace
            Path(base).mkdir(parents=True, exist_ok=True)
            _caches[namespace] = diskcache.Cache(str(base), size_limit=int(20e9))
        return _caches[namespace]


def cached_call(
    namespace: str,
    key_parts: tuple[Any, ...],
    compute: Callable[[], T],
    expire_seconds: int | None = None,
) -> T:
    """Memoize ``compute`` under (namespace, key_parts)."""
    cache = get_cache(namespace)
    key = cache_key(namespace, *key_parts)
    if key in cache:
        return cache[key]  # type: ignore[no-any-return]
    value = compute()
    cache.set(key, value, expire=expire_seconds)
    return value


def clear_namespace(namespace: str) -> int:
    """Clear all entries in a cache namespace. Returns number of items removed."""
    cache = get_cache(namespace)
    n = len(cache)
    cache.clear()
    return n
