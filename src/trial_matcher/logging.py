"""Loguru-based structured logging with run_id propagation."""

from __future__ import annotations

import sys
import uuid
from contextvars import ContextVar
from typing import Any

from loguru import logger as _logger

from .config import get_settings

_run_id_var: ContextVar[str] = ContextVar("run_id", default="-")


def setup_logging(level: str | None = None) -> None:
    """Configure loguru with a structured format and run_id propagation.

    Idempotent: safe to call multiple times.
    """
    settings = get_settings()
    _logger.remove()

    def _patcher(record: dict[str, Any]) -> None:
        record["extra"]["run_id"] = _run_id_var.get()

    _logger.configure(patcher=_patcher)

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{extra[run_id]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    _logger.add(
        sys.stderr,
        level=(level or settings.log_level).upper(),
        format=fmt,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )


def set_run_id(run_id: str | None = None) -> str:
    """Set the run_id for the current context. Returns the value used."""
    rid = run_id or uuid.uuid4().hex[:8]
    _run_id_var.set(rid)
    return rid


def get_run_id() -> str:
    return _run_id_var.get()


# Re-export the configured logger
logger = _logger
