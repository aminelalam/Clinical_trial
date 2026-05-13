"""Filter a stream of Trial objects to those usable by the matcher."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date

from ..models.trial import Trial


def filter_corpus(
    trials: Iterable[Trial],
    *,
    require_interventional: bool = True,
    require_eligibility: bool = True,
    min_last_update: date | None = date(2018, 1, 1),
) -> Iterator[Trial]:
    """Yield only trials that should enter the matcher's corpus.

    Defaults follow the spec: interventional + non-empty eligibility text
    + last update on or after 2018-01-01 (cuts ~50% of the snapshot).
    """
    for t in trials:
        if require_interventional and not t.interventional:
            continue
        if require_eligibility and not (t.eligibility.raw_text or "").strip():
            continue
        if (
            min_last_update is not None
            and t.last_update_date is not None
            and t.last_update_date < min_last_update
        ):
            continue
        yield t
