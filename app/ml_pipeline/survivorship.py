"""Survivor selection. Dependency-free: used by the resolution worker and by
the feedback API route, which must not import task modules (SAD 3.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def select_master_record(members: Sequence[tuple[Any, Mapping[str, Any]]]) -> Any | None:
    """The most complete record (most non-empty raw fields) survives.

    The surviving row is what the tenant exports (FR-09), and the fullest
    record loses the least. Ties break on record id, so the same members
    always yield the same master — a re-run or a redelivery cannot silently
    change which row a downstream consumer sees as canonical.
    """
    if not members:
        return None

    def completeness(item: tuple[Any, Mapping[str, Any]]) -> tuple[int, str]:
        record_id, raw_row = item
        filled = sum(1 for value in raw_row.values() if value not in (None, ""))
        return (filled, str(record_id))

    return max(members, key=completeness)[0]
