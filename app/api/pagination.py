"""Keyset cursors for the review queue — TRD 2.5 / FIG 2.9.

TRD 2.5 mandates keyset pagination rather than OFFSET, and the reason shows
up precisely in this queue: stewards are resolving clusters *while* paging
through them. Under OFFSET, every cluster resolved out of the filtered set
shifts the remaining rows left, so page 2 silently skips the rows that moved
into page 1's window. A reviewer working a backlog would never see them, and
nothing anywhere would report a problem.

A keyset cursor is positional in the data rather than in the result set, so
rows leaving the filter cannot move the boundary.

The cursor encodes `(created_at, id)` — the trailing id is what makes the
sort total. Two clusters created in the same microsecond with an ordering by
timestamp alone can repeat or skip across a page boundary, and the composite
index `ix_clusters_review_queue` is declared on exactly this pair so the page
is an index range scan.

It is base64 and documented as opaque. The moment a client parses it, the
sort key cannot change without breaking them.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime
from typing import NamedTuple


class Cursor(NamedTuple):
    """Decoded keyset position."""

    created_at: datetime
    id: uuid.UUID


class InvalidCursorError(ValueError):
    """The cursor was not one we issued. Mapped to 400 by the route."""


def encode_cursor(created_at: datetime, row_id: uuid.UUID) -> str:
    """Opaque cursor for the last row of a page."""
    payload = json.dumps({"created_at": created_at.isoformat(), "id": str(row_id)})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(raw: str) -> Cursor:
    """Parse a cursor we issued, or raise `InvalidCursorError`.

    Every failure mode collapses to one error: a cursor is either ours and
    parseable or it is not, and the distinction between "bad base64" and
    "missing key" is of interest only to someone probing the endpoint.
    """
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw.encode()))
        return Cursor(
            created_at=datetime.fromisoformat(payload["created_at"]),
            id=uuid.UUID(payload["id"]),
        )
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidCursorError("Malformed cursor") from exc
