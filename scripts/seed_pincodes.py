#!/usr/bin/env python
"""Bulk-seed `reference_pincodes` from the All-India Pincode CSV.

    python scripts/seed_pincodes.py --csv-path all_india_pincodes.csv

Streams the file straight into Postgres via `COPY FROM STDIN` in batches of
10,000 rows (`--batch-size` to change it) — no ORM object construction, no
per-row round trip. Column names from the government portal vary across
releases ("Pincode" vs "pincode", "officename" vs "office_name", ...); see
`_COLUMN_ALIASES` below for what is recognised. A row that is missing a
required field, or whose pincode is not exactly six digits, is skipped and
counted rather than aborting the run — large government exports reliably
have a handful of these.

Connects as the schema owner by default (`MIGRATION_DATABASE_URL`), the same
role Alembic uses, since this is a bulk administrative load rather than
request-path traffic. `reference_pincodes` carries no RLS policy (it is a
global, un-tenanted lookup table — see app/models/reference_pincode.py), so
`DATABASE_URL` would work too; the owner DSN is just the more conventional
choice for an out-of-band data load. Override either with --database-url.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import click
import psycopg

logger = logging.getLogger("seed_pincodes")

# --- Column mapping -------------------------------------------------------------
# Keys are normalised (lower-cased, with anything but letters/digits
# stripped) before lookup, so "Office Name", "OfficeName" and "office_name"
# all resolve to the same target column.
_COLUMN_ALIASES: dict[str, str] = {
    "pincode": "pincode",
    "pin": "pincode",
    "pincodenumber": "pincode",
    "officename": "office_name",
    "office": "office_name",
    "postofficename": "office_name",
    "districtname": "district_name",
    "district": "district_name",
    "statename": "state_name",
    "state": "state_name",
    "taluk": "taluk",
    "taluka": "taluk",
    "block": "taluk",
    "deliverystatus": "delivery_status",
    "delivery": "delivery_status",
}

REQUIRED_TARGET_COLUMNS = ("pincode", "office_name", "district_name", "state_name")
COPY_COLUMNS = (
    "pincode",
    "office_name",
    "district_name",
    "state_name",
    "taluk",
    "delivery_status",
)

#: Values a source file uses in place of a real PIN. Six digits, never real.
_PLACEHOLDER_PINCODES = frozenset(
    {
        "000000",
        "111111",
        "222222",
        "333333",
        "444444",
        "555555",
        "666666",
        "777777",
        "888888",
        "999999",
    }
)


def _normalize_header(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def _build_column_map(fieldnames: list[str]) -> dict[str, str]:
    """Source CSV column -> canonical target column.

    Raises before reading a single data row if a required target has no
    source column mapped to it — better than silently seeding every row with
    an empty district.
    """
    mapping: dict[str, str] = {}
    seen_targets: set[str] = set()
    for source in fieldnames:
        canonical = _COLUMN_ALIASES.get(_normalize_header(source))
        if canonical is None or canonical in seen_targets:
            continue
        mapping[source] = canonical
        seen_targets.add(canonical)

    missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in seen_targets]
    if missing:
        raise click.ClickException(
            f"CSV is missing required column(s): {', '.join(missing)}. Header seen: {fieldnames}"
        )
    return mapping


@dataclass
class SeedStats:
    total_rows_read: int = 0
    rows_seeded: int = 0
    rows_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        self.rows_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def _clean_row(
    raw: dict[str, str | None], column_map: dict[str, str], stats: SeedStats, line_no: int
) -> tuple[str, str, str, str, str | None, str | None] | None:
    record: dict[str, str | None] = {}
    for source, canonical in column_map.items():
        value = raw.get(source)
        record[canonical] = value.strip() if isinstance(value, str) else None

    pincode = record.get("pincode") or ""
    if not (pincode.isdigit() and len(pincode) == 6):
        logger.debug("line %d: malformed pincode %r", line_no, pincode)
        stats.record_skip("invalid_pincode")
        return None
    if pincode in _PLACEHOLDER_PINCODES:
        logger.debug("line %d: placeholder pincode %r", line_no, pincode)
        stats.record_skip("placeholder_pincode")
        return None

    office_name = record.get("office_name") or ""
    district_name = record.get("district_name") or ""
    state_name = record.get("state_name") or ""
    if not (office_name and district_name and state_name):
        logger.debug("line %d: missing required field(s)", line_no)
        stats.record_skip("missing_required_field")
        return None

    taluk = record.get("taluk") or None
    delivery_status = record.get("delivery_status") or None

    return (pincode, office_name, district_name, state_name, taluk, delivery_status)


def _iter_batches(
    csv_path: Path, batch_size: int, stats: SeedStats
) -> Iterator[list[tuple[str, str, str, str, str | None, str | None]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise click.ClickException("CSV has no header row")
        column_map = _build_column_map(list(reader.fieldnames))

        batch: list[tuple[str, str, str, str, str | None, str | None]] = []
        for line_no, raw in enumerate(reader, start=2):  # header occupies line 1
            stats.total_rows_read += 1
            try:
                cleaned = _clean_row(raw, column_map, stats, line_no)
            except Exception:
                # A malformed row must never take the whole load down with it.
                logger.exception("line %d: unexpected error parsing row, skipping", line_no)
                stats.record_skip("unexpected_error")
                continue
            if cleaned is None:
                continue
            batch.append(cleaned)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _seed(conn: psycopg.Connection, csv_path: Path, batch_size: int, truncate: bool) -> SeedStats:
    stats = SeedStats()
    copy_sql = f"COPY reference_pincodes ({', '.join(COPY_COLUMNS)}) FROM STDIN"

    with conn.cursor() as cur:
        if truncate:
            cur.execute("TRUNCATE TABLE reference_pincodes")
            conn.commit()

        for batch in _iter_batches(csv_path, batch_size, stats):
            with cur.copy(copy_sql) as copy:
                for row in batch:
                    copy.write_row(row)
            conn.commit()
            stats.rows_seeded += len(batch)
            logger.info(
                "seeded batch of %d rows (running total: %d)", len(batch), stats.rows_seeded
            )

    return stats


def _resolve_dsn(database_url: str | None) -> str:
    dsn = database_url or os.environ.get("MIGRATION_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise click.ClickException(
            "No database URL. Pass --database-url or set MIGRATION_DATABASE_URL "
            "(preferred) / DATABASE_URL in the environment."
        )
    # SQLAlchemy-style "postgresql+psycopg://" DSNs are used throughout this
    # project's settings (app/core/config.py); psycopg.connect() wants plain
    # libpq conninfo, so the driver suffix has to come off.
    return dsn.replace("postgresql+psycopg://", "postgresql://")


@click.command()
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the All-India Pincode CSV.",
)
@click.option(
    "--database-url",
    default=None,
    help="Postgres DSN. Defaults to $MIGRATION_DATABASE_URL, then $DATABASE_URL.",
)
@click.option(
    "--batch-size",
    default=10_000,
    show_default=True,
    type=int,
    help="Rows per COPY batch / commit.",
)
@click.option(
    "--truncate",
    is_flag=True,
    default=False,
    help="TRUNCATE reference_pincodes before seeding (fresh load, not an upsert).",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Log each skipped row's reason.")
def main(
    csv_path: Path, database_url: str | None, batch_size: int, truncate: bool, verbose: bool
) -> None:
    """Bulk-load the All-India Pincode reference dataset into Postgres."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    dsn = _resolve_dsn(database_url)

    started = time.monotonic()
    with psycopg.connect(dsn) as conn:
        stats = _seed(conn, csv_path, batch_size, truncate)
    elapsed = time.monotonic() - started

    click.echo("")
    click.echo("=== reference_pincodes seed summary ===")
    click.echo(f"  Rows read:    {stats.total_rows_read}")
    click.echo(f"  Rows seeded:  {stats.rows_seeded}")
    click.echo(f"  Rows skipped: {stats.rows_skipped}")
    for reason, count in sorted(stats.skip_reasons.items()):
        click.echo(f"    - {reason}: {count}")
    click.echo(f"  Elapsed:      {elapsed:.2f}s")
    if stats.rows_seeded and elapsed > 0:
        click.echo(f"  Rate:         {stats.rows_seeded / elapsed:.0f} rows/sec")


if __name__ == "__main__":
    main()
