"""Every cascading foreign key must be indexed on its referencing column.

Postgres does not index them automatically. Deleting a parent row runs one
internal query per child table per parent row, so an unindexed child makes a
job delete quadratic: deleting one 51,668-record job read some tables 51,668
times over and ran for most of an hour (migration 0009).
"""

from __future__ import annotations

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

#: Parents whose deletion cascades through the pipeline's tables.
CASCADE_PARENTS = ("jobs", "school_records", "clusters")


def test_every_foreign_key_into_the_cascade_parents_is_indexed(owner_engine: Engine) -> None:
    with Session(owner_engine) as session:
        unindexed = session.execute(
            text(
                """
                SELECT c.conrelid::regclass::text AS child, a.attname AS col,
                       c.confrelid::regclass::text AS parent
                  FROM pg_constraint c
                  JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
                 WHERE c.contype = 'f'
                   AND c.confrelid::regclass::text = ANY(:parents)
                   AND NOT EXISTS (
                         SELECT 1 FROM pg_index i
                          WHERE i.indrelid = c.conrelid
                            AND (i.indkey::int2[])[0] = c.conkey[1])
                 ORDER BY 1, 2
                """
            ),
            {"parents": list(CASCADE_PARENTS)},
        ).all()

    assert not unindexed, (
        "Foreign keys with no index on their referencing column (each makes a "
        f"parent delete scan the child table once per row): {unindexed}"
    )
