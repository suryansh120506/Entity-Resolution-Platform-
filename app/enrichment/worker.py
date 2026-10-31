"""EnrichmentWorker: natural key -> registry scrape, else SERP + LLM -> cache.

Runs between Silver and Gold, beside the matching chain rather than in it.
Work is keyed by *natural key*, not by record: a job with 900 rows naming
300 distinct affiliation numbers makes at most 300 lookups, and a key
already enriched by an earlier job (and not expired) makes none.

Each key is committed as it completes, so a crash or a rate-limit deferral
resumes from the cache rather than from zero.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.enrichment.sources import (
    LlmExtractor,
    Outcome,
    PoliteFetcher,
    SearchFn,
    registry_lookup,
    serp_llm_lookup,
)
from app.models.medallion import EnrichmentResult, SilverRecord
from app.ontology.engine import EnrichmentRules

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KeyWork:
    key_type: str
    key_value: str
    name: str
    district: str
    state: str


class EnrichmentWorker:
    def __init__(
        self,
        *,
        rules: EnrichmentRules,
        fetcher: PoliteFetcher | None,
        search: SearchFn | None,
        llm: LlmExtractor | None,
    ) -> None:
        self.rules = rules
        self.fetcher = fetcher
        self.search = search
        self.llm = llm

    @property
    def serp_enabled(self) -> bool:
        serp = self.rules.serp
        return bool(serp and serp.enabled and self.search and self.llm)

    def registries_for(self, key_type: str) -> list[Any]:
        if self.fetcher is None:
            return []
        return [r for r in self.rules.registries if r.enabled and r.key == key_type]

    def run(self, session: Session, *, tenant_id: str, job_id: str) -> dict[str, int]:
        stats = {"keys": 0, "cached": 0, "found": 0, "not_found": 0, "blocked": 0, "error": 0}
        for work in self._keys(session, job_id):
            stats["keys"] += 1
            strategies = self._plan(work)
            if not strategies:
                continue
            fresh = self._fresh_strategies(session, tenant_id, work)
            for strategy, lookup in strategies:
                if strategy in fresh:
                    stats["cached"] += 1
                    if fresh[strategy] == "found":
                        break
                    continue
                outcome = lookup()
                self._persist(session, tenant_id, work, strategy, outcome)
                session.commit()
                stats[outcome.status] += 1
                if outcome.status == "found":
                    break
        return stats

    # --- planning ---------------------------------------------------------------

    def _plan(self, work: KeyWork) -> list[tuple[str, Callable[[], Outcome]]]:
        """Registry first (authoritative), SERP+LLM only as the fallback."""
        plan: list[tuple[str, Callable[[], Outcome]]] = []
        if self.fetcher is not None:
            for adapter in self.registries_for(work.key_type):
                plan.append(
                    ("registry", partial(registry_lookup, self.fetcher, adapter, work.key_value))
                )
        serp = self.rules.serp
        if self.serp_enabled and serp is not None and self.search and self.llm:
            query = serp.query.format(name=work.name, district=work.district, state=work.state)
            plan.append(
                (
                    "serp_llm",
                    partial(
                        serp_llm_lookup,
                        self.search,
                        self.llm,
                        query=" ".join(query.split()),
                        subject=(
                            f"{work.name}, {work.district}, {work.state} "
                            f"({work.key_type} {work.key_value})"
                        ),
                        fields=serp.fields,
                        max_results=serp.max_results,
                    ),
                )
            )
        return plan

    @staticmethod
    def _keys(session: Session, job_id: str) -> Iterator[KeyWork]:
        """One KeyWork per distinct (key_type, key_value), first record wins."""
        seen: set[tuple[str, str]] = set()
        rows = session.execute(
            select(
                SilverRecord.natural_keys,
                SilverRecord.name,
                SilverRecord.district,
                SilverRecord.state,
            )
            .where(SilverRecord.job_id == job_id, SilverRecord.natural_keys != {})
            .order_by(SilverRecord.record_id)
        ).all()
        for natural_keys, name, district, state in rows:
            for key_type, key_value in sorted((natural_keys or {}).items()):
                if (key_type, key_value) in seen:
                    continue
                seen.add((key_type, key_value))
                yield KeyWork(key_type, str(key_value), name or "", district or "", state or "")

    @staticmethod
    def _fresh_strategies(session: Session, tenant_id: str, work: KeyWork) -> dict[str, str]:
        rows = session.execute(
            select(EnrichmentResult.strategy, EnrichmentResult.status).where(
                EnrichmentResult.tenant_id == uuid.UUID(tenant_id),
                EnrichmentResult.key_type == work.key_type,
                EnrichmentResult.key_value == work.key_value,
                EnrichmentResult.expires_at > datetime.now(UTC),
            )
        ).all()
        return {strategy: status for strategy, status in rows}

    def _persist(
        self, session: Session, tenant_id: str, work: KeyWork, strategy: str, outcome: Outcome
    ) -> None:
        now = datetime.now(UTC)
        ttl = self.rules.ttl_days if outcome.status == "found" else self.rules.negative_ttl_days
        if outcome.status == "error":
            ttl = 1  # transient by nature; retry tomorrow, not next week
        fields: dict[str, Any] = {name: f.as_json() for name, f in outcome.fields.items()}
        if outcome.detail and outcome.status != "found":
            fields["_detail"] = outcome.detail
        stmt = insert(EnrichmentResult).values(
            tenant_id=tenant_id,
            key_type=work.key_type,
            key_value=work.key_value,
            strategy=strategy,
            source=outcome.source,
            status=outcome.status,
            fields=fields,
            fetched_at=now,
            expires_at=now + timedelta(days=ttl),
        )
        session.execute(
            stmt.on_conflict_do_update(
                constraint="uq_enrichment_results_key",
                set_={
                    "source": stmt.excluded.source,
                    "status": stmt.excluded.status,
                    "fields": stmt.excluded.fields,
                    "fetched_at": stmt.excluded.fetched_at,
                    "expires_at": stmt.excluded.expires_at,
                },
            )
        )
