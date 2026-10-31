"""Where a job's ontology comes from: a shipped YAML file or a tenant's own.

Shipped domains (education, healthcare, corporate, generic) are files under
`ontologies/`. Custom domains (`custom_*`) were generated from a tenant's
description, confirmed by them, and stored in `tenant_ontologies`. Every
caller that loads "the ontology of this job" goes through here, so a custom
domain works in every stage without the stages knowing the difference.

Custom specs are immutable once saved, so a compiled engine can be cached
for the life of the process. The cache is keyed by tenant as well as name:
names carry a random suffix, but isolation should not rest on that.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from app.models.tenant_settings import TenantOntology
from app.ontology.engine import OntologyEngine, OntologyError, OntologySpec, available_ontologies

CUSTOM_PREFIX = "custom_"

_compiled: dict[tuple[str, str], OntologyEngine] = {}


def is_custom(name: str) -> bool:
    return name.startswith(CUSTOM_PREFIX)


def engine_from_spec(spec: dict[str, Any]) -> OntologyEngine:
    """Validate and compile a stored spec. Raises OntologyError if invalid."""
    try:
        return OntologyEngine(OntologySpec.model_validate(spec))
    except ValueError as exc:
        raise OntologyError(f"custom ontology {spec.get('name')!r}: {exc}") from exc


def _cached(tenant_id: uuid.UUID | str, name: str, spec: dict[str, Any] | None) -> OntologyEngine:
    key = (str(tenant_id), name)
    engine = _compiled.get(key)
    if engine is None:
        if spec is None:
            raise OntologyError(f"Data domain {name!r} does not exist for this tenant")
        engine = _compiled[key] = engine_from_spec(spec)
    return engine


def load_ontology(session: Any, name: str, tenant_id: uuid.UUID | str) -> OntologyEngine:
    """The compiled ontology a job names, for a worker's sync session."""
    if not is_custom(name):
        return OntologyEngine.load(name)
    if (str(tenant_id), name) in _compiled:
        return _compiled[(str(tenant_id), name)]
    spec = session.execute(
        select(TenantOntology.spec).where(
            TenantOntology.tenant_id == uuid.UUID(str(tenant_id)), TenantOntology.name == name
        )
    ).scalar_one_or_none()
    return _cached(tenant_id, name, spec)


async def load_ontology_async(db: Any, name: str, tenant_id: uuid.UUID) -> OntologyEngine:
    """The same, for an API request's async session."""
    if not is_custom(name):
        return OntologyEngine.load(name)
    if (str(tenant_id), name) in _compiled:
        return _compiled[(str(tenant_id), name)]
    spec = (
        await db.execute(
            select(TenantOntology.spec).where(
                TenantOntology.tenant_id == tenant_id, TenantOntology.name == name
            )
        )
    ).scalar_one_or_none()
    return _cached(tenant_id, name, spec)


async def ontology_exists(db: Any, name: str, tenant_id: uuid.UUID) -> bool:
    if not is_custom(name):
        return name in available_ontologies()
    found = (
        await db.execute(
            select(TenantOntology.id).where(
                TenantOntology.tenant_id == tenant_id, TenantOntology.name == name
            )
        )
    ).scalar_one_or_none()
    return found is not None
