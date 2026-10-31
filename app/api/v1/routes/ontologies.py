"""Custom data domains: draft an ontology from a description, then save it.

Two calls, with the user in between: `POST /generate` returns a draft and
the notes on what was removed from the LLM's proposal; the user reviews and
edits it; `POST /` saves it. Saving re-sanitises and re-validates the draft:
the browser is no more trusted than the LLM.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from app.api.deps import DbDep, PrincipalDep, get_principal
from app.core.secrets import SecretsUnavailableError
from app.enrichment.llm_config import LlmConfig, config_for, load_settings_row_async
from app.enrichment.sources import LlmClient, LlmUnusableError
from app.enrichment.throttle import patiently
from app.models.tenant_settings import TenantOntology
from app.ontology.engine import OntologyError
from app.ontology.generator import build_spec, draft_with_llm, new_ontology_name, sanitize_draft
from app.ontology.registry import engine_from_spec
from app.queue.registry import RetryLaterError
from app.schemas.jobs import OntologyOption
from app.schemas.ontologies import DomainDraftResponse, GenerateDomainRequest, SaveDomainRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/ontologies", tags=["ontologies"], dependencies=[Depends(get_principal)]
)

_DRAFT_TIMEOUT = 90.0


def _draft(config: LlmConfig, payload: GenerateDomainRequest) -> dict:
    with httpx.Client(timeout=_DRAFT_TIMEOUT) as client:
        llm = LlmClient(client, **config.client_kwargs())
        # On Standard AI the free tier's per-minute budget is shared with
        # running jobs, so a 429 is routine. Short waits happen here, up to
        # about a minute in all, rather than as an error in the user's face.
        return patiently(
            lambda: draft_with_llm(llm, payload.description, payload.headers, payload.sample_rows),
            attempts=4,
            max_pause=20.0,
            what="domain draft",
        )


@router.post(
    "/generate",
    response_model=DomainDraftResponse,
    summary="Draft a data domain from a description, with the active AI tier",
)
async def generate_domain(
    db: DbDep, principal: PrincipalDep, payload: GenerateDomainRequest
) -> DomainDraftResponse:
    try:
        config = config_for(await load_settings_row_async(db, principal.tenant_id))
    except SecretsUnavailableError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if config is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="No AI is configured: set LLM_BASE_URL on the server, or add a Premium key "
            "in Settings.",
        )

    try:
        raw = await run_in_threadpool(_draft, config, payload)
    except LlmUnusableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except RetryLaterError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI provider is rate-limiting requests. Try again in a minute.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"The AI provider could not be reached: {type(exc).__name__}",
        ) from exc
    if not raw:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="The AI returned no usable draft. Try again."
        )

    draft, notes = sanitize_draft(
        raw, description=payload.description, sample_rows=payload.sample_rows
    )
    return DomainDraftResponse(draft=draft, notes=notes, drafted_by=config.audit_label)


@router.post(
    "",
    response_model=OntologyOption,
    status_code=status.HTTP_201_CREATED,
    summary="Save a reviewed draft as one of this tenant's data domains",
)
async def save_domain(
    db: DbDep, principal: PrincipalDep, payload: SaveDomainRequest
) -> OntologyOption:
    draft, notes = sanitize_draft(
        payload.draft.model_dump(), description=payload.prompt, sample_rows=payload.sample_rows
    )
    name = new_ontology_name(draft.label)
    try:
        spec = build_spec(draft, name)
    except OntologyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    db.add(
        TenantOntology(
            tenant_id=principal.tenant_id,
            name=name,
            label=draft.label,
            description=draft.description,
            spec=spec,
            prompt=payload.prompt,
            created_by=principal.subject,
        )
    )
    await db.commit()
    logger.info("Tenant %s saved data domain %s (%s)", principal.tenant_id, name, notes or "clean")
    return custom_option(spec)


@router.get("", response_model=list[OntologyOption], summary="This tenant's custom data domains")
async def list_domains(db: DbDep, principal: PrincipalDep) -> list[OntologyOption]:
    rows = (
        await db.execute(
            select(TenantOntology)
            .where(TenantOntology.tenant_id == principal.tenant_id)
            .order_by(TenantOntology.created_at.desc())
        )
    ).scalars()
    return [custom_option(r.spec) for r in rows]


def custom_option(spec: dict) -> OntologyOption:
    return OntologyOption.from_engine(engine_from_spec(spec), custom=True)
