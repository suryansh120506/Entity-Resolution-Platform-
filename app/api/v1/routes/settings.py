"""Tenant settings: the AI tier (Standard, free; or Premium, the tenant's key).

A Premium key is verified before it is stored: one known-answer call to the
provider (`LlmClient.check`). A wrong key, a retired model, or a reasoning
model that answers nothing within Tier 2's 3-token budget is refused here,
with the provider's reason, instead of failing a job hours later.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import DbDep, PrincipalDep, get_principal, require_role
from app.core.config import get_settings
from app.core.secrets import (
    SecretsUnavailableError,
    decrypt_secret,
    encrypt_secret,
    key_hint,
    scrub,
    secrets_available,
)
from app.core.security import Role
from app.enrichment.llm_config import (
    PROVIDERS,
    LlmConfig,
    load_settings_row_async,
    premium_config,
)
from app.enrichment.sources import LlmJudge, LlmUnusableError
from app.models.tenant_settings import TenantAiSettings
from app.queue.registry import RetryLaterError
from app.schemas.settings import AiSettingsUpdate, AiSettingsView, ProviderOption

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/settings", tags=["settings"], dependencies=[Depends(get_principal)])

_ADMIN = [Depends(require_role(Role.ORG_ADMIN))]
_VERIFY_TIMEOUT = 20.0


def _view(row: TenantAiSettings | None) -> AiSettingsView:
    settings = get_settings()
    premium = row is not None and row.mode == "premium"
    return AiSettingsView(
        mode="premium" if premium else "standard",
        provider=row.provider if row else None,  # type: ignore[arg-type]
        model=row.model if row else None,
        key_hint=row.key_hint if row else None,
        verified_at=row.verified_at if row else None,
        standard_model=settings.llm_model if settings.llm_base_url else None,
        premium_available=secrets_available(),
        providers=[
            ProviderOption(
                id=pid,
                label=info.label,
                default_model=info.default_model,
                free=info.free,
                note=info.note,
                key_url=info.key_url,
                key_placeholder=info.key_placeholder,
            )
            for pid, info in PROVIDERS.items()
        ],
    )


@router.get("/ai", response_model=AiSettingsView, summary="This tenant's AI tier")
async def get_ai_settings(db: DbDep, principal: PrincipalDep) -> AiSettingsView:
    return _view(await load_settings_row_async(db, principal.tenant_id))


def _verify(config: LlmConfig) -> None:
    with httpx.Client(timeout=_VERIFY_TIMEOUT) as client:
        LlmJudge(client, **config.client_kwargs()).check()


@router.put(
    "/ai",
    response_model=AiSettingsView,
    summary="Choose Standard AI, or Premium AI with your own key",
    dependencies=_ADMIN,
)
async def put_ai_settings(
    db: DbDep, principal: PrincipalDep, payload: AiSettingsUpdate
) -> AiSettingsView:
    row = await load_settings_row_async(db, principal.tenant_id)
    if row is None:
        row = TenantAiSettings(tenant_id=principal.tenant_id, mode="standard")
        db.add(row)

    if payload.mode == "standard":
        # The stored key is kept, so switching back to Premium needs no re-entry.
        row.mode = "standard"
    else:
        if not secrets_available():
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="Premium AI is unavailable: the server has no SECRETS_ENCRYPTION_KEY.",
            )
        provider = payload.provider or row.provider
        if provider not in PROVIDERS:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Choose a provider.")
        model = (payload.model or "").strip() or (
            row.model if row.provider == provider and row.model else None
        )
        model = model or PROVIDERS[provider].default_model

        if payload.api_key is not None:
            key = payload.api_key.get_secret_value().strip()
        elif row.api_key_ciphertext and row.provider == provider:
            try:
                key = decrypt_secret(row.api_key_ciphertext)
            except SecretsUnavailableError as exc:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        else:
            key = ""
        if not key:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Enter your {PROVIDERS[provider].label} API key.",
            )
        if not 10 <= len(key) <= 400 or any(ch.isspace() for ch in key):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail="That does not look like an API key."
            )

        config = premium_config(provider, model, key)
        try:
            await run_in_threadpool(_verify, config)
        except LlmUnusableError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The key was not accepted: {scrub(str(exc), key)}",
            ) from exc
        except RetryLaterError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The provider is rate-limiting this key right now. Try again in a minute.",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach the provider: {type(exc).__name__}",
            ) from exc

        row.mode = "premium"
        row.provider = provider
        row.model = model
        row.api_key_ciphertext = encrypt_secret(key)
        row.key_hint = key_hint(key)
        row.verified_at = datetime.now(UTC)

    row.updated_by = principal.subject
    row.updated_at = datetime.now(UTC)
    await db.commit()
    logger.info(
        "Tenant %s AI tier set to %s (%s/%s) by %s",
        principal.tenant_id,
        row.mode,
        row.provider,
        row.model,
        principal.subject,
    )
    return _view(row)


@router.delete(
    "/ai/key",
    response_model=AiSettingsView,
    summary="Forget the stored key and return to Standard AI",
    dependencies=_ADMIN,
)
async def delete_ai_key(db: DbDep, principal: PrincipalDep) -> AiSettingsView:
    row = await load_settings_row_async(db, principal.tenant_id)
    if row is not None:
        row.mode = "standard"
        row.provider = row.model = row.api_key_ciphertext = row.key_hint = None
        row.verified_at = None
        row.updated_by = principal.subject
        row.updated_at = datetime.now(UTC)
        await db.commit()
    return _view(row)
