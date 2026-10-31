"""Which LLM a tenant's work runs on: Standard AI or Premium AI (BYOK).

* **Standard AI** is the server's own free LLM from `.env` (Groq or Ollama):
  zero cost to the platform, shared by every standard tenant, and slow at
  volume. Groq's free tier allows about 1,000 calls a day.
* **Premium AI** is the tenant's own OpenAI or Anthropic key. The provider
  bills the tenant, the daily cap goes away, and Tier 3 on a 3,000-entity
  job takes hours instead of days.

What Premium deliberately does *not* change is web search. DuckDuckGo bans
by IP address, and every tenant's searches leave from the same server, so
the 2-second spacing (app/enrichment/throttle.py) protects all tenants from
one tenant's job. A faster search tier would need its own paid search key;
an LLM key does not make DuckDuckGo more tolerant.

Resolved when a task runs, not when the job is created: a key is never
copied into a job row or a task payload, where it would outlive a rotation
and appear in dead letters.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.secrets import decrypt_secret
from app.enrichment.sources import Provider
from app.models.tenant_settings import TenantAiSettings

PremiumProvider = Literal["openai", "anthropic", "cerebras"]


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    label: str
    base_url: str
    #: Suggested in the Settings screen. Any model works if it passes the
    #: known-answer check on save; it must answer without "thinking", because
    #: Tier 2 caps its YES/NO reply at 3 tokens.
    default_model: str
    #: The request format it speaks (see `Provider` in sources.py).
    wire: Provider = "openai_compatible"
    #: A free tier exists, so a key can cost nothing.
    free: bool = False
    #: One line for the Settings screen: what to expect.
    note: str = ""
    key_url: str = ""
    key_placeholder: str = "sk-…"
    #: Extra request fields this provider needs to behave as the prompts expect.
    extra: tuple[tuple[str, Any], ...] = ()


PROVIDERS: dict[PremiumProvider, ProviderInfo] = {
    "openai": ProviderInfo(
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4.1-mini",
        wire="openai",
        note="Paid: billed to your OpenAI account.",
        key_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderInfo(
        "Anthropic",
        "https://api.anthropic.com/v1",
        "claude-haiku-5-5",
        wire="anthropic",
        note="Paid: billed to your Anthropic account.",
        key_url="https://console.anthropic.com/settings/keys",
        key_placeholder="sk-ant-…",
    ),
    "cerebras": ProviderInfo(
        "Cerebras",
        "https://api.cerebras.ai/v1",
        "qwen-3.8-27b",
        free=True,
        note=(
            "Free trial: 1M tokens a day, 5 requests a minute, and $5 of credit that "
            "expires 30 days after it is granted."
        ),
        key_url="https://cloud.cerebras.ai",
        key_placeholder="csk-…",
        # Qwen on Cerebras reasons by default (reasoning_effort "high"), and a
        # reasoning model spends the 3-token YES/NO budget thinking and replies
        # with nothing. "none" turns that off.
        extra=(("reasoning_effort", "none"),),
    ),
}


@dataclass(frozen=True, slots=True)
class LlmConfig:
    provider: Provider
    base_url: str
    model: str
    premium: bool
    #: Excluded from repr so a config can be logged without leaking the key.
    api_key: str | None = field(default=None, repr=False)
    #: Who the key belongs to ("cerebras"); `provider` is only the wire format.
    vendor: str = ""
    extra: tuple[tuple[str, Any], ...] = ()
    #: See LlmClient.reply_tokens.
    reply_tokens: int = 3

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "provider": self.provider,
            "extra_body": dict(self.extra),
            "reply_tokens": self.reply_tokens,
        }

    @property
    def audit_label(self) -> str:
        """Recorded on every machine decision: which tier and model made it."""
        tier = "premium" if self.premium else "standard"
        return f"{tier}:{self.vendor or self.provider}/{self.model}"

    @property
    def display_name(self) -> str:
        """Who is answering, for a progress line: "Cerebras", or "Standard AI"."""
        if not self.premium:
            return "Standard AI"
        info = PROVIDERS.get(self.vendor)  # type: ignore[call-overload]
        return info.label if info else (self.vendor or "Your key")


#: Models that reason before answering, and what they need to behave as the
#: prompts expect: reasoning kept short, and room in the reply budget for it.
#: Measured on Groq: gpt-oss at "low" answers a judgment correctly in ~525
#: tokens in total, about 25 of them reasoning.
_REASONING_MODELS: tuple[tuple[str, tuple[tuple[str, Any], ...], int], ...] = (
    ("openai/gpt-oss", (("reasoning_effort", "low"),), 400),
)


def _quirks(model: str) -> tuple[tuple[tuple[str, Any], ...], int]:
    for prefix, extra, reply_tokens in _REASONING_MODELS:
        if model.startswith(prefix):
            return extra, reply_tokens
    return (), 3


def _server_config(settings: Settings, model: str) -> LlmConfig:
    extra, reply_tokens = _quirks(model)
    return LlmConfig(
        provider="openai_compatible",
        base_url=str(settings.llm_base_url),
        model=model,
        api_key=settings.llm_api_key,
        premium=False,
        extra=extra,
        reply_tokens=reply_tokens,
    )


def standard_config(settings: Settings | None = None) -> LlmConfig | None:
    settings = settings or get_settings()
    if not settings.llm_base_url:
        return None
    return _server_config(settings, settings.llm_model)


def standard_pool(settings: Settings | None = None) -> list[LlmConfig]:
    """The server's own models, in the order to use them: `LLM_MODEL` first,
    then `LLM_POOL_MODELS`. Each is a separate daily budget on providers that
    cap per model, so the task moves to the next when one runs out."""
    settings = settings or get_settings()
    if not settings.llm_base_url:
        return []
    return [_server_config(settings, m) for m in [settings.llm_model, *settings.llm_pool_list]]


def premium_config(provider: PremiumProvider, model: str, api_key: str) -> LlmConfig:
    info = PROVIDERS[provider]
    return LlmConfig(
        provider=info.wire,
        base_url=info.base_url,
        model=model,
        api_key=api_key,
        premium=True,
        vendor=provider,
        extra=info.extra,
    )


def config_for(row: TenantAiSettings | None, settings: Settings | None = None) -> LlmConfig | None:
    """The tenant's LLM, or None when no LLM is available to them at all.

    Raises SecretsUnavailableError (permanent, with a message saying so)
    when a premium key exists but cannot be decrypted. Falling back to the
    free tier there would silently move a paying tenant's data to another
    provider; failing tells them to re-enter the key.
    """
    if row is not None and row.mode == "premium":
        assert row.provider in PROVIDERS and row.model and row.api_key_ciphertext
        return premium_config(
            row.provider,
            row.model,
            decrypt_secret(row.api_key_ciphertext),
        )
    return standard_config(settings)


def llm_available(row: TenantAiSettings | None, settings: Settings | None = None) -> bool:
    """Whether the tenant has any LLM, without decrypting anything."""
    if row is not None and row.mode == "premium":
        return True
    return bool((settings or get_settings()).llm_base_url)


def load_settings_row(session: Any, tenant_id: uuid.UUID | str) -> TenantAiSettings | None:
    return session.execute(
        select(TenantAiSettings).where(TenantAiSettings.tenant_id == uuid.UUID(str(tenant_id)))
    ).scalar_one_or_none()


async def load_settings_row_async(db: Any, tenant_id: uuid.UUID) -> TenantAiSettings | None:
    return (
        await db.execute(select(TenantAiSettings).where(TenantAiSettings.tenant_id == tenant_id))
    ).scalar_one_or_none()


def resolve_llm_config(session: Any, tenant_id: uuid.UUID | str) -> LlmConfig | None:
    return config_for(load_settings_row(session, tenant_id))


def resolve_llm_pool(session: Any, tenant_id: uuid.UUID | str) -> list[LlmConfig]:
    """Every LLM the tenant's work may use, best first. A tenant's own key is
    always alone (it has its own limits and its own bill); Standard AI is the
    server's whole pool."""
    row = load_settings_row(session, tenant_id)
    if row is not None and row.mode == "premium":
        config = config_for(row)
        return [config] if config else []
    return standard_pool()
