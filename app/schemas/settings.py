"""Request/response models for tenant settings (the AI tier).

The API key travels in one direction only: in, on a PUT. No response model
has a field that could carry it back, and it is a `SecretStr` with no
validators of its own, so neither a repr nor a 422 body (FastAPI echoes the
offending input) can expose it. Its checks run in the route, with messages
that never quote it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

ProviderId = Literal["openai", "anthropic", "cerebras"]


class ProviderOption(BaseModel):
    id: ProviderId
    label: str
    default_model: str
    #: A free tier exists: a key can cost nothing.
    free: bool = False
    note: str = ""
    key_url: str = ""
    key_placeholder: str = "sk-…"


class AiSettingsView(BaseModel):
    mode: Literal["standard", "premium"]
    provider: ProviderId | None = None
    model: str | None = None
    #: "…3f9a": enough to recognise the configured key, not to use it.
    key_hint: str | None = None
    verified_at: datetime | None = None
    #: The server's free model, or None when the server has no LLM configured.
    standard_model: str | None = None
    #: False when the server has no SECRETS_ENCRYPTION_KEY: keys cannot be stored.
    premium_available: bool
    providers: list[ProviderOption]


class AiSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["standard", "premium"]
    provider: ProviderId | None = None
    model: str | None = Field(default=None, max_length=100)
    #: Omit to keep the stored key (e.g. when only the model changes).
    api_key: SecretStr | None = None
