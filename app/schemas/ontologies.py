"""Request/response models for custom data domains."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.ontology.generator import OntologyDraft


class GenerateDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=8, max_length=1000)
    #: The file's header row, so the draft can name the columns that hold
    #: the entity's name, address and identifiers.
    headers: list[str] = Field(default_factory=list, max_length=200)
    #: A handful of rows, sent to the AI provider to ground the draft and
    #: used to test proposed identifier formats. Truncated server-side.
    sample_rows: list[dict[str, str]] = Field(default_factory=list, max_length=20)


class DomainDraftResponse(BaseModel):
    draft: OntologyDraft
    #: Everything the sanitiser removed from the LLM's proposal, and why.
    notes: list[str]
    #: Which AI drafted it ("standard:openai_compatible/qwen..." or "premium:...").
    drafted_by: str


class SaveDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft: OntologyDraft
    #: The description the user typed, kept with the domain.
    prompt: str = Field(default="", max_length=1000)
    #: The same sample rows, so identifier formats added or edited by the
    #: user are tested against the data too.
    sample_rows: list[dict[str, str]] = Field(default_factory=list, max_length=20)
