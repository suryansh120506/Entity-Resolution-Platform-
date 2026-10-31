"""Request/response models for the job routes — TRD 2.5 / FIG 2.9.

No model here carries a `tenant_id`. That is the rule TRD 2.5 states and
`app/core/security.py` enforces: the tenant comes from the verified token and
from nowhere else, so a tenant field on a request body is an attack surface
rather than an input. If one appears here, that is the bug.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ml_pipeline.column_mapping import ColumnMapping


class JobAccepted(BaseModel):
    """202 response to an upload. FR-01: the id comes back immediately."""

    job_id: uuid.UUID
    status: str
    total_bytes: int = Field(description="Bytes spooled from the upload.")


class NaturalKeyOption(BaseModel):
    key: str
    #: Normalised header spellings that hold it; the first is the one a
    #: mapped column is stored under.
    headers: list[str]


class OntologyOption(BaseModel):
    name: str
    label: str
    description: str
    #: True for a domain this tenant generated from a description.
    custom: bool = False
    #: Field -> normalised header spellings the domain recognises. The upload
    #: screen pre-fills the column mapping from these.
    field_aliases: dict[str, list[str]] = Field(default_factory=dict)
    natural_keys: list[NaturalKeyOption] = Field(default_factory=list)

    @classmethod
    def from_engine(cls, engine: Any, *, custom: bool = False) -> OntologyOption:
        from app.ml_pipeline.column_mapping import alias_tables

        spec = engine.spec
        aliases, keys = alias_tables(
            spec.fields, {key: rule.headers for key, rule in spec.natural_keys.items()}
        )
        return cls(
            name=spec.name,
            label=spec.label or spec.name.title(),
            description=spec.description,
            custom=custom,
            field_aliases=aliases,
            natural_keys=[NaturalKeyOption(key=k, headers=h) for k, h in keys.items()],
        )


class UploadOptions(BaseModel):
    """What the upload screen needs before the user picks a file."""

    ontologies: list[OntologyOption]
    default_ontology: str
    #: Whether "Custom data domain" can draft one: some LLM (the server's or
    #: the tenant's own key) is available.
    custom_domains_available: bool = False
    #: True = request a presigned URL and PUT straight to object storage.
    #: False = multipart POST /v1/jobs (no bucket configured; local dev).
    direct_upload: bool
    max_upload_bytes: int


class PresignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    ontology: str = "education"
    column_mapping: ColumnMapping | None = None


class PresignedUpload(BaseModel):
    job_id: uuid.UUID
    upload_url: str
    method: str
    #: Must be sent verbatim with the PUT: they are part of the signature.
    headers: dict[str, str]
    expires_in: int


class JobStatusResponse(BaseModel):
    """Everything the frontend's poll needs, from Postgres only.

    Deliberately not read from Celery's result backend: that backend is
    transient (TRD 2.7 is explicit that a job must be diagnosable from
    Postgres alone), and a status that disappears when Redis is flushed is
    not a status. `jobs.status` and `jobs.stage_progress` are the record.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    stage_progress: dict[str, Any]
    total_rows: int | None
    created_by: str | None
    original_filename: str | None = None
    created_at: datetime
    completed_at: datetime | None


class JobPage(BaseModel):
    """One keyset page of a tenant's jobs, newest first.

    Mirrors `ClusterPage`: `next_cursor` is opaque, encoding `(created_at,
    id)` of the last row returned. Newest-first rather than the review
    queue's oldest-first — a dashboard is "what's running or just
    finished", not a FIFO a steward works down.
    """

    items: list[JobStatusResponse]
    next_cursor: str | None = None


class ClusterStatusCounts(BaseModel):
    """Cluster counts by status — the resolution-rate dashboard (TRD 2.7)."""

    AUTO_MERGED: int = 0
    PENDING_API: int = 0
    PENDING_REVIEW: int = 0
    RESOLVED: int = 0
    REJECTED: int = 0


class JobStats(BaseModel):
    """Aggregates for one job, computed in the database.

    Counted with SQL aggregates rather than by loading rows and counting in
    Python: at 50k records the difference is a millisecond against a
    megabyte, and this is the endpoint a dashboard polls.
    """

    job_id: uuid.UUID
    status: str
    total_rows: int | None
    records_by_classification: dict[str, int]
    clusters_by_status: ClusterStatusCounts
    clusters_total: int
    records_clustered: int
    mean_confidence: float | None


class ClusterMemberSummary(BaseModel):
    """One record inside a cluster, as the review UI shows it.

    `raw_row` stays exactly what was uploaded (TRD 2.3) — it is not the
    field to read for display. `clean_fields` is the canonical five fields
    (name/address/locality/state/pincode), alias-resolved the same way the
    classifier reads them, with the redundant-comma/null-sentinel noise a raw
    export carries stripped out. Computed fresh on every response rather than
    stored, so it can never drift out of sync with `raw_row`.
    """

    record_id: uuid.UUID
    pairwise_score: Decimal | None
    is_master: bool
    raw_row: dict[str, Any]
    clean_fields: dict[str, str] = Field(default_factory=dict)


class ClusterSummary(BaseModel):
    """A row in the review queue (FIG 2.9). No members — see ClusterDetail."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_id: uuid.UUID
    status: str
    confidence_score: Decimal | None
    matcher_version: str | None
    master_record_id: uuid.UUID | None
    created_at: datetime
    member_count: int = 0


class ClusterPage(BaseModel):
    """One keyset page of the review queue.

    `next_cursor` is opaque by contract. It encodes `(created_at, id)`, and
    the moment a client parses it we lose the ability to change the sort key
    without breaking them — so it is base64 and documented as meaningless.
    """

    items: list[ClusterSummary]
    next_cursor: str | None = None


class ClusterDetail(ClusterSummary):
    """A single cluster with its members, for the review screen."""

    members: list[ClusterMemberSummary]
    #: The survivor's fields as a steward corrected them, if they did.
    golden_fields: dict[str, str] | None = None


class JobFunnel(BaseModel):
    """Where every uploaded row ended up — the results page's headline."""

    total_rows: int
    valid: int
    missing_context: int
    junk: int
    unclassified: int
    embedded: int
    candidate_pairs: int
    records_in_clusters: int
    unique_records: int
    clusters: int


class JobSummary(BaseModel):
    """Everything the results page needs, in one request."""

    job: JobStatusResponse
    funnel: JobFunnel
    clusters_by_status: ClusterStatusCounts
    records_merged_away: int = Field(
        description="Records that were duplicates of a surviving record (auto or steward merge)."
    )
    issues_by_class: dict[str, dict[str, int]]
    issue_labels: dict[str, str]
    blocking: dict[str, Any] | None = None
    matching: dict[str, Any] | None = None


class RecordItem(BaseModel):
    """One source row, as the results drill-down shows it."""

    record_id: uuid.UUID
    classification: str | None
    clean_fields: dict[str, str]
    issues: list[str]
    cluster_id: uuid.UUID | None = None
    cluster_status: str | None = None


class RecordPage(BaseModel):
    items: list[RecordItem]
    next_cursor: str | None = None
