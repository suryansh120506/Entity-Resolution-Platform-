"""Application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: str = "INFO"
    sql_echo: bool = False

    # --- Database -------------------------------------------------------------
    # The runtime connection. Must be the non-owner role so that RLS binds.
    # The default is constructed rather than left as a bare string: pydantic
    # coerces a str at runtime either way, but `Field(default=<str>)` is typed
    # as returning that str, so a `PostgresDsn`-annotated field assigned a str
    # default is a genuine type error rather than a quirk to silence.
    database_url: PostgresDsn = Field(
        default=PostgresDsn(
            "postgresql+psycopg://er_app:er_app_dev_pw@localhost:5432/entity_resolution"
        )
    )
    # Owner role, used by Alembic only. Never read by the API or the workers.
    migration_database_url: PostgresDsn | None = None

    # Control-plane role (BYPASSRLS). Only the three legitimately cross-tenant
    # operations use it: tenant provisioning, dead-letter triage, and the
    # nightly retrain's tenant fan-out. Optional — absent, the cross-tenant
    # helpers raise instead of silently degrading.
    platform_database_url: PostgresDsn | None = None

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # --- Auth (TRD 2.5) -------------------------------------------------------
    jwt_secret_key: str = "change-me-dev-only-not-a-real-secret"
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = "er-platform"
    jwt_issuer: str | None = "er-platform-auth"

    # --- ML (TRD 2.2) ---------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    #: Exact revision hash. TRD 2.2 requires pinning by revision, not name.
    embedding_model_revision: str | None = None

    #: Pairs scored at or above this probability link records into a cluster
    #: (see app/ml_pipeline/matching/clustering.py). Below it the model's own
    #: verdict is "different schools".
    match_link_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Largest cluster the matcher will emit before re-splitting it at a
    #: stricter threshold.
    max_cluster_size: int = Field(default=50, ge=2)

    #: A cluster of 3+ records below this edge density is never auto-merged,
    #: whatever its bottleneck score: low density under a high score is the
    #: signature of a chain merge (A~B~C~D, A never linked to D). 0.6 admits
    #: a triangle missing one edge (0.67) and stops a 4-chain (0.5).
    auto_merge_min_density: float = Field(default=0.6, ge=0.0, le=1.0)

    #: Pseudo-label weight of the EM prior when steward labels update m.
    #: m = (labels at level + strength * m_prior) / (labels + strength), so
    #: ten labels nudge the model and a thousand dominate it.
    feedback_prior_strength: float = Field(default=50.0, gt=0.0)

    # --- Uploads ----------------------------------------------------------------
    #: Where POST /v1/jobs spools an uploaded CSV. A local directory is the
    #: development posture; production points this at a mounted volume, and
    #: object storage replaces it entirely — which is a change to
    #: `app/api/v1/routes/jobs.py` and to the `csv_path` the ingest task
    #: receives, and to nothing else.
    upload_dir: Path = Path("./var/uploads")

    #: Upload ceiling. Enforced while streaming, not after: a limit checked
    #: once the file is on disk is a limit that already cost the disk.
    max_upload_bytes: int = 256 * 1024 * 1024

    # --- Object storage (S3-compatible: Cloudflare R2, AWS S3, MinIO) ----------
    #: All of bucket + keys set = browsers upload straight to the bucket via
    #: presigned PUT URLs; otherwise uploads stream through the API (dev).
    #: R2: endpoint https://<account_id>.r2.cloudflarestorage.com, region "auto".
    object_storage_endpoint_url: str | None = None
    object_storage_bucket: str | None = None
    object_storage_access_key_id: str | None = None
    object_storage_secret_access_key: str | None = None
    object_storage_region: str = "auto"
    presigned_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    @property
    def object_storage_enabled(self) -> bool:
        return bool(
            self.object_storage_bucket
            and self.object_storage_access_key_id
            and self.object_storage_secret_access_key
        )

    # --- Enrichment (zero-cost: open web + free/local LLM) ----------------------
    #: Identifies the scraper to the sites it reads. Put a real contact URL or
    #: address here before enabling any registry adapter.
    enrichment_user_agent: str = "ER-Platform-Enrichment/1.0"
    enrichment_min_interval_seconds: float = Field(default=2.0, ge=0.5)
    enrichment_timeout_seconds: float = 15.0

    #: Any OpenAI-compatible chat endpoint. Unset = SERP+LLM enrichment off.
    #:   Groq:   https://api.groq.com/openai/v1   (LLM_API_KEY required)
    #:   Ollama: http://localhost:11434/v1        (no key)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    # Groq retired llama-3.1-8b-instant. Must be a NON-reasoning model: the YES/NO
    # judge caps replies at 3 tokens, which a reasoning model spends thinking.
    llm_model: str = "qwen/qwen3.8-27b"
    #: Further models on the same endpoint and key, comma-separated, used when
    #: the first model's *daily* limit is reached. A provider such as Groq caps
    #: each model separately, so each extra model is another free daily budget:
    #: LLM_POOL_MODELS=openai/gpt-oss-120b,openai/gpt-oss-20b
    llm_pool_models: str = ""

    #: Fernet key that encrypts tenants' own LLM API keys (BYOK) at rest.
    #: Unset = Premium AI cannot be enabled: a key is never stored in clear.
    #: Generate with `python -c "from cryptography.fernet import Fernet;
    #: print(Fernet.generate_key().decode())"`. Rotating it orphans stored keys.
    secrets_encryption_key: str | None = None

    #: Tier 2: the 0.80-0.95 band (and suspected chains) is settled by a web
    #: search and an LLM YES/NO instead of a steward. Needs an LLM; without
    #: one that band goes to the review queue as before.
    auto_triage_enabled: bool = True
    #: Seconds between the *starts* of DuckDuckGo searches, across every worker
    #: process (an advisory lock spaces them). Faster than this gets the IP blocked.
    ddgs_min_interval_seconds: float = Field(default=2.0, ge=2.0)
    #: Searches Tier 3 keeps in flight at once. A search takes 3-10 s, so one at
    #: a time is one per 5-12 s however small the interval; two or three
    #: overlapping reach the interval's rate without exceeding it. Raise it only
    #: if searches stay slow; the interval above is what protects the IP.
    ddgs_concurrency: int = Field(default=2, ge=1, le=4)

    @property
    def llm_pool_list(self) -> list[str]:
        models = [m.strip() for m in self.llm_pool_models.split(",") if m.strip()]
        return [m for m in dict.fromkeys(models) if m != self.llm_model]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url)

    @property
    def auto_triage_active(self) -> bool:
        return self.auto_triage_enabled and self.llm_enabled

    @field_validator("jwt_secret_key")
    @classmethod
    def _reject_default_secret_outside_local(cls, v: str, info: ValidationInfo) -> str:
        # `environment` is declared above this field, so it is already
        # validated and present in info.data.
        env = (info.data or {}).get("environment", "local")
        if env in {"staging", "production"} and v.startswith("change-me"):
            raise ValueError("JWT_SECRET_KEY must be set outside local/ci")
        return v

    @property
    def sync_database_url(self) -> str:
        """Sync DSN — queue workers and Alembic. psycopg 3 driver."""
        return str(self.database_url)

    @property
    def async_database_url(self) -> str:
        """Async DSN — FastAPI request handlers.

        Identical string to the sync DSN: the psycopg 3 dialect serves both,
        and SQLAlchemy picks the mode from create_engine vs create_async_engine.
        Exposed as its own property anyway so the two call sites read clearly
        and so a future driver split is a one-line change here.
        """
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
