/**
 * Wire types for the ER-PLATFORM API (TRD 2.5 / FIG 2.9 / FIG 2.10).
 *
 * Hand-written rather than generated from OpenAPI. That is a deliberate
 * trade for now — generation is the better answer once the surface stops
 * moving, and the moment it does these should be replaced by
 * `openapi-typescript` output so drift becomes a build error instead of a
 * runtime surprise.
 *
 * Note what is absent: there is no `tenant_id` field on any request type.
 * TRD 2.5 takes the tenant from the verified token only, and the backend has
 * a test that fails if any request schema grows one. Adding it here would be
 * the first half of that mistake.
 */

/** FIG 2.2 — the job lifecycle, in order. */
export type JobStatus =
  | "PENDING"
  | "PREPROCESSING"
  | "EMBEDDING"
  | "BLOCKING"
  | "MATCHING"
  | "AWAITING_REVIEW"
  | "RESOLVING"
  | "COMPLETED"
  | "FAILED";

/** FIG 2.5 — where a candidate cluster sits in the confidence gate. */
export type ClusterStatus =
  | "AUTO_MERGED"
  | "PENDING_API"
  | "PENDING_REVIEW"
  | "RESOLVED"
  | "REJECTED";

/** FIG 2.3 / FR-02 — the ingestion classifier's three classes. */
export type RecordClassification = "Valid" | "Junk" | "Missing_Context";

/** FIG 2.7 / FR-08 — a steward's verdict. */
export type FeedbackDecision = "MERGE" | "REJECT" | "SPLIT";

/** One stage's counters. Timestamps are server time (ISO 8601). */
export interface StageEntry {
  done: number;
  total?: number;
  started_at?: string;
  updated_at?: string;
  details?: Record<string, unknown>;
}

/** Per-stage counters, keyed by stage name. `error` carries a failure reason. */
export interface StageProgress {
  [stage: string]: StageEntry | { message: string; at: string } | undefined;
}

/** 202 response to an upload (FR-01). */
export interface JobAccepted {
  job_id: string;
  status: JobStatus;
  total_bytes: number;
}

export interface JobStatusResponse {
  id: string;
  status: JobStatus;
  stage_progress: StageProgress;
  total_rows: number | null;
  created_by: string | null;
  /** The uploaded file's own name; null for jobs created before it was recorded. */
  original_filename: string | null;
  created_at: string;
  completed_at: string | null;
}

/** One keyset page of a tenant's jobs, newest first — GET /v1/jobs. */
export interface JobPage {
  items: JobStatusResponse[];
  next_cursor: string | null;
}

export interface ClusterStatusCounts {
  AUTO_MERGED: number;
  PENDING_API: number;
  PENDING_REVIEW: number;
  RESOLVED: number;
  REJECTED: number;
}

export interface JobStats {
  job_id: string;
  status: JobStatus;
  total_rows: number | null;
  records_by_classification: Record<string, number>;
  clusters_by_status: ClusterStatusCounts;
  clusters_total: number;
  records_clustered: number;
  mean_confidence: number | null;
}

export interface ClusterSummary {
  id: string;
  job_id: string;
  status: ClusterStatus;
  /** NUMERIC(4,3) — serialised as a string by the backend, not a number. */
  confidence_score: string | null;
  matcher_version: string | null;
  master_record_id: string | null;
  created_at: string;
  member_count: number;
}

export interface ClusterMemberSummary {
  record_id: string;
  pairwise_score: string | null;
  is_master: boolean;
  raw_row: Record<string, unknown>;
  /**
   * name/address/locality/state/pincode, alias-resolved and cleaned of the
   * redundant-comma / null-sentinel noise a raw export carries — computed
   * server-side (`app/ml_pipeline/fields.py` + `cleaning.py`) so this stays
   * the one place that logic lives. Prefer this over parsing `raw_row`
   * directly; `raw_row` itself stays untouched for the record's own sake.
   */
  clean_fields: Record<string, string>;
}

export interface ClusterDetail extends ClusterSummary {
  members: ClusterMemberSummary[];
  /** The survivor's fields as a steward corrected them at merge time. */
  golden_fields: Record<string, string> | null;
}

/** Where every uploaded row ended up — GET /v1/jobs/{id}/summary. */
export interface JobFunnel {
  total_rows: number;
  valid: number;
  missing_context: number;
  junk: number;
  unclassified: number;
  embedded: number;
  candidate_pairs: number;
  records_in_clusters: number;
  unique_records: number;
  clusters: number;
}

export interface JobSummary {
  job: JobStatusResponse;
  funnel: JobFunnel;
  clusters_by_status: ClusterStatusCounts;
  records_merged_away: number;
  issues_by_class: Record<string, Record<string, number>>;
  issue_labels: Record<string, string>;
  blocking: Record<string, unknown> | null;
  matching: Record<string, unknown> | null;
}

/** The results page's record tabs. */
export type RecordView = "needs_info" | "junk" | "unique" | "all";

export interface RecordItem {
  record_id: string;
  classification: RecordClassification | null;
  clean_fields: Record<string, string>;
  issues: string[];
  cluster_id: string | null;
  cluster_status: ClusterStatus | null;
}

export interface RecordPage {
  items: RecordItem[];
  next_cursor: string | null;
}

/** One standardisation ontology (ontologies/<name>.yaml on the backend). */
export interface NaturalKeyOption {
  key: string;
  /** Normalised header spellings that hold it. */
  headers: string[];
}

export interface OntologyOption {
  name: string;
  label: string;
  description: string;
  /** A domain this tenant generated from a description. */
  custom?: boolean;
  /** Field -> normalised header spellings; the mapping screen suggests from these. */
  field_aliases?: Record<string, string[]>;
  natural_keys?: NaturalKeyOption[];
}

/** The fields a column can be mapped to. `name` is required. */
export const MAPPABLE_FIELDS = ["name", "address", "district", "state", "pincode"] as const;
export type MappableField = (typeof MAPPABLE_FIELDS)[number];

/** Sent with an upload: which CSV column means what. Values are CSV headers. */
export interface ColumnMapping {
  fields: Partial<Record<MappableField, string>>;
  record_id?: string;
  natural_keys?: Record<string, string>;
}

/** GET /v1/jobs/upload-options — what the upload screen needs up front. */
export interface UploadOptions {
  ontologies: OntologyOption[];
  default_ontology: string;
  /** Some LLM (the server's or the tenant's own key) can draft a custom domain. */
  custom_domains_available?: boolean;
  /** True: presigned PUT straight to object storage. False: multipart via the API. */
  direct_upload: boolean;
  max_upload_bytes: number;
}

/** POST /v1/jobs/uploads — a job plus where to PUT its CSV. */
export interface PresignedUpload {
  job_id: string;
  upload_url: string;
  method: string;
  /** Part of the signature: must be sent exactly as given. */
  headers: Record<string, string>;
  expires_in: number;
}

/** GET /system/status — is every part of the platform running? */
export interface SystemStatus {
  status: "ok" | "degraded";
  checks: Record<string, string>;
  workers: number;
  /** An LLM endpoint is configured, so SERP enrichment can run. */
  enrichment_llm: boolean;
}

/**
 * One keyset page. `next_cursor` is opaque by contract — it encodes
 * `(created_at, id)`, and the moment this client parses it the backend can no
 * longer change its sort key without breaking us. Pass it back untouched.
 */
export interface ClusterPage {
  items: ClusterSummary[];
  next_cursor: string | null;
}

/** Fields a steward may correct on the surviving record. */
export type CorrectableField = "name" | "address" | "district" | "state" | "pincode";

export interface FeedbackRequest {
  decision: FeedbackDecision;
  master_record_id?: string | null;
  excluded_record_ids?: string[] | null;
  /** MERGE only. Stored beside the record; the upload itself is never edited. */
  field_corrections?: Partial<Record<CorrectableField, string>> | null;
}

export interface FeedbackResponse {
  feedback_id: string;
  cluster_id: string;
  decision: FeedbackDecision;
  cluster_status: ClusterStatus;
  created_at: string;
}

// --- AI tier (GET/PUT /v1/settings/ai) -------------------------------------------

export type AiMode = "standard" | "premium";
export type PremiumProvider = "openai" | "anthropic" | "cerebras";

export interface ProviderOption {
  id: PremiumProvider;
  label: string;
  default_model: string;
  /** A free tier exists: a key can cost nothing. */
  free: boolean;
  /** One line on what to expect (limits, billing). */
  note: string;
  /** Where to create a key. */
  key_url: string;
  key_placeholder: string;
}

/** Never carries the key itself: `key_hint` is "…3f9a". */
export interface AiSettings {
  mode: AiMode;
  provider: PremiumProvider | null;
  model: string | null;
  key_hint: string | null;
  verified_at: string | null;
  /** The server's free model, or null when the server has no LLM. */
  standard_model: string | null;
  /** False when the server cannot encrypt keys (no SECRETS_ENCRYPTION_KEY). */
  premium_available: boolean;
  providers: ProviderOption[];
}

export interface AiSettingsUpdate {
  mode: AiMode;
  provider?: PremiumProvider;
  model?: string;
  /** Omit to keep the stored key. */
  api_key?: string;
}

// --- Custom data domains (/v1/ontologies) ----------------------------------------

export type TargetType = "text" | "person" | "phone" | "email" | "url" | "number";

export interface NaturalKeyDraft {
  key: string;
  label: string;
  headers: string[];
  pattern: string;
}

export interface TargetDraft {
  key: string;
  type: TargetType;
  description: string;
}

export interface OntologyDraft {
  label: string;
  description: string;
  entity_noun: string;
  name_headers: string[];
  address_headers: string[];
  abbreviations: Record<string, string>;
  generic_name_tokens: string[];
  natural_keys: NaturalKeyDraft[];
  enrichment_targets: TargetDraft[];
}

export interface DomainDraftResponse {
  draft: OntologyDraft;
  /** What the server removed from the AI's proposal, and why. */
  notes: string[];
  drafted_by: string;
}
