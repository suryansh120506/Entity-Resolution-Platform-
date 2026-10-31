-- ============================================================================
-- ER-PLATFORM-001 · migration 0006 — zero-cost medallion refactor
--
-- 1. task_queue            Postgres job queue (FOR UPDATE SKIP LOCKED), replacing
--                          Celery + Redis.
-- 2. silver_records        Ontology-standardised, typed view of each raw row.
-- 3. enrichment_results    Natural-key enrichment cache (registry / SERP+LLM).
-- 4. Fellegi-Sunter feedback loop:
--      splink_settings, pair_scores, feedback_events, pair_labels,
--      splink_level_stats (trigger-maintained), entity_constraints,
--      splink_level_estimates (view).
-- 5. clusters.entity_id / clusters.density for stable IDs and chain detection.
-- 6. Drops the two unused HNSW indexes on record_embeddings.
--
-- Every new table is tenant-scoped with ENABLE + FORCE + USING/WITH CHECK,
-- identical to migration 0002, so tests/test_rls_isolation.py's drift guards
-- hold without a new exception. The worker claims queue rows through the
-- BYPASSRLS control-plane role; everything a task then does runs on a
-- tenant-bound data-plane session.
--
-- No BEGIN/COMMIT: composes with Alembic's transaction (psql -1 by hand).
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 1. task_queue
-- ---------------------------------------------------------------------------

CREATE TABLE task_queue (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id        uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  -- Deleting a job deletes its outstanding work. Under Celery a deleted job's
  -- messages survived in Redis and resurfaced as JobMissingError noise.
  job_id           uuid        REFERENCES jobs(id) ON DELETE CASCADE,
  queue            text        NOT NULL,
  task_name        text        NOT NULL,
  payload          jsonb       NOT NULL DEFAULT '{}'::jsonb,
  status           text        NOT NULL DEFAULT 'queued',
  priority         smallint    NOT NULL DEFAULT 0,
  attempts         integer     NOT NULL DEFAULT 0,
  max_attempts     integer     NOT NULL DEFAULT 3,
  run_at           timestamptz NOT NULL DEFAULT now(),
  locked_by        text,
  lease_expires_at timestamptz,
  dedupe_key       text,
  last_error       text,
  enqueued_at      timestamptz NOT NULL DEFAULT now(),
  started_at       timestamptz,
  finished_at      timestamptz,

  CONSTRAINT ck_task_queue_status_valid
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
  CONSTRAINT ck_task_queue_payload_object CHECK (jsonb_typeof(payload) = 'object'),
  CONSTRAINT ck_task_queue_attempts_bounded
    CHECK (max_attempts >= 1 AND attempts >= 0 AND attempts <= max_attempts),
  -- A lease exists exactly when a worker holds the row. Makes "running with
  -- no owner" (unreapable) and "queued but locked" (unclaimable) unwritable.
  CONSTRAINT ck_task_queue_lease_iff_running
    CHECK ((status = 'running') = (locked_by IS NOT NULL AND lease_expires_at IS NOT NULL)),
  CONSTRAINT ck_task_queue_finished_iff_terminal
    CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL))
);

-- The claim path. Partial, so it holds only claimable rows and stays small
-- however much history accumulates.
CREATE INDEX ix_task_queue_claimable
  ON task_queue (queue, priority DESC, run_at, enqueued_at)
  WHERE status = 'queued';

-- The reaper's scan for expired leases.
CREATE INDEX ix_task_queue_leases
  ON task_queue (lease_expires_at)
  WHERE status = 'running';

-- Idempotent enqueue: one live task per (tenant, dedupe_key). A finished task
-- frees the key, so the same stage can be re-run later on purpose.
CREATE UNIQUE INDEX uq_task_queue_live_dedupe
  ON task_queue (tenant_id, dedupe_key)
  WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'running');

CREATE INDEX ix_task_queue_tenant_id_job_id ON task_queue (tenant_id, job_id);
CREATE INDEX ix_task_queue_finished_at ON task_queue (finished_at) WHERE finished_at IS NOT NULL;

-- Wake idle workers immediately instead of on their next poll. NOTIFY is
-- transactional (delivered on commit) and Postgres folds identical payloads
-- within a transaction, so a bulk enqueue sends one wake-up per queue.
CREATE FUNCTION task_queue_notify() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_notify('er_task_queue', NEW.queue);
  RETURN NULL;
END
$$;

CREATE TRIGGER trg_task_queue_notify
  AFTER INSERT OR UPDATE OF status, run_at ON task_queue
  FOR EACH ROW WHEN (NEW.status = 'queued')
  EXECUTE FUNCTION task_queue_notify();


-- ---------------------------------------------------------------------------
-- 2. silver_records
-- ---------------------------------------------------------------------------

-- Which ontology standardises this job (ontologies/<name>.yaml). Catalog-only
-- add: a constant default does not rewrite the table.
ALTER TABLE jobs ADD COLUMN ontology text NOT NULL DEFAULT 'education';
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_ontology_name CHECK (ontology ~ '^[a-z][a-z0-9_]{0,62}$');
-- One row per school_records row, produced by the ontology engine. raw_row
-- (Bronze) is never edited; this is the typed, versioned interpretation of
-- it. Re-standardising under a new ontology version overwrites this row and
-- leaves Bronze untouched.

CREATE TABLE silver_records (
  record_id        uuid        PRIMARY KEY REFERENCES school_records(id) ON DELETE CASCADE,
  tenant_id        uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  job_id           uuid        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  ontology         text        NOT NULL,
  ontology_version text        NOT NULL,
  name             text,
  name_distinctive text,
  address          text,
  locality         text,
  district         text,
  state            text,
  pincode          text,
  attributes       jsonb       NOT NULL DEFAULT '{}'::jsonb,
  natural_keys     jsonb       NOT NULL DEFAULT '{}'::jsonb,
  standardized_at  timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT ck_silver_records_pincode_shape
    CHECK (pincode IS NULL OR pincode ~ '^[1-9][0-9]{5}$'),
  CONSTRAINT ck_silver_records_attributes_object CHECK (jsonb_typeof(attributes) = 'object'),
  CONSTRAINT ck_silver_records_natural_keys_object CHECK (jsonb_typeof(natural_keys) = 'object'),
  CONSTRAINT ck_silver_records_ontology_version_sha256
    CHECK (ontology_version ~ '^[0-9a-f]{64}$')
);

CREATE INDEX ix_silver_records_tenant_id_job_id ON silver_records (tenant_id, job_id);
-- natural_keys @> '{"cbse_affiliation_no": "2130176"}' — the enrichment
-- join and the deterministic-match lookup.
CREATE INDEX ix_silver_records_natural_keys ON silver_records USING gin (natural_keys jsonb_path_ops);


-- ---------------------------------------------------------------------------
-- 3. enrichment_results
-- ---------------------------------------------------------------------------
-- Keyed by natural key, not by record: the same affiliation number arriving
-- in ten jobs is fetched once and re-read nine times. Negative results are
-- cached too (status not_found/blocked) so a key that cannot be resolved is
-- not re-scraped on every job until it expires.

CREATE TABLE enrichment_results (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  key_type    text        NOT NULL,
  key_value   text        NOT NULL,
  strategy    text        NOT NULL,
  source      text        NOT NULL,
  status      text        NOT NULL,
  fields      jsonb       NOT NULL DEFAULT '{}'::jsonb,
  fetched_at  timestamptz NOT NULL DEFAULT now(),
  expires_at  timestamptz NOT NULL,

  CONSTRAINT ck_enrichment_results_strategy_valid CHECK (strategy IN ('registry', 'serp_llm')),
  CONSTRAINT ck_enrichment_results_status_valid
    CHECK (status IN ('found', 'not_found', 'blocked', 'error')),
  CONSTRAINT ck_enrichment_results_fields_object CHECK (jsonb_typeof(fields) = 'object'),
  -- A 'found' row with nothing in it is a scraper bug wearing a success code.
  CONSTRAINT ck_enrichment_results_found_has_fields
    CHECK (status <> 'found' OR fields <> '{}'::jsonb),
  CONSTRAINT ck_enrichment_results_expiry_after_fetch CHECK (expires_at > fetched_at),
  CONSTRAINT uq_enrichment_results_key UNIQUE (tenant_id, key_type, key_value, strategy)
);

CREATE INDEX ix_enrichment_results_expires_at ON enrichment_results (tenant_id, expires_at);


-- ---------------------------------------------------------------------------
-- 4. Fellegi-Sunter feedback loop
-- ---------------------------------------------------------------------------

-- The comparison layout a gamma vector is read against. gamma[i] means
-- nothing without comparison_names[i] and the level count of comparison i,
-- so every gamma written anywhere carries the settings_hash that decodes it.
CREATE TABLE splink_settings (
  tenant_id        uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  settings_hash    text        NOT NULL,
  comparison_names text[]      NOT NULL,
  level_counts     smallint[]  NOT NULL,
  settings         jsonb       NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, settings_hash),
  CONSTRAINT ck_splink_settings_hash_sha256 CHECK (settings_hash ~ '^[0-9a-f]{64}$'),
  CONSTRAINT ck_splink_settings_layout_aligned
    CHECK (cardinality(comparison_names) >= 1
           AND cardinality(comparison_names) = cardinality(level_counts))
);

-- Every scored candidate pair with its comparison vector. Splink's gamma
-- is -1 for "null on either side" and 0..n-1 for the levels.
CREATE TABLE pair_scores (
  job_id            uuid        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  record_l          uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  record_r          uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  tenant_id         uuid        NOT NULL,
  settings_hash     text        NOT NULL,
  match_probability real        NOT NULL,
  match_weight      real        NOT NULL,
  gamma             smallint[]  NOT NULL,
  scored_at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, record_l, record_r),
  FOREIGN KEY (tenant_id, settings_hash)
    REFERENCES splink_settings (tenant_id, settings_hash) ON DELETE CASCADE,
  CONSTRAINT ck_pair_scores_canonical_order CHECK (record_l < record_r),
  CONSTRAINT ck_pair_scores_probability_unit CHECK (match_probability BETWEEN 0 AND 1),
  CONSTRAINT ck_pair_scores_gamma_nonempty CHECK (cardinality(gamma) >= 1)
);

CREATE INDEX ix_pair_scores_tenant_id_job_id ON pair_scores (tenant_id, job_id);
CREATE INDEX ix_pair_scores_record_r ON pair_scores (record_r);

-- Append-only log of every adjudication. UPDATE/DELETE are revoked below.
-- `partition` is the steward's final grouping of the cluster's members:
-- MERGE = one group, SPLIT = several, REJECT = all singletons. Pair labels
-- are derived from it, so all three verdicts share one representation.
CREATE TABLE feedback_events (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id        uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  job_id           uuid        REFERENCES jobs(id) ON DELETE SET NULL,
  -- SET NULL, not CASCADE: a re-run deletes unreviewed clusters, and the
  -- labels must outlive the cluster row they were given against.
  cluster_id       uuid        REFERENCES clusters(id) ON DELETE SET NULL,
  actor_type       text        NOT NULL,
  actor_id         text        NOT NULL,
  decision         text        NOT NULL,
  partition        jsonb       NOT NULL,
  -- Probability this item was sent for labelling. 1.0 for the review queue,
  -- the sampling rate for audit draws: inverse-propensity weights undo the
  -- bias of only ever labelling the uncertain band.
  propensity       real        NOT NULL DEFAULT 1.0,
  model_version_id uuid        REFERENCES model_versions(id) ON DELETE SET NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT ck_feedback_events_actor_type_valid CHECK (actor_type IN ('steward', 'audit', 'llm')),
  CONSTRAINT ck_feedback_events_decision_valid CHECK (decision IN ('MERGE', 'SPLIT', 'REJECT')),
  CONSTRAINT ck_feedback_events_partition_shape
    CHECK (jsonb_typeof(partition) = 'array' AND jsonb_array_length(partition) >= 1),
  CONSTRAINT ck_feedback_events_propensity_unit CHECK (propensity > 0 AND propensity <= 1)
);

CREATE INDEX ix_feedback_events_tenant_id_created_at ON feedback_events (tenant_id, created_at);

-- One label per (pair, source); a relabel replaces the old one, and the
-- trigger below moves the sufficient statistics accordingly.
CREATE TABLE pair_labels (
  tenant_id     uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  record_l      uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  record_r      uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  source        text        NOT NULL,
  label         boolean     NOT NULL,
  weight        real        NOT NULL,
  -- NULL when the pair was never scored (no comparison vector to learn
  -- from). Such labels still become constraints; they skip the statistics.
  settings_hash text,
  gamma         smallint[],
  event_id      uuid        NOT NULL REFERENCES feedback_events(id) ON DELETE CASCADE,
  labelled_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, record_l, record_r, source),
  CONSTRAINT ck_pair_labels_canonical_order CHECK (record_l < record_r),
  CONSTRAINT ck_pair_labels_source_valid CHECK (source IN ('steward', 'audit', 'llm')),
  CONSTRAINT ck_pair_labels_weight_positive CHECK (weight > 0),
  CONSTRAINT ck_pair_labels_gamma_with_layout CHECK ((gamma IS NULL) = (settings_hash IS NULL))
);

-- The model's trainable state. Fellegi-Sunter m/u are ratios of counts, so
-- these weighted counts are sufficient statistics: retraining reads
-- sum(levels) rows, independent of corpus size.
CREATE TABLE splink_level_stats (
  tenant_id       uuid             NOT NULL,
  settings_hash   text             NOT NULL,
  comparison_idx  smallint         NOT NULL,
  level           smallint         NOT NULL,
  match_w         double precision NOT NULL DEFAULT 0,
  nonmatch_w      double precision NOT NULL DEFAULT 0,
  updated_at      timestamptz      NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, settings_hash, comparison_idx, level),
  FOREIGN KEY (tenant_id, settings_hash)
    REFERENCES splink_settings (tenant_id, settings_hash) ON DELETE CASCADE,
  -- Tolerance for float drift from repeated add/subtract of the same weight.
  CONSTRAINT ck_splink_level_stats_nonnegative CHECK (match_w >= -1e-9 AND nonmatch_w >= -1e-9)
);

-- Steward decisions as hard graph constraints. Applied at clustering time,
-- so they hold across re-runs and model changes regardless of score.
CREATE TABLE entity_constraints (
  tenant_id  uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  record_l   uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  record_r   uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  kind       text        NOT NULL,
  event_id   uuid        NOT NULL REFERENCES feedback_events(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, record_l, record_r),
  CONSTRAINT ck_entity_constraints_canonical_order CHECK (record_l < record_r),
  CONSTRAINT ck_entity_constraints_kind_valid CHECK (kind IN ('must_link', 'cannot_link'))
);

CREATE INDEX ix_entity_constraints_record_r ON entity_constraints (tenant_id, record_r);

-- O(k) incremental training. Every insert/update/delete of a learnable label
-- (steward or audit, with a gamma) moves exactly one counter per comparison.
-- LLM labels are stored but never counted: an unaudited model's opinion
-- feeding the model that is being judged is a self-confirming loop.
-- Gamma -1 ("null on a side") carries no evidence and is skipped, matching
-- Splink's own treatment.
CREATE FUNCTION pair_labels_apply_stats() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  i integer;
BEGIN
  IF TG_OP IN ('UPDATE', 'DELETE')
     AND OLD.gamma IS NOT NULL AND OLD.source IN ('steward', 'audit') THEN
    FOR i IN 1 .. cardinality(OLD.gamma) LOOP
      CONTINUE WHEN OLD.gamma[i] < 0;
      UPDATE splink_level_stats
         SET match_w    = match_w    - CASE WHEN OLD.label THEN OLD.weight ELSE 0 END,
             nonmatch_w = nonmatch_w - CASE WHEN OLD.label THEN 0 ELSE OLD.weight END,
             updated_at = now()
       WHERE tenant_id = OLD.tenant_id AND settings_hash = OLD.settings_hash
         AND comparison_idx = i - 1 AND level = OLD.gamma[i];
    END LOOP;
  END IF;

  IF TG_OP IN ('INSERT', 'UPDATE')
     AND NEW.gamma IS NOT NULL AND NEW.source IN ('steward', 'audit') THEN
    FOR i IN 1 .. cardinality(NEW.gamma) LOOP
      CONTINUE WHEN NEW.gamma[i] < 0;
      INSERT INTO splink_level_stats
             (tenant_id, settings_hash, comparison_idx, level, match_w, nonmatch_w)
      VALUES (NEW.tenant_id, NEW.settings_hash, i - 1, NEW.gamma[i],
              CASE WHEN NEW.label THEN NEW.weight ELSE 0 END,
              CASE WHEN NEW.label THEN 0 ELSE NEW.weight END)
      ON CONFLICT (tenant_id, settings_hash, comparison_idx, level) DO UPDATE
         SET match_w    = splink_level_stats.match_w    + EXCLUDED.match_w,
             nonmatch_w = splink_level_stats.nonmatch_w + EXCLUDED.nonmatch_w,
             updated_at = now();
    END LOOP;
  END IF;

  RETURN NULL;
END
$$;

CREATE TRIGGER trg_pair_labels_apply_stats
  AFTER INSERT OR UPDATE OR DELETE ON pair_labels
  FOR EACH ROW EXECUTE FUNCTION pair_labels_apply_stats();

-- Laplace-smoothed empirical m and u per level. The Bayesian blend with the
-- EM prior happens in Python (it needs the prior's m values, which live in
-- the settings JSON); this view supplies the evidence side of it.
-- security_invoker so RLS is evaluated against the caller, not the owner.
CREATE VIEW splink_level_estimates WITH (security_invoker = true) AS
SELECT s.tenant_id,
       s.settings_hash,
       s.comparison_idx,
       cfg.comparison_names[s.comparison_idx + 1] AS comparison_name,
       s.level,
       s.match_w,
       s.nonmatch_w,
       (s.match_w + 1.0)
         / (sum(s.match_w) OVER w + cfg.level_counts[s.comparison_idx + 1]) AS m_laplace,
       (s.nonmatch_w + 1.0)
         / (sum(s.nonmatch_w) OVER w + cfg.level_counts[s.comparison_idx + 1]) AS u_laplace
  FROM splink_level_stats s
  JOIN splink_settings cfg USING (tenant_id, settings_hash)
WINDOW w AS (PARTITION BY s.tenant_id, s.settings_hash, s.comparison_idx);


-- ---------------------------------------------------------------------------
-- 5. Stable entity identity on clusters
-- ---------------------------------------------------------------------------
-- entity_id survives re-runs (assigned by Jaccard overlap with the previous
-- run's clusters); clusters.id does not. Added nullable, backfilled, then
-- tightened, so the column add itself is catalog-only.

ALTER TABLE clusters ADD COLUMN entity_id uuid;
UPDATE clusters SET entity_id = id WHERE entity_id IS NULL;
ALTER TABLE clusters ALTER COLUMN entity_id SET DEFAULT gen_random_uuid();
ALTER TABLE clusters ALTER COLUMN entity_id SET NOT NULL;
CREATE INDEX ix_clusters_tenant_id_entity_id ON clusters (tenant_id, entity_id);

-- Edge density of the cluster's scored graph: 1.0 = every member pair
-- scored above threshold; low density with a high bottleneck score is the
-- signature of a chain merge (A~B~C~D with A!~D).
ALTER TABLE clusters ADD COLUMN density real;
ALTER TABLE clusters ADD CONSTRAINT ck_clusters_density_unit
  CHECK (density IS NULL OR (density >= 0 AND density <= 1));


-- ---------------------------------------------------------------------------
-- 6. Drop the HNSW indexes
-- ---------------------------------------------------------------------------
-- Blocking runs exact cosine inside location blocks; nothing queries these.
-- Every embedding insert was paying HNSW graph maintenance for no reader.

DROP INDEX IF EXISTS ix_record_embeddings_name_embedding_hnsw;
DROP INDEX IF EXISTS ix_record_embeddings_address_embedding_hnsw;


-- ---------------------------------------------------------------------------
-- 7. Row-level security on every new table (FIG 2.8 shape, verbatim)
-- ---------------------------------------------------------------------------

DO $rls$
DECLARE
  t text;
  tables CONSTANT text[] := ARRAY[
    'task_queue', 'silver_records', 'enrichment_results', 'splink_settings',
    'pair_scores', 'feedback_events', 'pair_labels', 'splink_level_stats',
    'entity_constraints'
  ];
BEGIN
  FOREACH t IN ARRAY tables LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I '
      'USING (tenant_id = current_setting(''app.tenant_id'')::uuid) '
      'WITH CHECK (tenant_id = current_setting(''app.tenant_id'')::uuid)', t);
  END LOOP;
END
$rls$;


-- ---------------------------------------------------------------------------
-- 8. Privileges
-- ---------------------------------------------------------------------------
-- 0002's ALTER DEFAULT PRIVILEGES already granted DML on the tables above.
-- Two narrowings on top of it:
--   * feedback_events is append-only for both roles. FK cascades still work:
--     referential actions run as the table owner, not the caller.
--   * The data plane may enqueue and read the queue but never claims,
--     completes or deletes rows; that is the worker's control-plane path.

DO $privs$
DECLARE
  r text;
BEGIN
  FOREACH r IN ARRAY ARRAY['er_app', 'er_platform'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
      RAISE WARNING 'Role % does not exist; skipping its 0006 privileges.', r;
      CONTINUE;
    END IF;
    EXECUTE format('REVOKE UPDATE, DELETE ON feedback_events FROM %I', r);
    EXECUTE format('GRANT SELECT ON splink_level_estimates TO %I', r);
  END LOOP;

  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'er_app') THEN
    REVOKE UPDATE, DELETE ON task_queue FROM er_app;
  END IF;
END
$privs$;
