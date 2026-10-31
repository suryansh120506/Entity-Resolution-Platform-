-- ============================================================================
-- ER-PLATFORM-001 · migration 0008 — stable identity + gold-layer linking
--
-- silver_records.stable_id / identity_kind
--     A deterministic identity per record, the same in every upload:
--       nk:<key_type>:<value>        an ontology natural key (UDISE, ...)
--       fp:<ontology>:<sha256/32>    fingerprint of stable Silver fields
--       rec:<record_id>              no stable basis; never carries over
--     Nullable until old jobs are re-standardised (silver.backfill_job).
-- entity_constraints  (re-keyed)
--     Steward must/never-merge rules between *identities*, not job-run
--     record ids, so a decision made on one upload binds every later one.
-- entity_group_constraints
--     "Not all together": a rejected group of 3+ identities may never be
--     re-proposed as one cluster (any proper subset still may).
-- gold_links
--     Incoming records matched (Splink link_only) against the tenant's gold
--     layer: decided masters and released records of earlier jobs.
-- ============================================================================

-- --- 1. Stable identity on Silver --------------------------------------------

ALTER TABLE silver_records ADD COLUMN stable_id text;
ALTER TABLE silver_records ADD COLUMN identity_kind text;
ALTER TABLE silver_records ADD CONSTRAINT ck_silver_records_identity_pair
  CHECK ((stable_id IS NULL) = (identity_kind IS NULL));
ALTER TABLE silver_records ADD CONSTRAINT ck_silver_records_identity_kind_valid
  CHECK (identity_kind IS NULL OR identity_kind IN ('natural_key', 'fingerprint', 'record'));
ALTER TABLE silver_records ADD CONSTRAINT ck_silver_records_stable_id_shape
  CHECK (stable_id IS NULL OR stable_id ~ '^(nk|fp|rec):.+$');
CREATE INDEX ix_silver_records_tenant_id_stable_id ON silver_records (tenant_id, stable_id);


-- --- 2. Constraints bound to identities ----------------------------------------
-- No rows exist to carry over (checked before this migration was written);
-- recreated rather than altered so the new shape has no record-id residue.

DROP TABLE entity_constraints;

CREATE TABLE entity_constraints (
  tenant_id   uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  identity_l  text        NOT NULL,
  identity_r  text        NOT NULL,
  kind        text        NOT NULL,
  event_id    uuid        NOT NULL REFERENCES feedback_events(id) ON DELETE CASCADE,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, identity_l, identity_r),
  -- Strict: a rule between an identity and itself is either trivially true
  -- (must) or unsatisfiable (cannot); neither is a constraint.
  CONSTRAINT ck_entity_constraints_canonical_order CHECK (identity_l < identity_r),
  CONSTRAINT ck_entity_constraints_kind_valid CHECK (kind IN ('must_link', 'cannot_link')),
  CONSTRAINT ck_entity_constraints_identity_shape
    CHECK (identity_l ~ '^(nk|fp|rec):.+$' AND identity_r ~ '^(nk|fp|rec):.+$')
);
CREATE INDEX ix_entity_constraints_identity_r ON entity_constraints (tenant_id, identity_r);

CREATE TABLE entity_group_constraints (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  -- Sorted, distinct: the same group is the same array however it was given.
  identities  text[]      NOT NULL,
  event_id    uuid        NOT NULL REFERENCES feedback_events(id) ON DELETE CASCADE,
  created_at  timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT ck_entity_group_constraints_size CHECK (cardinality(identities) >= 3)
);
CREATE UNIQUE INDEX uq_entity_group_constraints_group
  ON entity_group_constraints (tenant_id, identities);
CREATE INDEX ix_entity_group_constraints_members
  ON entity_group_constraints USING gin (identities);


-- --- 3. Gold links ---------------------------------------------------------------

CREATE TABLE gold_links (
  job_id            uuid        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  record_id         uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  target_record_id  uuid        NOT NULL REFERENCES school_records(id) ON DELETE CASCADE,
  tenant_id         uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  -- gold_master: the survivor of a merged cluster in an earlier job.
  -- released:    a singleton, rejected or split-out record of an earlier job.
  target_kind       text        NOT NULL,
  target_entity_id  uuid,
  target_stable_id  text        NOT NULL,
  match_probability real        NOT NULL,
  match_weight      real,
  gamma             smallint[],
  settings_hash     text,
  -- auto_linked: confident, unambiguous, constraint-clean.
  -- review:      plausible but needs a steward (or ambiguous between targets).
  status            text        NOT NULL,
  -- natural_key / must_link: linked by identity, whatever the score.
  reason            text        NOT NULL DEFAULT 'score',
  created_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, record_id, target_record_id),
  CONSTRAINT ck_gold_links_target_kind_valid CHECK (target_kind IN ('gold_master', 'released')),
  CONSTRAINT ck_gold_links_status_valid
    CHECK (status IN ('auto_linked', 'review', 'confirmed', 'rejected')),
  CONSTRAINT ck_gold_links_reason_valid CHECK (reason IN ('score', 'natural_key', 'must_link')),
  CONSTRAINT ck_gold_links_probability_unit CHECK (match_probability BETWEEN 0 AND 1),
  CONSTRAINT ck_gold_links_not_self CHECK (record_id <> target_record_id),
  CONSTRAINT ck_gold_links_gamma_with_layout CHECK ((gamma IS NULL) = (settings_hash IS NULL))
);
CREATE INDEX ix_gold_links_tenant_id_job_id_status ON gold_links (tenant_id, job_id, status);
CREATE INDEX ix_gold_links_target ON gold_links (tenant_id, target_record_id);


-- --- 4. Row-level security ---------------------------------------------------------

DO $rls$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['entity_constraints', 'entity_group_constraints', 'gold_links'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I '
      'USING (tenant_id = current_setting(''app.tenant_id'')::uuid) '
      'WITH CHECK (tenant_id = current_setting(''app.tenant_id'')::uuid)', t);
  END LOOP;
END
$rls$;
