-- Reverses 0008. Identity-keyed constraints cannot be mapped back to the
-- record ids of any one job, so they are dropped, not converted.

DROP TABLE IF EXISTS gold_links;
DROP TABLE IF EXISTS entity_group_constraints;
DROP TABLE IF EXISTS entity_constraints;

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
ALTER TABLE entity_constraints ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_constraints FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON entity_constraints
  USING (tenant_id = current_setting('app.tenant_id')::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);

DROP INDEX IF EXISTS ix_silver_records_tenant_id_stable_id;
ALTER TABLE silver_records DROP CONSTRAINT IF EXISTS ck_silver_records_stable_id_shape;
ALTER TABLE silver_records DROP CONSTRAINT IF EXISTS ck_silver_records_identity_kind_valid;
ALTER TABLE silver_records DROP CONSTRAINT IF EXISTS ck_silver_records_identity_pair;
ALTER TABLE silver_records DROP COLUMN IF EXISTS identity_kind;
ALTER TABLE silver_records DROP COLUMN IF EXISTS stable_id;
