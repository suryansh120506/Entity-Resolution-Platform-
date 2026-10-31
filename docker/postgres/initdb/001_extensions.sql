-- ============================================================================
-- ER-PLATFORM-001 · initdb 001 — extensions
-- Runs once, as the owner role, on first initialisation of an empty volume.
-- ============================================================================

-- pgvector: backs record_embeddings.name_embedding / .address_embedding
-- (TRD FIG 2.4) and the HNSW candidate search in the `blocking` stage.
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid() is core since PostgreSQL 13, so pgcrypto is NOT required
-- for FIG 2.1's DEFAULT gen_random_uuid(). Kept out deliberately — one less
-- extension in the trust surface.

-- Record what we installed, so a drifted environment is diagnosable from
-- Postgres alone (the same principle as dead_letter_tasks in TRD 2.7).
DO $$
BEGIN
  RAISE NOTICE 'pgvector installed: version %',
    (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END
$$;
