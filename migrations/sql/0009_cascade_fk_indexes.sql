-- ============================================================================
-- ER-PLATFORM-001 · migration 0009 — index every cascading foreign key
--
-- Postgres does not index a foreign key's referencing column. Deleting a
-- parent row runs one internal query per child table, per parent row:
--
--     DELETE FROM pair_scores WHERE record_l = $1
--
-- With no index on record_l that is a sequential scan of the whole table,
-- once per deleted record. Deleting a 51,668-record job therefore read each
-- child table 51,668 times (billions of rows in total), and a delete that
-- should take seconds ran for the better part of an hour.
--
-- Every column below is the *leading* column of a foreign key into jobs,
-- school_records or clusters that no existing index leads with. Several
-- (pair_scores, pair_labels, gold_links, feedback_events, task_queue,
-- silver_records) are tables added after migration 0005.
-- ============================================================================

CREATE INDEX IF NOT EXISTS ix_school_records_job_id        ON school_records (job_id);
CREATE INDEX IF NOT EXISTS ix_silver_records_job_id        ON silver_records (job_id);
CREATE INDEX IF NOT EXISTS ix_clusters_job_id              ON clusters (job_id);
CREATE INDEX IF NOT EXISTS ix_clusters_master_record_id    ON clusters (master_record_id)
  WHERE master_record_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_cluster_members_record_id    ON cluster_members (record_id);
CREATE INDEX IF NOT EXISTS ix_candidate_pairs_record_id_a  ON candidate_pairs (record_id_a);
CREATE INDEX IF NOT EXISTS ix_candidate_pairs_record_id_b  ON candidate_pairs (record_id_b);
CREATE INDEX IF NOT EXISTS ix_pair_scores_record_l         ON pair_scores (record_l);
CREATE INDEX IF NOT EXISTS ix_pair_labels_record_l         ON pair_labels (record_l);
CREATE INDEX IF NOT EXISTS ix_pair_labels_record_r         ON pair_labels (record_r);
CREATE INDEX IF NOT EXISTS ix_gold_links_record_id         ON gold_links (record_id);
CREATE INDEX IF NOT EXISTS ix_gold_links_target_record_id  ON gold_links (target_record_id);
CREATE INDEX IF NOT EXISTS ix_human_feedback_cluster_id    ON human_feedback (cluster_id);
CREATE INDEX IF NOT EXISTS ix_human_feedback_master_record_id ON human_feedback (master_record_id)
  WHERE master_record_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_feedback_events_cluster_id   ON feedback_events (cluster_id)
  WHERE cluster_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_feedback_events_job_id       ON feedback_events (job_id)
  WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_task_queue_job_id            ON task_queue (job_id)
  WHERE job_id IS NOT NULL;
