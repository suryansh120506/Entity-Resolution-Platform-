"""Queue tasks, one module per stage.

Nothing is imported here. The worker loads these modules through
`app.queue.registry.TASK_MODULES`; importing them at package scope would mean
`import app.workers.tasks` pulls in xgboost, torch and DuckDB, which SAD 3.2
keeps out of the API process.

    ingest.py       ingest.ingest_and_classify          FR-01 / FR-02
    silver.py       silver.standardize                  ontology -> Silver
    enrichment.py   enrichment.enrich_job               natural-key enrichment
    embeddings.py   embeddings.embeddings_stage         FIG 2.4
    blocking.py     blocking.blocking_stage             FR-03
    matching.py     matching.probabilistic_match        FR-04
    resolution.py   resolution.resolve_by_confidence    FR-05/06/07
    jobs.py         jobs.finalize_reviewed              AWAITING_REVIEW -> COMPLETED
"""
