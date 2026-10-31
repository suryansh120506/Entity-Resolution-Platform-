"""FR-02 / TRD 2.2 — the ingestion classifier stage.

`classifier.py` holds the whole stage: feature extraction, the Snorkel-style
labelling functions that cold-start it, and the XGBoost model itself. Nothing
is re-exported here on purpose — importing this package pulls in xgboost and
pandas, which SAD 3.2 keeps out of the API process, so the import should be
an explicit `from app.ml_pipeline.classification.classifier import ...` at the
point of use rather than something a stray `import app.ml_pipeline` drags in.
"""
