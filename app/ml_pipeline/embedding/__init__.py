"""TRD 2.2 / FIG 2.4 — semantic vectorisation stage.

`encoder.py` holds `SemanticEncoder` (the model wrapper) and `normalize_text`
(the abbreviation/punctuation normalisation that produces the
`*_normalized` columns the encoder reads). Nothing is re-exported here:
importing the encoder module is cheap, but touching `SemanticEncoder.model`
pulls in torch and sentence-transformers, which SAD 3.2 keeps out of the API
process entirely.
"""
