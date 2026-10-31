"""Unit tests for the semantic encoder — TRD 2.2 / FIG 2.4.

No model is downloaded or loaded anywhere in this file. Everything here
exercises `normalize_text`, device resolution, and the parts of
`SemanticEncoder` that are reachable without weights — which is deliberate:
the normaliser is where the subtle bugs live (it decides which strings are
*the same school*), while the model itself is a pinned third-party artefact
whose behaviour is not ours to unit-test.

`test_encode_batch_on_empty_input` is the one method test, and it passes only
because the empty case returns before touching `.model`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.constants import EMBEDDING_DIMENSIONS
from app.ml_pipeline.embedding.encoder import (
    DEFAULT_BATCH_SIZE,
    SemanticEncoder,
    normalize_text,
    resolve_device,
)

# --- Normalisation basics ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Govt. Sr. Sec. School, Hosur", "government senior secondary school hosur"),
        ("DPS Rohini (Sector-24)", "delhi public school rohini sector 24"),
        ("A&B School", "a and b school"),
        ("ADARSH   VIDHYALAYA!!!", "adarsh school"),
        ("Test Valley Sr Sec School", "test valley senior secondary school"),
        ("", ""),
        ("   ...---   ", ""),
    ],
)
def test_normalize_text_expected_output(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent() -> None:
    """Running the ingest normaliser twice must not change the answer.

    Re-running a pipeline stage is routine (TRD 2.3 keeps `raw_row` immutable
    precisely so stages can be re-run). A normaliser that drifted on a second
    pass would quietly re-embed rows into slightly different vectors each time.
    """
    for raw in [
        "Govt. Sr. Sec. School, Hosur",
        "KV No 2 Jalahalli",
        "St. Joseph's Convent High School",
        "Z.P. Pry. School & Anganwadi",
        "केंद्रीय विद्यालय, नई दिल्ली",
    ]:
        once = normalize_text(raw)
        assert normalize_text(once) == once, raw


# --- The convergence property --------------------------------------------------
# These are the tests that matter. Each pair is the same school written two
# ways; if a pair stops agreeing, those rows stop matching and the duplicate
# survives into the tenant's output.


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("KV No 2 Jalahalli", "Kendriya Vidyalaya No. 2 Jalahalli"),
        ("Adarsh Vidya Mandir", "Adarsh Vidyalaya"),
        ("Adarsh Vidhyalaya", "Adarsh Vidyalay"),
        ("Govt. Sr. Sec. School", "Government Senior Secondary School"),
        ("St. Joseph's Convent", "St Josephs Convent"),
        ("Z.P. Pry. School", "ZP Primary School"),
        ("B.R. Ambedkar Memorial School", "BR Ambedkar Memorial School"),
        ("JNV Bangalore", "Jawahar Navodaya Vidyalaya Bangalore"),
    ],
)
def test_spelling_variants_converge(left: str, right: str) -> None:
    assert normalize_text(left) == normalize_text(right)


def test_abbreviations_expand_to_a_fixed_point() -> None:
    """Chained expansion: kv -> kendriya vidyalaya -> kendriya school.

    Single-pass expansion would leave "kv" as "kendriya vidyalaya" while the
    spelled-out form became "kendriya school" — the two most common ways of
    writing the same school, normalised into non-matching strings.
    """
    assert normalize_text("KV") == "kendriya school"
    assert normalize_text("Kendriya Vidyalaya") == "kendriya school"


def test_possessive_apostrophe_does_not_split_a_word() -> None:
    """Regression: "Joseph's" must not become "joseph s".

    Apostrophes are deleted while other punctuation becomes a space. With the
    space rule, "St. Joseph's" and "St Josephs" normalise differently and the
    matcher never sees them as the same institution.
    """
    assert normalize_text("St. Joseph's") == normalize_text("St Josephs") == "saint josephs"


def test_dotted_acronyms_rejoin_before_expansion() -> None:
    """Regression: "Z.P." strips to "z p", which matches no table entry."""
    assert normalize_text("Z.P.") == "zilla parishad"
    assert normalize_text("K.V.") == "kendriya school"


# --- Multilingual --------------------------------------------------------------


@pytest.mark.parametrize(
    "devanagari",
    [
        "केंद्रीय विद्यालय, नई दिल्ली",
        "सरस्वती शिशु मंदिर",
        "राजकीय उच्च विद्यालय",
    ],
)
def test_devanagari_survives_normalisation(devanagari: str) -> None:
    """Regression: combining vowel signs must not be stripped as punctuation.

    Python's `\\w` tests `str.isalnum()`, and Indic matras are category Mn, so
    a `[\\W_]+` punctuation rule deletes them — turning "केंद्रीय" into a row of
    bare consonants. This model is multilingual specifically so that a
    Devanagari row can match its transliterated twin; shredding the script
    first makes that impossible, and does it silently.
    """
    normalised = normalize_text(devanagari)

    # Punctuation gone, every Devanagari code point retained — the block
    # spans U+0900-U+097F, which covers the consonants *and* the combining
    # matras that the buggy rule silently ate.
    assert "," not in normalised
    devanagari_chars = [c for c in devanagari if 0x0900 <= ord(c) <= 0x097F]
    assert devanagari_chars, "test input contains no Devanagari"
    for character in devanagari_chars:
        assert character in normalised, f"{character!r} was dropped"


def test_mixed_script_expands_latin_and_preserves_the_rest() -> None:
    result = normalize_text("KV केंद्रीय विद्यालय No. 2")

    assert result.startswith("kendriya school")
    assert "केंद्रीय" in result
    assert result.endswith("number 2")


# --- Tenant synonym table ------------------------------------------------------


def test_extra_abbreviations_layer_over_the_defaults() -> None:
    """TRD 2.2's per-tenant synonym table, without forking the platform one."""
    assert normalize_text("ABC Intl School", {"intl": "international"}) == (
        "abc international school"
    )
    # The platform defaults still apply alongside the tenant's entries.
    assert normalize_text("Govt Intl School", {"intl": "international"}) == (
        "government international school"
    )


def test_extra_abbreviations_win_on_conflict() -> None:
    assert normalize_text("St Marys", {"st": "street"}) == "street marys"


# --- Device resolution ---------------------------------------------------------


def test_resolve_device_returns_a_supported_backend() -> None:
    assert resolve_device() in {"cuda", "mps", "cpu"}


def test_preferred_device_short_circuits_detection() -> None:
    """A worker pinned to one of several GPUs must get exactly what it asked."""
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


# --- Encoder wiring ------------------------------------------------------------


def test_constructing_an_encoder_does_not_load_the_model() -> None:
    """Construction must stay free — the load is hundreds of MB and seconds."""
    encoder = SemanticEncoder()

    assert encoder._model is None
    assert encoder.batch_size == DEFAULT_BATCH_SIZE


def test_encoder_honours_explicit_configuration() -> None:
    encoder = SemanticEncoder(
        model_name="some/other-model", revision="abc123", device="cpu", batch_size=64
    )

    assert encoder.model_name == "some/other-model"
    assert encoder.revision == "abc123"
    assert encoder.device == "cpu"
    assert encoder.batch_size == 64


def test_encode_batch_on_empty_input() -> None:
    """Shape, not just emptiness: callers vstack batches together.

    A (0,) array will not stack against a (n, 768) one, so an empty batch has
    to carry the right width. Reached without loading the model.
    """
    vectors = SemanticEncoder().encode_batch([])

    assert vectors.shape == (0, EMBEDDING_DIMENSIONS)
    assert vectors.dtype == np.float32
