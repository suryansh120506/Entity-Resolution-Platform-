"""TRD 2.2 — semantic vectorisation for the name and address strings.

Two independent 768-dim vectors per record (FIG 2.4), produced by
paraphrase-multilingual-mpnet-base-v2. Multilingual because Indian school
names arrive transliterated inconsistently and sometimes not transliterated
at all; the same school may be "Kendriya Vidyalaya" in one row and
"केंद्रीय विद्यालय" in another, and a model that only speaks Latin script
scores those as unrelated.

**Normalisation is a separate stage from encoding, on purpose.** TRD 2.2 has
the ingest stage write `name_normalized` / `address_normalized`, and the
embedding stage reads those columns. So `encode_batch` embeds exactly the
strings it is handed and never silently re-normalises: if it did, the vector
in the database would correspond to a string that appears nowhere in the
database, and debugging "why did these two rows not match" would mean
reproducing a hidden transform. `normalize_text` is the function that
produces those columns, exported from here because the abbreviation table is
the same knowledge either way.

**Imports are deferred.** torch and sentence-transformers together cost a
couple of seconds and hundreds of MB to import. The ingest stage wants
`normalize_text` and nothing else, so neither is imported at module level —
only inside the code paths that actually touch a model.

Embeddings come back L2-normalised, which is what makes the `vector_cosine_ops`
HNSW indexes in migration 0001 behave: on unit vectors cosine distance and
inner product agree, so the index and any ad-hoc similarity query cannot
disagree about which neighbour is closest.
"""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import get_settings
from app.core.constants import EMBEDDING_DIMENSIONS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

#: SAD 3.3's throughput target assumes the GPU is fed in large chunks. 256 is
#: comfortable for a 768-dim mpnet on a mid-range card and harmless on CPU,
#: where the batch size mostly affects peak memory rather than speed.
DEFAULT_BATCH_SIZE = 256


# --- Normalisation -------------------------------------------------------------

#: Platform-default abbreviation expansions for Indian school names.
#:
#: TRD 2.2 describes this as "the tenant's synonym table", so this dictionary
#: is the floor, not the whole story — `normalize_text` takes an
#: `extra_abbreviations` mapping so a tenant's own table can be layered on top
#: without forking this one.
#:
#: Expansion runs to a fixed point (see `_expand`), so entries may expand into
#: other entries: "kv" -> "kendriya vidyalaya" -> "kendriya school". That is
#: what makes "KV Jalahalli" and "Kendriya Vidyalaya Jalahalli" converge on
#: the same string instead of embedding as two different schools.
ABBREVIATIONS: dict[str, str] = {
    # Governance
    "govt": "government",
    "gov": "government",
    "gvt": "government",
    "mpl": "municipal",
    "mun": "municipal",
    "zp": "zilla parishad",
    # School type / level
    "sr sec": "senior secondary",
    "srsec": "senior secondary",
    "hr sec": "higher secondary",
    "hrsec": "higher secondary",
    "hs": "high school",
    "hss": "higher secondary school",
    "ps": "primary school",
    "ups": "upper primary school",
    "pu": "pre university",
    "sr": "senior",
    "jr": "junior",
    "pry": "primary",
    "sec": "secondary",
    # Vernacular terms for "school". Collapsing these is the single highest
    # -value expansion here: the same institution is written half a dozen ways
    # and the model should not have to learn that they are synonyms.
    "vidyalaya": "school",
    "vidyalay": "school",
    "vidhyalaya": "school",
    "vidya mandir": "school",
    "vidyamandir": "school",
    "shikshan sanstha": "school",
    "pathshala": "school",
    "patshala": "school",
    "shala": "school",
    "madrasa": "school",
    "madrassa": "school",
    # Well-known networks
    "kv": "kendriya vidyalaya",
    "jnv": "jawahar navodaya vidyalaya",
    "dps": "delhi public school",
    "dav": "dayanand anglo vedic",
    # Common name tokens
    "st": "saint",  # "St. Josephs" — not "street"; see the module tests.
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "eng": "english",
    "med": "medium",
    "co ed": "co educational",
    "coed": "co educational",
    "int": "international",
    "pub": "public",
    "conv": "convent",
    "acad": "academy",
    "inst": "institute",
}

#: How many expansion passes before giving up. Three is already more chaining
#: than the table contains ("kv" -> "kendriya vidyalaya" -> "kendriya school"
#: is two); the cap exists so that a future entry pair that expands into each
#: other cannot hang the ingest stage.
_MAX_EXPANSION_PASSES = 5

#: Punctuation is stripped by *naming* it, rather than by keeping `\w` and
#: discarding the rest. That inversion is load-bearing for the multilingual
#: case this model exists to serve: Python's `\w` tests `str.isalnum()`, and
#: Indic combining vowel signs (Unicode category Mn — the matras and virama in
#: "केंद्रीय") are not alphanumeric. A `[\W_]+` rule therefore deletes the
#: vowel marks out of every Devanagari string and leaves a row of bare
#: consonants, which is not the same word and does not embed like it. Listing
#: the punctuation instead leaves every script intact by default.
_PUNCTUATION_CHARS = frozenset(string.punctuation) | frozenset(
    "–—‒―…“”„‘’«»·•‹›¡¿।॥"  # typographic dashes/quotes plus the Devanagari danda
)
_PUNCTUATION_RE = re.compile("[" + re.escape("".join(sorted(_PUNCTUATION_CHARS))) + "]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: Apostrophes are deleted rather than turned into a space, unlike every other
#: punctuation mark. "St. Joseph's" and "St Josephs" are the same school, and
#: the space-substituting rule would make them "joseph s" and "josephs" — two
#: strings that differ in exactly the way the matcher is least able to forgive.
_APOSTROPHE_RE = re.compile(r"[’'`ʼ]")

#: A run of two or more single letters is an acronym that lost its dots:
#: "Z.P." survives punctuation stripping as "z p", which matches no entry in
#: a table keyed on "zp". Re-joining them before expansion is what lets
#: "K.V.", "KV" and "Kendriya Vidyalaya" all land on the same string.
#:
#: Deliberately ASCII-only. Dotted acronyms are a Latin-script convention,
#: and a Unicode-letter version of this rule would fuse genuinely separate
#: short words in scripts that have them — including, if the punctuation rule
#: above ever regresses, every Devanagari string that got shredded into
#: single consonants.
_SPACED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\s+)+[a-z]\b")


@lru_cache(maxsize=8)
def _compiled_table(items: tuple[tuple[str, str], ...]) -> tuple[re.Pattern[str], dict[str, str]]:
    """Compile an abbreviation table into one alternation regex.

    Longest key first so that "sr sec" wins over "sr", and "vidya mandir" over
    a hypothetical "vidya" — Python's `|` is first-match, not longest-match,
    so the ordering here is load-bearing rather than cosmetic.
    """
    table = dict(items)
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pattern, table


def _expand(text: str, table: Mapping[str, str]) -> str:
    """Apply the abbreviation table until the string stops changing."""
    pattern, lookup = _compiled_table(tuple(sorted(table.items())))
    for _ in range(_MAX_EXPANSION_PASSES):
        expanded = pattern.sub(lambda m: lookup[m.group(0)], text)
        if expanded == text:
            return text
        text = expanded
    logger.warning(
        "Abbreviation expansion did not converge in %d passes for %r — check the "
        "table for entries that expand into each other.",
        _MAX_EXPANSION_PASSES,
        text[:80],
    )
    return text


def normalize_text(text: str, extra_abbreviations: Mapping[str, str] | None = None) -> str:
    """Lowercase, strip punctuation, expand abbreviations, collapse whitespace.

    This is what produces `school_records.name_normalized` and
    `address_normalized`. Order matters and is not arbitrary: punctuation is
    stripped *before* expansion so that "Govt." and "Sr. Sec." reduce to the
    bare tokens the table is keyed on, and whitespace is collapsed *after*
    expansion because expansions introduce spaces of their own.

    `extra_abbreviations` layers a tenant's synonym table over the platform
    default (TRD 2.2); its entries win on conflict.

    Returns "" for empty or punctuation-only input rather than raising — a
    blank name is the classifier's problem to label, not this function's to
    reject.
    """
    if not text:
        return ""

    lowered = text.lower().replace("&", " and ")
    depunctuated = _PUNCTUATION_RE.sub(" ", _APOSTROPHE_RE.sub("", lowered))
    collapsed = _WHITESPACE_RE.sub(" ", depunctuated).strip()
    if not collapsed:
        return ""

    dedotted = _SPACED_ACRONYM_RE.sub(lambda m: m.group(0).replace(" ", ""), collapsed)

    table = ABBREVIATIONS if not extra_abbreviations else {**ABBREVIATIONS, **extra_abbreviations}
    return _WHITESPACE_RE.sub(" ", _expand(dedotted, table)).strip()


# --- Hardware ------------------------------------------------------------------


def resolve_device(preferred: str | None = None) -> str:
    """Pick the fastest device present: CUDA, then Apple MPS, then CPU.

    `preferred` short-circuits the detection, which matters for a worker
    pinned to one of several GPUs ("cuda:1") and for reproducing a CPU result
    on a machine that happens to have a card.
    """
    if preferred:
        return preferred

    import torch

    if torch.cuda.is_available():
        return "cuda"
    # `torch.backends.mps` exists on every 2.x build, but guard anyway: this
    # runs on developer laptops and CI images alike, and a missing attribute
    # should degrade to CPU rather than crash the embedding stage.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# --- The encoder ---------------------------------------------------------------


class SemanticEncoder:
    """Wraps the sentence-transformers model behind a batch interface.

    The model is loaded lazily, on first encode. Constructing an encoder is
    therefore free, which is what lets a Celery worker build one at import
    time and only pay the (multi-second, multi-hundred-MB) load if a task
    actually needs to embed something.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.revision = revision if revision is not None else settings.embedding_model_revision
        self.batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

        if not self.revision:
            # TRD 2.2 requires pinning by revision hash, not by name. Warned
            # rather than raised because local development against a freshly
            # cloned repo has no revision configured yet, and refusing to run
            # would make the first embedding run a configuration exercise.
            logger.warning(
                "%s is not pinned to a revision (EMBEDDING_MODEL_REVISION is unset). "
                "TRD 2.2 requires a pinned revision: an upstream model update would "
                "otherwise silently change the meaning of every vector already stored.",
                self.model_name,
            )

    @property
    def device(self) -> str:
        """Resolved lazily so that constructing an encoder never imports torch."""
        if self._device is None:
            self._device = resolve_device()
        return self._device

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading %s (revision=%s) onto %s",
                self.model_name,
                self.revision or "UNPINNED",
                self.device,
            )
            self._model = SentenceTransformer(
                # `revision=""` (an env var present but empty, as opposed to
                # unset) builds a malformed `resolve//<file>` URL against the
                # Hub and fails every request — reported as a generic
                # connection error with no hint that the revision was the
                # cause. `None` lets huggingface_hub resolve the default
                # branch the way "no revision configured" is supposed to.
                self.model_name,
                device=self.device,
                revision=self.revision or None,
            )
        return self._model

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch. Returns float32 of shape (len(texts), 768).

        Embeds the strings as given — see the module docstring on why this
        does not normalise. Callers pass `name_normalized` /
        `address_normalized`, not raw CSV values.

        Vectors are L2-normalised to match the cosine operator class on the
        HNSW indexes, and returned as float32 because that is what pgvector
        stores; converting here rather than at insert time keeps the
        conversion in one place and out of the hot insert loop.
        """
        if len(texts) == 0:
            # Shape matters even when empty: callers stack this with other
            # batches, and a (0,) array will not stack against a (n, 768) one.
            return np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape[1] != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"{self.model_name} produced {vectors.shape[1]}-dim vectors, but the "
                f"schema's vector({EMBEDDING_DIMENSIONS}) columns (FIG 2.4) cannot "
                "hold them. Changing the model is a migration plus a full "
                "re-embedding backfill, not a configuration change."
            )
        return vectors
