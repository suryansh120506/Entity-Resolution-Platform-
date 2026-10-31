"""TRD 2.2 / FR-02 — the ingestion classifier: Valid | Junk | Missing_Context.

Every row of an uploaded CSV passes through here before any embedding or
matching work is spent on it. The three classes are decided by an XGBoost
model over five cheap tabular features; the point is to spend a millisecond
sorting a row rather than a GPU-second embedding a row that reads "N/A".

Two design points worth reading before changing anything:

**The cold start is weak supervision, not a guess.** A new tenant has no
steward labels, so iteration 0 trains on pseudo-labels produced by the
Snorkel-style labelling functions below: each LF votes for one class or
abstains, votes are combined by a weighted majority, and rows where the LFs
disagree or all abstain are *dropped from the training set* rather than
labelled by a coin flip. Those rows are exactly the ones a steward should see
first. Pseudo-labels are written with `LabelSource.PSEUDO`, which is what
makes them discardable later without contaminating the ground-truth set
(TRD 2.2, and the `label_source` column added in migration 0003).

**Batch, not row-at-a-time.** `build_feature_frame` is the real entry point:
it resolves the whole batch's pincodes in one query (`warm_pincode_cache`),
builds one DataFrame, and the labelling functions vote as vectorised NumPy
over that frame rather than per row. `extract_features` exists for the
single-row case and delegates to the same code path, so the two can never
drift apart — a feature computed one way in training and another way at
inference is the classic silent model bug, and it is not available here.

FEATURE_NAMES is a model contract: the order is baked into every artefact
ever fitted. Appending is safe, reordering or removing is a retrain.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from app.ml_pipeline.fields import REQUIRED_FIELDS, canonical_fields
from app.ml_pipeline.utils import is_placeholder_pincode, validate_pincode, warm_pincode_cache
from app.models.enums import RecordClassification

logger = logging.getLogger(__name__)

# --- Feature contract ----------------------------------------------------------

#: Column order of the model's input matrix. See the module docstring.
FEATURE_NAMES: tuple[str, ...] = (
    "name_len",
    "name_token_count",
    "missing_required_field_count",
    "pincode_is_placeholder",
    "name_is_placeholder",
)

#: Sentinel strings that occupy a name field without naming anything. Matched
#: against the *whole* trimmed value (`str.fullmatch`) — a school genuinely
#: called "Test Valley High School" must not be caught by the `test` branch.
_PLACEHOLDER_NAME_PATTERN = r"""
    \W*(?:
        n[./\s]?a | nil | none | null | nan | tbd | na
      | test\w* | dummy | sample | unknown | unnamed | no[\s_-]?name
      | x+ | asdf\w* | abc\w* | qwerty
      | school | schools | new[\s_-]?school
      | \d+ | -+ | \.+ | _+
    )\W*
"""

# --- Class encoding ------------------------------------------------------------

#: Fixed positional order for probability columns and label integers. Ties the
#: model's output to the values the `classification` CHECK constraint on
#: school_records accepts, so a prediction is directly writable.
CLASS_ORDER: tuple[RecordClassification, ...] = (
    RecordClassification.VALID,
    RecordClassification.JUNK,
    RecordClassification.MISSING_CONTEXT,
)
VALID, JUNK, MISSING_CONTEXT = 0, 1, 2
N_CLASSES = len(CLASS_ORDER)

#: A labelling function that declines to vote on a row returns this.
ABSTAIN = -1


def build_feature_frame(
    rows: Sequence[Mapping[str, Any]], db: Session | None = None
) -> pd.DataFrame:
    """Feature matrix for a batch, plus the auxiliary columns the LFs read.

    Columns are FEATURE_NAMES (the model's input) followed by `aux_*` columns,
    which exist only so the labelling functions can reason about signals the
    model does not take as features. Select `frame[list(FEATURE_NAMES)]` before
    handing anything to XGBoost.

    `db` is optional. With a session, `pincode_is_placeholder` means "not a
    real, resolvable Indian PIN" — placeholder, malformed, or absent from
    `reference_pincodes`. Without one, only the first two can be detected and
    an unseeded-but-well-formed PIN reads as fine. That degradation is why the
    argument is threaded through rather than the function quietly holding its
    own session.
    """
    fields = pd.DataFrame(
        [canonical_fields(row) for row in rows], columns=list(REQUIRED_FIELDS), dtype="object"
    )
    # A zero-row batch still has to produce a correctly-shaped frame, or the
    # caller's `predict_batch` blows up on an empty job instead of returning
    # nothing.
    if fields.empty:
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in FEATURE_NAMES}
            | {
                "aux_name_has_letters": pd.Series(dtype="bool"),
                "aux_pincode_present": pd.Series(dtype="bool"),
                "aux_has_district": pd.Series(dtype="bool"),
                "aux_longest_word": pd.Series(dtype="float64"),
            }
        )

    name = fields["name"].astype(str)
    pincode = fields["pincode"].astype(str)

    name_is_placeholder = (
        name.str.fullmatch(_PLACEHOLDER_NAME_PATTERN, flags=re.IGNORECASE | re.VERBOSE)
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    if db is not None:
        # One query for the whole batch, then every lookup below is a dict hit.
        warm_pincode_cache(pincode.unique().tolist(), db)
        pincode_bad = ~pincode.map(lambda p: validate_pincode(p, db)).to_numpy(dtype=bool)
    else:
        pincode_bad = pincode.map(is_placeholder_pincode).to_numpy(dtype=bool)

    frame = pd.DataFrame(
        {
            "name_len": name.str.len().to_numpy(dtype="float64"),
            # `count(r"\S+")` rather than `split().str.len()`: same number for
            # whitespace-delimited text, without allocating a list per row.
            "name_token_count": name.str.count(r"\S+").to_numpy(dtype="float64"),
            "missing_required_field_count": (fields == "").sum(axis=1).to_numpy(dtype="float64"),
            "pincode_is_placeholder": pincode_bad.astype("float64"),
            "name_is_placeholder": name_is_placeholder.astype("float64"),
        }
    )
    frame["aux_name_has_letters"] = name.str.contains(r"[^\W\d_]", regex=True).to_numpy(bool)
    frame["aux_pincode_present"] = (pincode != "").to_numpy(bool)
    # Blocking is by district as well as by pincode (blocking/location.py),
    # so a record with a district can be matched without a usable PIN.
    frame["aux_has_district"] = (fields["district"].astype(str) != "").to_numpy(bool)
    # Longest word in the name, punctuation ignored: "D.P.S" and "K V 2" are
    # all short words, "OASIS" and "GURUKUL" are not.
    words = name.str.replace(r"[^\w\s]", " ", regex=True).str.split()
    frame["aux_longest_word"] = words.map(lambda ws: max((len(w) for w in ws), default=0)).to_numpy(
        dtype="float64"
    )
    return frame


def extract_features(row: Mapping[str, Any], db: Session | None = None) -> list[float]:
    """Features for a single row, in FEATURE_NAMES order.

    Delegates to `build_feature_frame` so there is exactly one definition of
    every feature. Batch callers should use that function directly — this one
    pays for a one-row DataFrame, which is irrelevant next to a database round
    trip but not something to do fifty thousand times in a loop.
    """
    frame = build_feature_frame([row], db=db)
    return [float(v) for v in frame.iloc[0][list(FEATURE_NAMES)]]


# --- Snorkel-style weak supervision --------------------------------------------


@dataclass(frozen=True)
class LabelingFunction:
    """One heuristic vote. `fn` returns a class index or ABSTAIN per row.

    `weight` is how much the aggregator trusts this heuristic relative to the
    others. It is a hand-set prior, not a learned accuracy — Snorkel proper
    would estimate LF accuracies from their agreement structure, which needs
    more LFs than are worth writing for a cold start whose whole job is to
    survive until real steward labels exist.
    """

    name: str
    weight: float
    fn: Callable[[pd.DataFrame], np.ndarray]

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fn(frame)


def _vote(condition: np.ndarray, label: int) -> np.ndarray:
    """`label` where the condition holds, ABSTAIN everywhere else."""
    return np.where(condition, label, ABSTAIN).astype(np.int8)


def _has_usable_name(frame: pd.DataFrame) -> np.ndarray:
    """Name present and not a sentinel.

    Every Missing_Context LF gates on this. That class means "a real school
    described too thinly to match", so a row with no name at all — or a name
    of "N/A" — belongs to Junk and these LFs must stay out of it. Without the
    gate a row that is empty in *every* field wins Missing_Context on sheer
    count of absent fields, outvoting the junk LFs that had it right.
    """
    return (frame["name_len"].to_numpy() > 0.0) & (frame["name_is_placeholder"].to_numpy() == 0.0)


def lf_placeholder_name(frame: pd.DataFrame) -> np.ndarray:
    """ "N/A", "TEST", "XXXX", "12345" — occupies the field, names nothing."""
    return _vote(frame["name_is_placeholder"].to_numpy() == 1.0, JUNK)


def lf_empty_name(frame: pd.DataFrame) -> np.ndarray:
    """No name at all. Nothing downstream can match on this row."""
    return _vote(frame["name_len"].to_numpy() == 0.0, JUNK)


def lf_name_has_no_letters(frame: pd.DataFrame) -> np.ndarray:
    """A non-empty name of pure digits/punctuation is not a school name."""
    has_name = frame["name_len"].to_numpy() > 0.0
    return _vote(has_name & ~frame["aux_name_has_letters"].to_numpy(), JUNK)


def lf_name_too_short(frame: pd.DataFrame) -> np.ndarray:
    """Under four characters. Too little to be a name, too much to be blank."""
    name_len = frame["name_len"].to_numpy()
    return _vote((name_len > 0.0) & (name_len < 4.0), JUNK)


def lf_unusable_pincode(frame: pd.DataFrame) -> np.ndarray:
    """A named row with neither a usable PIN nor a district is thin, not junk.

    Only *both* missing makes a row unlocatable. A bad PIN alone used to be
    enough — and on a real 51,668-row export, where most rows carry the
    placeholder "000000" but a full name, address, district and state, that
    set aside 29,794 complete records (58% of the file) as unmatchable. Since
    blocking also partitions by district, those rows can be matched; the bad
    PIN is still reported as a data-quality issue, just not as disqualifying.

    Abstains unless the row has a usable name (`_has_usable_name`) — a
    nameless or sentinel-named row is Junk's.
    """
    return _vote(
        _has_usable_name(frame)
        & (frame["pincode_is_placeholder"].to_numpy() == 1.0)
        & ~frame["aux_has_district"].to_numpy(),
        MISSING_CONTEXT,
    )


def lf_several_missing_fields(frame: pd.DataFrame) -> np.ndarray:
    """Two or more required fields absent."""
    return _vote(
        _has_usable_name(frame) & (frame["missing_required_field_count"].to_numpy() >= 2.0),
        MISSING_CONTEXT,
    )


def lf_one_missing_field(frame: pd.DataFrame) -> np.ndarray:
    """Exactly one required field absent — weak, and weighted accordingly."""
    return _vote(
        _has_usable_name(frame) & (frame["missing_required_field_count"].to_numpy() == 1.0),
        MISSING_CONTEXT,
    )


def lf_complete_record(frame: pd.DataFrame) -> np.ndarray:
    """Every required field present, locatable, name reads like a name.

    "Locatable" is a real PIN *or* a district — see `lf_unusable_pincode`.
    """
    locatable = (frame["pincode_is_placeholder"].to_numpy() == 0.0) | frame[
        "aux_has_district"
    ].to_numpy()
    return _vote(
        (frame["missing_required_field_count"].to_numpy() == 0.0)
        & locatable
        & (frame["name_is_placeholder"].to_numpy() == 0.0)
        & (frame["name_len"].to_numpy() >= 8.0)
        & (frame["name_token_count"].to_numpy() >= 2.0),
        VALID,
    )


def lf_abbreviated_name(frame: pd.DataFrame) -> np.ndarray:
    """Initials only ("D P S", "K.V NO1", "HMY HSS") — a school, thinly named.

    Measured on a real export: 155 such rows, every one a recognisable school
    with a district, were voted Junk by `lf_name_too_short` and the model.
    Initials cannot be matched to the full name ("DPS" vs "Delhi Public
    School"), so they are not usable as-is — but they are "needs more info",
    not "not a school". Outweighs the junk LFs for exactly that reason.
    """
    return _vote(
        _has_usable_name(frame)
        & frame["aux_name_has_letters"].to_numpy()
        & (frame["aux_longest_word"].to_numpy() <= 4.0),
        MISSING_CONTEXT,
    )


def lf_single_word_name_with_location(frame: pd.DataFrame) -> np.ndarray:
    """A real one-word name ("OASIS", "GURUKUL") with a full location.

    Matchable: the name gate compares distinctive words, and "oasis" is the
    distinctive word of "Oasis Public School" in the same district too.
    """
    return _vote(
        _has_usable_name(frame)
        & (frame["name_token_count"].to_numpy() == 1.0)
        & (frame["aux_longest_word"].to_numpy() >= 5.0)
        & (frame["missing_required_field_count"].to_numpy() == 0.0)
        & frame["aux_has_district"].to_numpy(),
        VALID,
    )


def lf_descriptive_name(frame: pd.DataFrame) -> np.ndarray:
    """A long multi-token name ("Govt Higher Primary School, Hosur")."""
    return _vote(
        (frame["name_is_placeholder"].to_numpy() == 0.0)
        & (frame["name_token_count"].to_numpy() >= 3.0)
        & (frame["name_len"].to_numpy() >= 12.0),
        VALID,
    )


#: Weights are priors on how much each heuristic is trusted. The junk LFs are
#: the most certain (a name of "N/A" admits no second reading); the
#: single-missing-field LF is the least (a school with no district recorded is
#: often still resolvable), so it cannot outvote a completeness signal alone.
LABELING_FUNCTIONS: tuple[LabelingFunction, ...] = (
    LabelingFunction("placeholder_name", 2.0, lf_placeholder_name),
    LabelingFunction("empty_name", 2.0, lf_empty_name),
    LabelingFunction("name_has_no_letters", 1.5, lf_name_has_no_letters),
    LabelingFunction("name_too_short", 1.0, lf_name_too_short),
    LabelingFunction("unusable_pincode", 1.5, lf_unusable_pincode),
    LabelingFunction("several_missing_fields", 2.0, lf_several_missing_fields),
    LabelingFunction("one_missing_field", 0.75, lf_one_missing_field),
    LabelingFunction("complete_record", 2.0, lf_complete_record),
    LabelingFunction("descriptive_name", 1.0, lf_descriptive_name),
    # 2.5: must outvote name_too_short (1.0) plus a stray junk vote, since
    # initials are a thin school, not a non-school.
    LabelingFunction("abbreviated_name", 2.5, lf_abbreviated_name),
    LabelingFunction("single_word_name_with_location", 1.5, lf_single_word_name_with_location),
)


def apply_labeling_functions(frame: pd.DataFrame) -> np.ndarray:
    """The Snorkel label matrix: (n_rows, n_lfs) of class index or ABSTAIN."""
    if frame.empty:
        return np.empty((0, len(LABELING_FUNCTIONS)), dtype=np.int8)
    return np.column_stack([lf(frame) for lf in LABELING_FUNCTIONS])


@dataclass(frozen=True)
class PseudoLabels:
    """Weak-supervision output for a batch.

    `labels` holds a class index per row, or ABSTAIN where the LFs said
    nothing. `confidence` is the winning class's share of the total weight
    cast on that row — 1.0 when every voting LF agreed, 0.5 for a two-way
    split, 0.0 when nothing voted.
    """

    labels: np.ndarray
    confidence: np.ndarray
    votes: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of rows any LF voted on at all."""
        if self.labels.size == 0:
            return 0.0
        return float(np.mean(self.labels != ABSTAIN))

    def training_mask(self, min_confidence: float) -> np.ndarray:
        """Rows confident enough to train on."""
        return (self.labels != ABSTAIN) & (self.confidence >= min_confidence)

    def label_counts(self) -> dict[str, int]:
        counts = {cls.value: 0 for cls in CLASS_ORDER}
        for index, cls in enumerate(CLASS_ORDER):
            counts[cls.value] = int(np.sum(self.labels == index))
        counts["ABSTAIN"] = int(np.sum(self.labels == ABSTAIN))
        return counts


def generate_pseudo_labels(frame: pd.DataFrame) -> PseudoLabels:
    """Combine the LF votes into one weak label per row.

    Weighted majority vote. A tie splits the weight evenly, so a contested row
    lands at 0.5 confidence and falls below any sane `min_confidence` — which
    is the intended outcome: an LF disagreement is a row for a steward, not a
    row to guess at.
    """
    votes = apply_labeling_functions(frame)
    weights = np.array([lf.weight for lf in LABELING_FUNCTIONS], dtype="float64")

    n_rows = votes.shape[0]
    scores = np.zeros((n_rows, N_CLASSES), dtype="float64")
    for class_index in range(N_CLASSES):
        scores[:, class_index] = ((votes == class_index) * weights).sum(axis=1)

    total = scores.sum(axis=1)
    voted = total > 0.0
    labels = np.where(voted, scores.argmax(axis=1), ABSTAIN).astype(np.int64)
    confidence = np.divide(
        scores.max(axis=1, initial=0.0), total, out=np.zeros(n_rows), where=voted
    )
    return PseudoLabels(labels=labels, confidence=confidence, votes=votes)


# --- The model -----------------------------------------------------------------


class DataQualityClassifier:
    """XGBoost three-class classifier over FEATURE_NAMES.

    Class integers are fixed by CLASS_ORDER, not by whatever XGBoost inferred
    from the training labels. That distinction matters: a cold-start batch may
    contain no Junk rows at all, and XGBoost's sklearn API requires labels to
    be contiguous from zero. Rather than let the column meaning shift with the
    training data — the failure mode where `predict_proba()[:, 1]` silently
    means Missing_Context in one tenant's model and Junk in another's — the
    fitted classes are remapped on the way in and expanded back to three fixed
    columns on the way out.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
        **xgb_kwargs: Any,
    ) -> None:
        self._model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **xgb_kwargs,
        )
        #: Original class indices the model was fitted on, in the order
        #: XGBoost knows them. None until trained.
        self._fitted_classes: np.ndarray | None = None

    @property
    def is_trained(self) -> bool:
        return self._fitted_classes is not None

    def _fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weight: np.ndarray) -> None:
        present = np.unique(labels)
        if present.size < 2:
            only = CLASS_ORDER[int(present[0])].value if present.size else "none"
            raise ValueError(
                f"Training set contains a single class ({only}). A classifier cannot "
                "be fitted on it, and one that could would be a constant. Widen the "
                "batch or lower min_confidence."
            )

        if present.size < N_CLASSES:
            missing = [CLASS_ORDER[i].value for i in range(N_CLASSES) if i not in present]
            logger.warning(
                "Training set covers only %d of %d classes (missing: %s). The model "
                "cannot predict what it has never seen; those probability columns "
                "will read 0.0.",
                present.size,
                N_CLASSES,
                ", ".join(missing),
            )

        remap = {original: encoded for encoded, original in enumerate(present)}
        encoded_labels = np.array([remap[v] for v in labels], dtype="int32")

        # The objective is settled here rather than in __init__ because it is a
        # function of the data, not of configuration. `multi:softprob` requires
        # num_class >= 1, but XGBoost's sklearn wrapper only sets num_class when
        # it infers more than two classes — so a two-class cold-start batch
        # (a tenant whose first upload happens to contain no Missing_Context
        # rows) fails inside the C library with "value 0 for Parameter
        # num_class" unless the binary objective is used instead. The three-class
        # objective remains what this classifier is *for*; this is the narrow
        # degradation that keeps iteration 0 trainable, and `_expand_proba` puts
        # the columns back where CLASS_ORDER says they go either way.
        objective = "multi:softprob" if present.size > 2 else "binary:logistic"
        self._model.set_params(objective=objective)

        self._model.fit(features[list(FEATURE_NAMES)], encoded_labels, sample_weight=sample_weight)
        self._fitted_classes = present

    def train_on_pseudo_labels(
        self,
        rows: Sequence[Mapping[str, Any]],
        db: Session | None = None,
        *,
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Iteration 0: fit on weak labels, with no human ground truth.

        Rows the LFs abstained on or disagreed about are excluded rather than
        forced into a class — see `generate_pseudo_labels`. Surviving rows are
        weighted by their LF agreement, so a unanimous row pulls harder on the
        fit than a 2-to-1 one.

        Returns a summary suitable for `model_versions.metrics`. The caller is
        responsible for stamping the trained rows with `LabelSource.PSEUDO`;
        this function does not write to the database.
        """
        if not rows:
            raise ValueError("train_on_pseudo_labels called with no rows")

        frame = build_feature_frame(rows, db=db)
        pseudo = generate_pseudo_labels(frame)
        mask = pseudo.training_mask(min_confidence)
        n_trainable = int(mask.sum())

        if n_trainable == 0:
            raise ValueError(
                f"No row reached min_confidence={min_confidence}: the labelling "
                f"functions abstained or disagreed on all {len(rows)} rows. This is "
                "a batch for human review, not a batch to train on."
            )

        self._fit(frame[mask], pseudo.labels[mask], pseudo.confidence[mask])
        fitted_classes = self._fitted_classes
        assert fitted_classes is not None  # _fit sets it or raises

        summary = {
            "n_rows": len(rows),
            "n_trained_on": n_trainable,
            "n_excluded": len(rows) - n_trainable,
            "lf_coverage": round(pseudo.coverage, 4),
            "mean_confidence": round(float(pseudo.confidence[mask].mean()), 4),
            "min_confidence": min_confidence,
            "pseudo_label_counts": pseudo.label_counts(),
            "classes_fitted": [CLASS_ORDER[int(i)].value for i in fitted_classes],
        }
        logger.info("Trained on pseudo-labels: %s", summary)
        return summary

    def predict_batch(
        self, rows: Sequence[Mapping[str, Any]], db: Session | None = None
    ) -> pd.DataFrame:
        """Classify a batch.

        Returns one row per input, with `classification` carrying the exact
        strings the `classification` CHECK constraint on `school_records`
        accepts, `confidence` the winning probability, and one probability
        column per class in CLASS_ORDER.
        """
        if not self.is_trained:
            raise RuntimeError(
                "DataQualityClassifier has not been trained. Call "
                "train_on_pseudo_labels() for a cold start, or load a fitted "
                "artefact from the tenant's active model_versions row."
            )

        frame = build_feature_frame(rows, db=db)
        proba = self._expand_proba(frame)
        best = proba.argmax(axis=1)

        return pd.DataFrame(
            {
                "classification": [CLASS_ORDER[i].value for i in best],
                "confidence": proba.max(axis=1, initial=0.0),
                **{
                    f"p_{cls.value.lower()}": proba[:, index]
                    for index, cls in enumerate(CLASS_ORDER)
                },
            }
        )

    def _expand_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Model output, widened back to N_CLASSES fixed columns.

        A model fitted on two classes returns two probability columns; this
        puts them back where CLASS_ORDER says they belong and leaves the
        unseen class at zero.
        """
        assert self._fitted_classes is not None  # guarded by is_trained
        n_rows = len(frame)
        full = np.zeros((n_rows, N_CLASSES), dtype="float64")
        if n_rows == 0:
            return full
        fitted = self._model.predict_proba(frame[list(FEATURE_NAMES)])
        full[:, self._fitted_classes] = fitted
        return full

    # --- Persistence -----------------------------------------------------------
    # One file, because `model_versions.artifact_uri` is one URI. The two
    # things this class knows that the booster does not — which classes it was
    # fitted on, and the feature order it was fitted against — ride along as
    # booster attributes inside the same XGBoost JSON rather than in a sidecar
    # that could go missing, or be copied without its partner, and leave a
    # model whose probability columns mean something other than they claim.

    #: Booster attribute keys. Namespaced so they cannot collide with anything
    #: XGBoost or the sklearn wrapper writes into the same section.
    _ATTR_FITTED_CLASSES = "er_fitted_classes"
    _ATTR_FEATURE_NAMES = "er_feature_names"

    def save(self, path: str) -> None:
        """Persist to XGBoost's native JSON format.

        JSON rather than pickle: a pickled sklearn wrapper is tied to the
        exact xgboost/sklearn versions that wrote it, which is not a property
        you want in an artefact a retrain job writes today and a worker loads
        six months and two dependency bumps from now.
        """
        if not self.is_trained:
            raise RuntimeError("Refusing to save an untrained DataQualityClassifier.")
        assert self._fitted_classes is not None  # guarded above

        booster = self._model.get_booster()
        booster.set_attr(
            **{
                self._ATTR_FITTED_CLASSES: json.dumps([int(c) for c in self._fitted_classes]),
                self._ATTR_FEATURE_NAMES: json.dumps(list(FEATURE_NAMES)),
            }
        )
        self._model.save_model(path)
        logger.info("Saved classifier to %s", path)

    @classmethod
    def load(cls, path: str) -> DataQualityClassifier:
        """Rebuild a classifier from an artefact written by `save`.

        Refuses an artefact whose feature order differs from the current
        FEATURE_NAMES. That mismatch is survivable at the array level — the
        shapes still line up — and catastrophic at the meaning level, because
        the model would read `name_len` out of the column now holding
        `missing_required_field_count` and score every row confidently wrong.
        A loud failure here is the only version of this that is debuggable.
        """
        instance = cls()
        instance._model.load_model(path)

        booster = instance._model.get_booster()
        raw_classes = booster.attr(cls._ATTR_FITTED_CLASSES)
        raw_features = booster.attr(cls._ATTR_FEATURE_NAMES)
        if raw_classes is None or raw_features is None:
            raise ValueError(
                f"{path} is missing the attributes this class writes "
                f"({cls._ATTR_FITTED_CLASSES}, {cls._ATTR_FEATURE_NAMES}). It was "
                "not written by DataQualityClassifier.save(), and loading it would "
                "produce a model whose probability columns cannot be interpreted."
            )

        saved_features = json.loads(raw_features)
        if saved_features != list(FEATURE_NAMES):
            raise ValueError(
                f"{path} was fitted on a different feature contract.\n"
                f"  artefact: {saved_features}\n"
                f"  current:  {list(FEATURE_NAMES)}\n"
                "Retrain — the columns no longer mean what the model thinks."
            )

        instance._fitted_classes = np.array(json.loads(raw_classes), dtype="int64")
        return instance
