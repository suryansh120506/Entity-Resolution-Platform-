"""FR-04 / TRD 2.2 — probabilistic scoring of the blocker's candidate pairs.

Splink on the DuckDB backend. DuckDB because the whole job fits in one
process at the 50k-row scale this platform targets, and an in-process
columnar engine beats shipping the same comparison SQL back to Postgres and
competing with the API for its connection pool.

**The score is Splink's, not the embedding distance.** SAD 3.3 is explicit
about this and it is worth restating: the blocker's cosine distance says two
records are *worth comparing*, which is a different claim from *these are the
same school*. Cosine distance between two generic government school names is
small whether or not they are the same institution. The Fellegi-Sunter model
here weighs field agreements against how discriminating each agreement is —
matching on a rare name means far more than matching on a common one — and
`clusters.confidence_score` is that output.

**Scoring an arbitrary pair set.** Splink normally chooses its own pairs via
blocking rules, but ours are already chosen: the HNSW search produced them and
re-deriving them in SQL would mean maintaining the same blocking logic twice,
in two languages, with no way to notice when they drift apart. So
`score_pairs` pivots the candidate list into one group per pair and blocks on
that group id, which makes Splink compare exactly the pairs it was handed and
nothing else.

That pivot duplicates a record once per pair it appears in, which is why
**model parameters are never estimated on the pivoted frame** — `train()`
fits on the real, deduplicated record set and `score_pairs` only applies the
result. Estimating `u` on the pivoted frame would sample two copies of the
same record as if they were a random pair, teach the model that random pairs
agree on everything, and collapse every match weight toward zero. For the
same reason term-frequency adjustments are off by default: they would count a
name once per candidate pair rather than once per record.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from splink import Linker

logger = logging.getLogger(__name__)

#: Columns `RecordMatcher` needs on the record frame. These are the normalised
#: forms written by the ingest stage (app/ml_pipeline/embedding/encoder.py),
#: not raw CSV values — comparing raw text would score "Govt." against
#: "Government" as a disagreement.
REQUIRED_RECORD_COLUMNS = ("record_id", "name_normalized", "address_normalized", "pincode")

#: Columns `score_pairs` needs. Produced directly by
#: `CandidateBlocker.candidate_frame`.
REQUIRED_PAIR_COLUMNS = ("record_id_a", "record_id_b")

#: Splink's own name for the group column `score_pairs` blocks on.
_PAIR_KEY = "er_pair_key"

#: `clusters.confidence_score` is NUMERIC(4,3) with a CHECK constraining it to
#: [0, 1], so scores are rounded to three decimals here rather than at insert
#: time. Rounding at the boundary would let a value that passed every check in
#: Python fail in the database.
CONFIDENCE_DECIMALS = 3

#: Comparison-level keys that hold trained parameters rather than definition.
_TRAINED_LEVEL_KEYS = frozenset(
    {
        "m_probability",
        "u_probability",
        "tf_adjustment_weight",
        "fix_m_probability",
        "fix_u_probability",
    }
)


def _pincode_comparison() -> Any:
    """Indian PINs compared by their hierarchy, not just exact equality.

    A 6-digit PIN is not an opaque token: the first digit is the postal
    region, the first three identify the sorting district, and only the last
    three narrow it to a delivery office. Two records differing in the last
    three digits are plausibly the same school recorded against neighbouring
    offices; two differing in the first digit are a thousand kilometres apart.
    `ExactMatch` throws that structure away and scores both as simply
    "different", which costs real recall on a field that is otherwise one of
    the most discriminating available.
    """
    import splink.comparison_level_library as cll
    from splink.comparison_library import CustomComparison

    # The SQL below references `pincode_l` / `pincode_r`, not `l.pincode` /
    # `r.pincode`. By the time a comparison level is evaluated, Splink has
    # already joined the pair into one row with suffixed columns, and the
    # `l.`/`r.` form — which *is* correct in a blocking rule — fails there
    # with "Referenced table l not found". The two contexts read alike and
    # take different syntax.
    return CustomComparison(
        output_column_name="pincode",
        comparison_description="Exact, then sorting district, then region",
        comparison_levels=[
            cll.NullLevel("pincode"),
            cll.ExactMatchLevel("pincode"),
            cll.CustomLevel(
                "substr(pincode_l, 1, 3) = substr(pincode_r, 1, 3)",
                label_for_charts="Same sorting district (first 3 digits)",
            ),
            cll.CustomLevel(
                "substr(pincode_l, 1, 1) = substr(pincode_r, 1, 1)",
                label_for_charts="Same postal region (first digit)",
            ),
            cll.ElseLevel(),
        ],
    )


def default_comparisons() -> list[Any]:
    """The three-field comparison set from TRD 2.2.

    Built by a function rather than held as a module constant because
    constructing these imports splink, and importing splink pulls in DuckDB —
    a cost the ingest and API processes should not pay for merely importing
    something else from this package.
    """
    import splink.comparison_library as cl

    return [
        # Purpose-built for names: exact, then jaro-winkler bands, so
        # "saint josephs" vs "saint joseph" is a near-match rather than a miss.
        cl.NameComparison("name_normalized"),
        # Addresses are longer and noisier than names, so the thresholds are
        # looser than NameComparison's defaults.
        cl.JaroWinklerAtThresholds("address_normalized", [0.92, 0.85]),
        _pincode_comparison(),
    ]


def default_training_rules() -> list[Any]:
    """Blocking rules used only for EM training, never for prediction.

    EM needs a slice of pairs dense enough in true matches to learn `m` from.
    Two rules, each holding a different field fixed, so that the parameters
    for one field are estimated on pairs not selected for agreeing on it.
    """
    from splink import block_on

    return [block_on("pincode"), block_on("name_normalized")]


def default_deterministic_rules() -> list[Any]:
    """Rules whose matches are true beyond reasonable doubt.

    Used only to anchor `probability_two_random_records_match` — the prior
    that scales every match weight. Without it the model has no idea how
    common duplicates are, and a genuine duplicate scores a few percent
    because the prior it is updating from is essentially zero.

    These have to be *precise*, not complete: they are paired with a `recall`
    estimate that accounts for the matches they miss.
    """
    from splink import block_on

    return [
        block_on("name_normalized", "pincode"),
        block_on("name_normalized", "address_normalized"),
    ]


def _describe_rule(rule: Any) -> str:
    """Readable SQL for a blocking rule, for the training summary.

    `str()` on a rule object yields `<... object at 0x7f...>`, and this
    summary is written to `model_versions.metrics` — a memory address stored
    as a training record is noise that also changes on every run.
    """
    try:
        return str(rule.get_blocking_rule("duckdb").blocking_rule_sql)
    except Exception:
        return str(rule)


class RecordMatcher:
    """Fellegi-Sunter scoring over a fixed record set.

    The record frame is supplied once at construction because every operation
    needs it: training estimates parameters from it, and scoring reads the
    field values out of it. `score_pairs` then takes only the pair list, which
    is what the blocker produces.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        comparisons: list[Any] | None = None,
        training_rules: list[Any] | None = None,
        deterministic_rules: list[Any] | None = None,
        deterministic_recall: float = 0.8,
        max_pairs_for_u: float = 1_000_000,
        random_seed: int = 42,
    ) -> None:
        missing = [c for c in REQUIRED_RECORD_COLUMNS if c not in records.columns]
        if missing:
            raise ValueError(
                f"records is missing required column(s): {', '.join(missing)}. "
                f"Expected {list(REQUIRED_RECORD_COLUMNS)}."
            )

        self._records = records.copy()
        # Splink keys on `unique_id`; ours is `record_id`. Stringified because
        # the ids are UUIDs and DuckDB round-trips them more predictably as
        # text than as a native UUID through pandas.
        self._records["record_id"] = self._records["record_id"].astype(str)
        self._records["unique_id"] = self._records["record_id"]

        self._comparisons = comparisons if comparisons is not None else default_comparisons()
        self._training_rules = (
            training_rules if training_rules is not None else default_training_rules()
        )
        self._deterministic_rules = (
            deterministic_rules
            if deterministic_rules is not None
            else default_deterministic_rules()
        )
        if not 0.0 < deterministic_recall <= 1.0:
            raise ValueError(
                f"deterministic_recall must be within (0, 1], got {deterministic_recall}"
            )
        self.deterministic_recall = deterministic_recall
        self.max_pairs_for_u = max_pairs_for_u
        self.random_seed = random_seed
        self._trained_settings: dict[str, Any] | None = None

    @property
    def is_trained(self) -> bool:
        return self._trained_settings is not None

    @property
    def model_dict(self) -> dict[str, Any]:
        """The trained Splink settings — the artefact `model_versions` stores."""
        if self._trained_settings is None:
            raise RuntimeError("RecordMatcher has not been trained.")
        return dict(self._trained_settings)

    def comparison_layout(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """(comparison names, non-null level count per comparison, layout).

        The layout is the comparison definitions with every trained parameter
        (m, u, the prior) stripped out. Gamma vectors mean the same thing
        under any two models sharing a layout, so this, not the trained
        settings, is what `splink_settings.settings_hash` identifies. Hashing
        the trained model instead would give each per-job retrain its own
        hash and scatter the feedback statistics across them.
        """
        return layout_of(self.model_dict)

    def untrained_layout(self) -> tuple[list[str], list[int], dict[str, Any]]:
        """The layout this matcher's comparisons define, before any training.

        What a stored base model is checked against: a model fitted under a
        different comparison set produces gammas that mean something else,
        and must not be reused.
        """
        from splink import block_on

        linker = self._linker(self._records, self._settings([block_on("pincode")]))
        return layout_of(linker.misc.save_model_to_json())

    def adopt(self, settings: dict[str, Any]) -> None:
        """Use previously trained settings instead of training (no EM)."""
        if len(settings.get("comparisons", ())) != len(self._comparisons):
            raise ValueError("stored settings do not match this matcher's comparisons")
        self._trained_settings = copy.deepcopy(settings)

    def reestimate_prior(self) -> bool:
        """Re-fit only `probability_two_random_records_match` on this job.

        The duplicate rate is a property of the upload, not of the tenant: a
        clean registry export and a merge of three overlapping lists differ
        by orders of magnitude. It is one count over deterministic-rule
        blocks — no EM — so it is re-estimated per job while m and u are
        reused. Returns False (and keeps the stored prior) if the rules find
        nothing to count on this job.
        """
        linker = self._linker(self._records, copy.deepcopy(self.model_dict))
        try:
            linker.training.estimate_probability_two_random_records_match(
                self._deterministic_rules, recall=self.deterministic_recall
            )
        except Exception:
            logger.warning("Prior re-estimation failed; keeping the stored prior", exc_info=True)
            return False
        estimated = linker.misc.save_model_to_json()
        self._trained_settings = self._floor_prior(estimated) or estimated
        return True

    #: Weak prior when the deterministic rules count nothing: this share of
    #: records is assumed to have one duplicate. m/u evidence dominates it.
    FALLBACK_DUPLICATE_RATE = 0.1

    def _floor_prior(self, settings: dict[str, Any]) -> dict[str, Any] | None:
        """Replace a collapsed prior; None when the estimate is usable.

        The prior is counted from exact (name, PIN) / (name, address)
        agreements. An upload whose duplicates are all near-misses ("School"
        vs "Schol") has none, the estimate comes out ~0, and every pair then
        scores ~0 however strong its evidence. An estimate implying fewer
        than one duplicate pair in the whole upload is not an estimate; a
        weak, explicit prior is used instead.
        """
        n = len(self._records)
        pairs = n * (n - 1) / 2
        estimate = float(settings.get("probability_two_random_records_match") or 0.0)
        if pairs <= 0 or estimate * pairs >= 1.0:
            return None
        floored = copy.deepcopy(settings)
        floored["probability_two_random_records_match"] = min(
            0.5, self.FALLBACK_DUPLICATE_RATE * n / 2 / pairs
        )
        logger.warning(
            "Prior estimate %.3g implies <1 duplicate in %d records; using %.3g",
            estimate,
            n,
            floored["probability_two_random_records_match"],
        )
        return floored

    # --- Internals -------------------------------------------------------------

    def _settings(self, blocking_rules: list[Any]) -> Any:
        from splink import SettingsCreator

        return SettingsCreator(
            link_type="dedupe_only",
            comparisons=self._comparisons,
            blocking_rules_to_generate_predictions=blocking_rules,
            # Carried through to the prediction output so results map back to
            # real records without parsing them out of the synthetic ids
            # `score_pairs` builds.
            additional_columns_to_retain=["record_id"],
            retain_intermediate_calculation_columns=False,
        )

    def _linker(self, frame: pd.DataFrame, settings: Any) -> Linker:
        from splink import DuckDBAPI, Linker

        # Splink annotates `input_table_or_tables` as `str | list[str]`, but
        # its runtime contract is wider and passing a DataFrame is the
        # documented usage for the DuckDB backend. The annotation is narrower
        # than the behaviour, so the ignore is on their hint, not on ours.
        return Linker(frame, settings, db_api=DuckDBAPI())  # type: ignore[arg-type]

    # --- Training --------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Splink's three-step recipe: the prior, then `u`, then `m`.

        Fitted on the real record set — see the module docstring on why this
        must never run against the frame `score_pairs` builds.

        Returns a summary for `model_versions.metrics`.
        """
        from splink import block_on

        linker = self._linker(self._records, self._settings([block_on("pincode")]))

        # 1. The prior: how often do two randomly drawn records match at all?
        #    Everything else is an update to this number, so skipping it is
        #    not a missing refinement — it leaves genuine duplicates scoring a
        #    few percent because they are updating from a prior of ~0.
        linker.training.estimate_probability_two_random_records_match(
            self._deterministic_rules, recall=self.deterministic_recall
        )
        floored = self._floor_prior(linker.misc.save_model_to_json())
        if floored is not None:
            linker = self._linker(self._records, floored)

        # 2. u = P(fields agree | the pair is NOT a match), learned by sampling
        #    random pairs, which at any realistic duplicate rate are
        #    overwhelmingly non-matches.
        linker.training.estimate_u_using_random_sampling(
            max_pairs=self.max_pairs_for_u, seed=self.random_seed
        )

        # 3. m = P(fields agree | the pair IS a match), by EM.
        trained_on: list[str] = []
        for rule in self._training_rules:
            # One rule can fail to converge on a small or skewed job without
            # the others being affected; losing one rule's contribution is a
            # weaker model, while aborting the retrain leaves the tenant on a
            # stale one. Log loudly and keep the rest.
            #
            # A failed EM session can leave the linker's settings half-
            # mutated (Splink sets aside the comparisons a training rule
            # blocks on, and an exception skips putting them back), so the
            # rule runs against a snapshot and a failure restores it. Without
            # this a single failed rule saved a model missing comparisons —
            # and with base models now persisted, every later job adopted it.
            snapshot = linker.misc.save_model_to_json()
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(rule)
                trained_on.append(_describe_rule(rule))
            except Exception:
                logger.exception(
                    "EM training failed for blocking rule %s; skipping", _describe_rule(rule)
                )
                linker = self._linker(self._records, snapshot)

        if not trained_on:
            raise RuntimeError(
                "Every EM training rule failed; the model has u-values but no "
                "estimated m-values and would score every pair on priors alone."
            )

        trained = linker.misc.save_model_to_json()
        if len(trained["comparisons"]) != len(self._comparisons):
            raise RuntimeError(
                f"Trained model has {len(trained['comparisons'])} comparisons, expected "
                f"{len(self._comparisons)}; refusing to save a partial model."
            )
        self._trained_settings = trained
        summary = {
            "n_records": int(len(self._records)),
            "em_rules_trained": trained_on,
            "em_rules_failed": len(self._training_rules) - len(trained_on),
            "deterministic_recall": self.deterministic_recall,
            "max_pairs_for_u": self.max_pairs_for_u,
        }
        logger.info("Trained matcher: %s", summary)
        return summary

    # --- Scoring ---------------------------------------------------------------

    def _pivot_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """One group of two rows per candidate pair, keyed by `_PAIR_KEY`.

        This is what lets a blocking rule select exactly the pairs we were
        handed: each group holds precisely the two records of one pair, so
        `l.<key> = r.<key>` yields that pair and nothing else.
        """
        pairs = candidate_pairs.copy()
        pairs["record_id_a"] = pairs["record_id_a"].astype(str)
        pairs["record_id_b"] = pairs["record_id_b"].astype(str)
        pairs[_PAIR_KEY] = [f"p{i}" for i in range(len(pairs))]

        attributes = self._records.drop(columns=["unique_id"])
        left = pairs.merge(attributes, left_on="record_id_a", right_on="record_id", how="inner")
        right = pairs.merge(attributes, left_on="record_id_b", right_on="record_id", how="inner")

        expected = len(pairs)
        if len(left) != expected or len(right) != expected:
            missing = expected - min(len(left), len(right))
            raise ValueError(
                f"{missing} candidate pair(s) reference record ids absent from the "
                "record frame. The blocker and the matcher are looking at "
                "different record sets."
            )

        pivoted = pd.concat([left, right], ignore_index=True)
        # Unique per row, not per record: a record appears once per pair it
        # belongs to, and Splink requires unique_id to be unique.
        pivoted["unique_id"] = pivoted[_PAIR_KEY] + "|" + pivoted["record_id"]
        return pivoted

    def score_pairs(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """Score the blocker's pairs.

        Returns one row per input pair with `confidence_score` in [0, 1]
        rounded to three decimals (`clusters.confidence_score` is
        NUMERIC(4,3)), alongside Splink's raw `match_weight` in log2 Bayes
        factors, which is the number to look at when asking *why* a pair
        scored as it did.

        Any extra columns on the input — the blocker's `distance` and
        `source` — are carried through untouched, so the vector evidence and
        the probabilistic score stay side by side for review.
        """
        if not self.is_trained:
            raise RuntimeError(
                "RecordMatcher has not been trained. Call train(), or load a "
                "fitted model from the tenant's active model_versions row."
            )

        missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in candidate_pairs.columns]
        if missing:
            raise ValueError(
                f"candidate_pairs is missing required column(s): {', '.join(missing)}."
            )

        result_columns = [
            "record_id_a",
            "record_id_b",
            *[c for c in candidate_pairs.columns if c not in REQUIRED_PAIR_COLUMNS],
            "confidence_score",
            "match_weight",
            "gamma",
        ]
        if candidate_pairs.empty:
            return pd.DataFrame(columns=result_columns)

        pivoted = self._pivot_pairs(candidate_pairs)

        scoring_settings = dict(self.model_dict)
        scoring_settings["blocking_rules_to_generate_predictions"] = [
            {"blocking_rule": f"l.{_PAIR_KEY} = r.{_PAIR_KEY}", "sql_dialect": "duckdb"}
        ]
        # Keeps the gamma_* comparison-vector columns in the output: the
        # feedback loop learns from them (pair_scores.gamma).
        scoring_settings["retain_intermediate_calculation_columns"] = True
        linker = self._linker(pivoted, scoring_settings)
        predictions = linker.inference.predict().as_pandas_dataframe()

        gamma_columns = [f"gamma_{name}" for name in self.comparison_layout()[0]]
        predictions["gamma"] = [
            [int(v) for v in row] for row in predictions[gamma_columns].itertuples(index=False)
        ]
        scored = predictions[
            ["record_id_l", "record_id_r", "match_probability", "match_weight", "gamma"]
        ]
        scored = scored.rename(
            columns={
                "record_id_l": "_scored_a",
                "record_id_r": "_scored_b",
                "match_probability": "confidence_score",
            }
        )

        # Splink orders each comparison by unique_id, which is prefixed with
        # the pair key and so need not preserve the input's a/b orientation.
        # Merging on an order-independent key keeps the caller's orientation.
        scored["_key"] = [
            _pair_key(a, b) for a, b in zip(scored["_scored_a"], scored["_scored_b"], strict=True)
        ]
        out = candidate_pairs.copy()
        out["record_id_a"] = out["record_id_a"].astype(str)
        out["record_id_b"] = out["record_id_b"].astype(str)
        out["_key"] = [
            _pair_key(a, b) for a, b in zip(out["record_id_a"], out["record_id_b"], strict=True)
        ]

        merged = out.merge(
            scored[["_key", "confidence_score", "match_weight", "gamma"]], on="_key", how="left"
        ).drop(columns=["_key"])
        merged["confidence_score"] = (
            merged["confidence_score"].astype(float).clip(0.0, 1.0).round(CONFIDENCE_DECIMALS)
        )
        return merged[result_columns]

    # --- Active learning -------------------------------------------------------

    def update_model_from_labels(self, labels: pd.DataFrame) -> dict[str, Any]:
        """Re-estimate `m` from human decisions (FR-08 / SAD 3.6).

        `labels` needs `record_id_a`, `record_id_b` and `clerical_match_score`
        in [0, 1] — the shape `human_feedback` reduces to, with MERGE as 1.0
        and REJECT as 0.0.

        Only `m` is re-estimated: `m` is P(fields agree | the pair IS a
        match), which is exactly what a steward's verdict is evidence about.
        `u` is a property of the population rather than of any labelled pair,
        and re-estimating it from a review queue — which is selected for being
        hard, not for being representative — would bias it badly. This is
        additive, so the nightly job should hold the previous artefact until
        the new one clears SAD 3.6's promotion gate.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Train the matcher before refining it from labels: "
                "estimate_m_from_pairwise_labels refines an existing model."
            )

        required = ("record_id_a", "record_id_b", "clerical_match_score")
        missing = [c for c in required if c not in labels.columns]
        if missing:
            raise ValueError(f"labels is missing required column(s): {', '.join(missing)}.")
        if labels.empty:
            raise ValueError("labels is empty; there is nothing to learn from.")

        prepared = pd.DataFrame(
            {
                "unique_id_l": labels["record_id_a"].astype(str),
                "unique_id_r": labels["record_id_b"].astype(str),
                "clerical_match_score": labels["clerical_match_score"].astype(float),
            }
        )
        if not prepared["clerical_match_score"].between(0.0, 1.0).all():
            raise ValueError("clerical_match_score values must lie within [0, 1].")

        # Rebuilt from the trained settings so the refinement starts from the
        # current model rather than from scratch.
        linker = self._linker(self._records, dict(self.model_dict))
        labels_table = linker.table_management.register_labels_table(prepared, overwrite=True)
        linker.training.estimate_m_from_pairwise_labels(labels_table)

        self._trained_settings = linker.misc.save_model_to_json()
        summary = {
            "n_labels": int(len(prepared)),
            "n_positive": int((prepared["clerical_match_score"] >= 0.5).sum()),
            "n_negative": int((prepared["clerical_match_score"] < 0.5).sum()),
        }
        logger.info("Refined matcher from labels: %s", summary)
        return summary

    # --- Persistence -----------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the trained settings to JSON for `model_versions.artifact_uri`."""
        import json

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.model_dict, handle, indent=2)
        logger.info("Saved matcher settings to %s", path)

    def load(self, path: str) -> None:
        """Adopt settings written by `save`, skipping training.

        An instance method rather than a constructor because the record frame
        is the other half of a usable matcher and is not part of the artefact.
        """
        import json

        with open(path, encoding="utf-8") as handle:
            self._trained_settings = json.load(handle)


#: Blocking for gold linking: same PIN, same standardised name, or the same
#: stable identity. Splink drops pairs an earlier rule already produced.
LINK_BLOCKING_COLUMNS: tuple[str, ...] = ("pincode", "name_normalized", "stable_id")


def link_against_pool(
    settings: dict[str, Any],
    incoming: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    deterministic_rules: list[Any] | None = None,
    deterministic_recall: float = 0.8,
    expected_overlap: float = 0.5,
) -> pd.DataFrame:
    """Splink `link_only`: score incoming records against a pool, no EM.

    Both frames carry record_id, name_normalized, address_normalized,
    pincode and stable_id. `settings` is the tenant's trained model (with
    steward feedback already blended in); only its link type, blocking and
    output options are changed, so m and u are exactly the ones in-upload
    matching used.

    The prior is re-counted for the link task, because "a random incoming
    record matches a random gold record" is a different base rate from
    "two random records in one upload match". Splink counts it from exact
    (name, PIN) / (name, address) agreements; on a small or novel upload
    there may be none, and the estimate collapses to ~0 — a prior that
    vetoes every link however strong the evidence. So an estimate implying
    fewer than one match across the whole pool is replaced by a weakly
    informative one: `expected_overlap` of incoming records already exist,
    i.e. λ = expected_overlap / |pool|. m/u evidence dominates it quickly.

    Returns one row per scored pair: incoming_id, pool_id,
    match_probability, match_weight, gamma (list in comparison order).
    """
    from splink import DuckDBAPI, Linker

    columns = ["incoming_id", "pool_id", "match_probability", "match_weight", "gamma"]
    if incoming.empty or pool.empty:
        return pd.DataFrame(columns=columns)

    link_settings = copy.deepcopy(settings)
    link_settings["link_type"] = "link_only"
    link_settings["unique_id_column_name"] = "record_id"
    link_settings["retain_intermediate_calculation_columns"] = True
    link_settings["additional_columns_to_retain"] = []
    link_settings["blocking_rules_to_generate_predictions"] = [
        {"blocking_rule": f'l."{c}" = r."{c}"', "sql_dialect": "duckdb"}
        for c in LINK_BLOCKING_COLUMNS
    ]

    frames = [frame.assign(record_id=frame["record_id"].astype(str)) for frame in (incoming, pool)]

    def make_linker(config: dict[str, Any]) -> Linker:
        # DataFrames are Splink's documented DuckDB input; its hint says str.
        return Linker(
            frames,  # type: ignore[arg-type]
            config,
            db_api=DuckDBAPI(),
            input_table_aliases=["incoming", "pool"],
        )

    comparisons = len(incoming) * len(pool)
    weak_prior = min(0.5, expected_overlap / len(pool))
    estimate = 0.0
    try:
        estimator = make_linker(copy.deepcopy(link_settings))
        estimator.training.estimate_probability_two_random_records_match(
            deterministic_rules or default_deterministic_rules(), recall=deterministic_recall
        )
        estimate = float(
            estimator.misc.save_model_to_json()["probability_two_random_records_match"]
        )
    except Exception:
        logger.warning("Link prior estimation failed; using the weak prior", exc_info=True)
    link_settings["probability_two_random_records_match"] = (
        estimate if estimate * comparisons >= 1.0 else weak_prior
    )

    predictions = make_linker(link_settings).inference.predict().as_pandas_dataframe()
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    names, _, _ = layout_of(link_settings)
    gamma_columns = [f"gamma_{name}" for name in names]
    left_is_incoming = predictions["source_dataset_l"] == "incoming"
    return pd.DataFrame(
        {
            "incoming_id": predictions["record_id_l"].where(
                left_is_incoming, predictions["record_id_r"]
            ),
            "pool_id": predictions["record_id_r"].where(
                left_is_incoming, predictions["record_id_l"]
            ),
            "match_probability": predictions["match_probability"].astype(float),
            "match_weight": predictions["match_weight"].astype(float),
            "gamma": [
                [int(v) for v in row] for row in predictions[gamma_columns].itertuples(index=False)
            ],
        }
    )[columns]


def layout_of(settings: dict[str, Any]) -> tuple[list[str], list[int], dict[str, Any]]:
    """(comparison names, non-null level counts, parameter-free layout)."""
    names: list[str] = []
    counts: list[int] = []
    layout: list[dict[str, Any]] = []
    for comparison in settings["comparisons"]:
        names.append(comparison["output_column_name"])
        levels = [
            {k: v for k, v in level.items() if k not in _TRAINED_LEVEL_KEYS}
            for level in comparison["comparison_levels"]
        ]
        counts.append(sum(1 for level in levels if not level.get("is_null_level")))
        layout.append({"output_column_name": names[-1], "comparison_levels": levels})
    return names, counts, {"comparisons": layout}


def layout_hash(layout: dict[str, Any]) -> str:
    """Stable identity of a comparison layout (splink_settings.settings_hash)."""
    canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pair_key(left: str, right: str) -> tuple[str, str]:
    """Orientation-independent key for a pair of record ids."""
    return (left, right) if left <= right else (right, left)


def labels_from_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    """Turn `human_feedback` rows into the frame `update_model_from_labels` wants.

    MERGE is unambiguous evidence the pair matches; REJECT that it does not.
    SPLIT is deliberately dropped: it is a verdict about a *cluster* being
    over-merged, and which specific pair within it was wrong is recorded in
    `excluded_record_ids` rather than being recoverable here. Treating a SPLIT
    as a pairwise negative would label the pairs the steward kept as
    non-matches.
    """
    scores = {"MERGE": 1.0, "REJECT": 0.0}
    usable = feedback[feedback["decision"].isin(scores)].copy()
    usable["clerical_match_score"] = usable["decision"].map(scores)
    return usable[["record_id_a", "record_id_b", "clerical_match_score"]].reset_index(drop=True)
