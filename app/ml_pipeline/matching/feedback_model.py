"""Bayesian update of Splink m-probabilities from steward labels.

Fellegi-Sunter's m for a comparison level is P(level | the pair is a match).
Labelled matches observed at each level are therefore exactly the evidence
for m, and they arrive pre-counted: the `pair_labels` trigger maintains
`splink_level_stats.match_w`, the (inverse-propensity weighted) number of
labelled matches seen at each (comparison, level).

The update is a Dirichlet-multinomial posterior with the EM estimate as the
prior, worth `prior_strength` pseudo-labels:

    m_post[l] = (count[l] + strength * m_prior[l]) / (sum(count) + strength)

Ten labels nudge a comparison; a thousand dominate it; zero leave it exactly
as EM fitted it. Cost is O(levels): nothing is re-scanned or re-fitted.

u is deliberately left alone. u = P(level | non-match) is estimated by
random sampling, which is unbiased; labelled non-matches come from the
review band and are the opposite of a random sample.

Level numbering follows Splink's settings JSON: the null level (if any) has
gamma -1; the remaining levels are listed most-similar first, so the k-th
non-null level of n has gamma n-1-k.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

#: No level's m may reach zero: Splink's Bayes factor m/u would be 0 and the
#: log weight -inf, so a single unseen level would veto every pair hitting it.
M_FLOOR = 1e-4


def level_gammas(levels: list[dict[str, Any]]) -> list[int | None]:
    """Gamma value of each level, in settings order (None for the null level)."""
    non_null = sum(1 for level in levels if not level.get("is_null_level"))
    out: list[int | None] = []
    k = 0
    for level in levels:
        if level.get("is_null_level"):
            out.append(None)
        else:
            out.append(non_null - 1 - k)
            k += 1
    return out


def _normalised_prior(levels: list[dict[str, Any]]) -> list[float]:
    """EM's m per non-null level; levels EM never observed share the rest."""
    raw = [level.get("m_probability") for level in levels]
    known = sum(p for p in raw if p is not None)
    missing = sum(1 for p in raw if p is None)
    fill = max(0.0, 1.0 - known) / missing if missing else 0.0
    prior = [max(M_FLOOR, p if p is not None else fill) for p in raw]
    total = sum(prior)
    return [p / total for p in prior]


def blend_m_probabilities(
    settings: Mapping[str, Any],
    match_counts: Mapping[tuple[int, int], float],
    *,
    prior_strength: float,
) -> tuple[dict[str, Any], float]:
    """Settings with m updated from `match_counts[(comparison_idx, gamma)]`.

    Returns the new settings and the evidence weight applied (the largest
    per-comparison label total — every labelled pair contributes to every
    comparison where it was not null).
    """
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    updated = copy.deepcopy(dict(settings))
    evidence = 0.0
    for idx, comparison in enumerate(updated["comparisons"]):
        levels = comparison["comparison_levels"]
        gammas = level_gammas(levels)
        scored = [(level, g) for level, g in zip(levels, gammas, strict=True) if g is not None]
        counts = [max(0.0, float(match_counts.get((idx, g), 0.0))) for _, g in scored]
        total = sum(counts)
        if total <= 0:
            continue
        evidence = max(evidence, total)
        prior = _normalised_prior([level for level, _ in scored])
        for (level, _), count, p in zip(scored, counts, prior, strict=True):
            posterior = (count + prior_strength * p) / (total + prior_strength)
            level["m_probability"] = max(M_FLOOR, posterior)
    return updated, evidence
