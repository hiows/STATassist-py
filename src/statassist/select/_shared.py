"""Internal helpers shared by the two searches.

A search is handed candidates rather than predictors and answers which of them to
keep, so its row axis is ``candidates`` and its result is
:class:`~statassist.core.SaSelection`. What the two searches have in common is
the part before the search: which model was asked for, whether the outcome agrees
with it, and how a set of per-candidate numbers becomes a ranking. That is what
is here.

Everything about the *data* is shared with the model functions instead and comes
from :mod:`statassist.fit._shared`: the listwise deletion, the dummy coding, the
resampling scheme and the fits themselves. A search is a wrapper around fits, so
reimplementing any of that here would be a second answer to a question the model
functions already answer.

The one thing this file decides that ``fit_*`` does not is who has the first say
about the kind of outcome. :func:`~statassist.fit._shared.resolve_outcome` reads
the outcome and decides; here ``model`` is already an answer to the question the
outcome would have been asked, so a disagreement between the two is an error
naming the model that would have fitted rather than a silently different
analysis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.contracts import selection_ranking_columns
from ..core.errors import SaValueError, notify
from ..fit._shared import (
    N_CLASSES,
    ModelInput,
    design_lv,
    encode_outcome,
    outcome_levels,
)

__all__ = [
    "SearchOutcome",
    "SearchWords",
    "check_choice",
    "ranking_table",
    "resolve_search_outcome",
    "search_design",
]


@dataclass(frozen=True)
class SearchWords:
    """The sentences that name one search rather than the other.

    The checks below are the same three checks in both searches and differ only
    in what they suggest instead: an elimination can rank with a forest and a
    stepwise search cannot, so the two cannot name the same way out. Holding the
    wording beside the check keeps one copy of the logic without making either
    message vaguer than the one R prints.

    Attributes:
        procedure: What to call the search where ``outcome_levels`` names it, as
            in "a recursive feature elimination".
        linear_refusal: Why ``model="linear"`` cannot take class labels, and
            which models can.
        logistic_refusal: Which models take a continuous outcome instead.
        numeric_note: Which models would have read a two-valued numeric column as
            classes.
        non_finite: Why a non-finite outcome cannot be searched on, in this
            search's own terms.
    """

    procedure: str
    linear_refusal: str
    logistic_refusal: str
    numeric_note: str
    non_finite: str


@dataclass
class SearchOutcome:
    """The outcome as the search reads it, once ``model`` has had its say.

    The counterpart of :class:`~statassist.fit._shared.Outcome`, and it holds the
    same four things for the same reasons.

    Attributes:
        classify: Whether the search is classifying.
        y: The outcome as the fits take it, floats for a regression and 0/1 for a
            classification.
        levels: The two classes, reference first, or ``None``.
        n_events: Rows in ``levels[1]``, or ``None``.
    """

    classify: bool
    y: np.ndarray
    levels: list[str] | None
    n_events: int | None


def check_choice(value: Any, allowed: Sequence[str], arg: str) -> str:
    """Resolve a string argument against its choices, R's ``match.arg()``.

    Both searches take several, and for ``model`` the choices differ between
    them: a criterion is a likelihood with a charge against its parameter count,
    and a forest has neither, so an elimination can rank with one and a stepwise
    search cannot.
    """
    if value not in allowed:
        raise SaValueError(f"`{arg}` must be one of: " + ", ".join(allowed) + ".")
    return str(value)


def resolve_search_outcome(
    input_: ModelInput,
    model: str,
    outcome_lv: Any,
    control_label: Any,
    words: SearchWords,
) -> SearchOutcome:
    """Decide what kind of outcome the search has, with ``model`` going first.

    Port of the block both ``perform_rfe()`` and ``perform_stepwise()`` open
    with. Naming a classifying model, naming the classes, or handing over a
    column that is not numeric all ask for a classification; everything else is a
    regression.

    A numeric column taking two values is the one case where both readings are
    plausible, and it is searched as a regression with a note saying how to ask
    for the other reading - the same call
    :func:`~statassist.fit._shared.resolve_outcome` makes, so that a search and a
    fit read the same column the same way.
    """
    y = input_.y
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    classify = (
        model == "logistic" or outcome_lv is not None or control_label is not None or not numeric
    )
    distinct = int(y.nunique())

    if model == "linear" and classify:
        raise SaValueError(words.linear_refusal)
    if model == "logistic" and numeric and distinct > N_CLASSES:
        raise SaValueError(
            '`model = "logistic"` classifies two classes, and `outcome` is a numeric '
            f"column taking {distinct} values. " + words.logistic_refusal
        )
    if not classify and distinct == N_CLASSES:
        notify(
            "`outcome` is numeric and takes two values, so it was searched as a "
            "regression. Pass `control_label`, or a categorical column, with "
            + words.numeric_note
            + " to treat it as a classification."
        )

    if not classify:
        values = y.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SaValueError(f"`outcome` holds non-finite value(s), {words.non_finite}")
        return SearchOutcome(classify=False, y=values, levels=None, n_events=None)

    levels = outcome_levels(y, outcome_lv, control_label, words.procedure)
    encoded = encode_outcome(y, levels)
    return SearchOutcome(classify=True, y=encoded, levels=levels, n_events=int(encoded.sum()))


def search_design(input_: ModelInput, outcome: SearchOutcome) -> dict[str, Any]:
    """The ``design`` slot: what the search saw.

    The same entries in the same order as a model's ``design``, since a selection
    is read beside the fit that follows it, and built here rather than shared with
    :func:`~statassist.fit._shared.model_design` because the outcome it reports is
    a :class:`SearchOutcome` rather than an
    :class:`~statassist.fit._shared.Outcome`.
    """
    design: dict[str, Any] = {
        "outcome": input_.outcome,
        "outcome_type": "two classes" if outcome.classify else "continuous",
    }
    if outcome.classify:
        design["outcome_lv"] = outcome.levels
        design["n_events"] = outcome.n_events
        design["event_rate"] = (outcome.n_events or 0) / input_.n_used
    design["n_obs"] = input_.n_obs
    design["n_used"] = input_.n_used
    design["n_dropped"] = input_.n_dropped
    design["predictors"] = input_.predictors
    design["dropped_predictors"] = input_.dropped_predictors
    design.update(design_lv(input_.predictor_lv))
    return design


def ranking_table(
    estimate: Mapping[str, float], predictors: Sequence[str], selected: Sequence[str]
) -> pd.DataFrame:
    """One row per candidate, most important first.

    Port of the tail both ``sa_rfe_ranking()`` and ``sa_step_ranking()`` end
    with. The candidates are put in alphabetical order first and then sorted by
    the estimate with a stable sort, so ties come out by name: a search over a
    set of candidates that are worth the same has to give the same ranking twice.

    A candidate nothing could answer for keeps a missing estimate and sorts last,
    rather than being given a zero that reads as a measurement.
    """
    candidates = sorted(str(name) for name in predictors)
    values = np.array([float(estimate.get(name, math.nan)) for name in candidates])
    # Descending on the estimate is ascending on its negation, and a missing
    # estimate goes to the end whichever way the finite ones are ordered.
    key = np.where(np.isnan(values), math.inf, -values)
    at = np.argsort(key, kind="stable")

    kept = set(str(name) for name in selected)
    ordered = [candidates[position] for position in at]
    table = pd.DataFrame(
        {
            "candidates": ordered,
            "estimate": values[at],
            "rank": np.arange(1, len(ordered) + 1, dtype=int),
            "selected": [name in kept for name in ordered],
        }
    )
    return table[selection_ranking_columns()]
