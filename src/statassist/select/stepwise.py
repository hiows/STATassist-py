"""Stepwise selection: take the one move that lowers the criterion the most.

The port of ``R/perform_stepwise.R``. The second search here and the first that
holds nothing out. :func:`~statassist.perform_rfe` scores every subset size on
rows that did not choose it, which is what a resampled score is for. This one has
no resampled score at all: what it compares candidates by is a penalised
likelihood, the fit's own log likelihood with a charge levied per parameter, so a
predictor has to earn its coefficient before the criterion will keep it.

That is why there is no ``cv`` argument and no ``seed``. Nothing is resampled, so
nothing is random and the same rows give the same path every time. It is also why
the criterion is not a performance claim: it is computed on the rows the model was
fitted to, and the model was chosen because it scored best on them, so the number
that chose the model cannot also be an honest estimate of it.

The charge per parameter is the whole difference between the two criteria. AIC
levies 2 and BIC levies ``log(n)``, so past seven observations BIC charges more
and keeps fewer predictors. Both are computed at every step whichever one is
searching, so a path chosen by one can be read against the other.

What this does not have to do is fold dummy columns back onto the columns they
came from, which is most of the work inside :func:`~statassist.perform_rfe`. A
move here is a whole candidate, and a factor is one candidate however many dummy
columns it becomes, so what enters and leaves the model is always a column of the
input and ``selected`` is a set of names a ``fit_*`` call takes as it stands.

R runs ``stats::step()``, which searches on :func:`stats::extractAIC`'s scale.
That differs from :func:`stats::AIC`'s by a constant which is the same for every
model on ``n`` rows, so the two order the path identically and only the printed
values differ. The walk below searches on the ``AIC()`` scale directly, which is
the scale ``profile`` reports and the one
:func:`~statassist.fit_linear_regression` puts in ``fit_stats``: a step of the
path and a fitted model are then the same number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.contracts import stepwise_profile_columns
from ..core.errors import SaValueError
from ..core.result import SaSelection, new_selection
from ..core.validate import fmt_est
from ..fit._shared import (
    ModelInput,
    design_matrix,
    least_squares,
    logistic_fit,
    logistic_scores,
    quiet_engine,
    resolve_model_input,
    search_label,
)
from ._shared import (
    SearchOutcome,
    SearchWords,
    check_choice,
    ranking_table,
    resolve_search_outcome,
    search_design,
)

__all__ = ["perform_stepwise"]

#: What can be fitted at every step, in the order R lists them.
#:
#: No forest, unlike :data:`~statassist.select.rfe.RFE_MODELS`: a criterion is a
#: likelihood with a charge against its parameter count, and a forest has neither.
STEPWISE_MODELS = ("linear", "logistic")

#: The criteria the moves can be judged by, in the order R lists them.
CRITERIA = ("AIC", "BIC")

#: Which moves are allowed, in the order R lists them.
DIRECTIONS = ("backward", "both", "forward")

#: The charge per parameter AIC levies.
AIC_CHARGE = 2.0

#: How much a move has to lower the criterion by before it is taken.
#:
#: ``stats::step()``'s own tolerance. Without it a move worth nothing but a
#: rounding error is an improvement, and the search walks on comparing models it
#: cannot tell apart.
STEP_TOL = 1e-7

#: What this port changed about the procedure, for ``engine["overridden"]``.
_OVERRIDDEN = ("search on stats::AIC()'s scale rather than stats::extractAIC()'s",)

#: The sentences that name this search where a message has to name one.
_WORDS = SearchWords(
    procedure="a stepwise selection",
    linear_refusal=(
        '`model = "linear"` searches for the predictors of a number, and `outcome` is '
        'a set of class labels. Use `model = "logistic"` for a two-class outcome.'
    ),
    logistic_refusal='Use `model = "linear"` for a continuous outcome.',
    numeric_note='`model = "logistic"`',
    non_finite="which a model fitted at each step has no likelihood for.",
)


def perform_stepwise(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    model: str = STEPWISE_MODELS[0],
    criterion: str = CRITERIA[0],
    direction: str = DIRECTIONS[0],
) -> SaSelection:
    """Stepwise feature selection by information criteria.

    Runs a stepwise search: the model is refitted with each candidate term taken
    out or put in, the move that lowers AIC or BIC the most is taken, and the
    search stops when no single move lowers it any further. What comes back is the
    predictors of the model it stopped at, the path it walked to get there, and
    what each candidate is worth to that model, so a set of four predictors can be
    read against what the other six would have cost.

    The input is the wide format the model functions take, one row per observation
    with one column as the outcome, and is normally the training half of a
    :func:`~statassist.split_data` result.

    **What the criterion charges.** Both criteria are the fit's own log likelihood
    with a charge levied per parameter, and they differ only in the size of the
    charge: ``"AIC"`` levies 2 and ``"BIC"`` levies ``log(n)``, so past seven
    observations BIC charges more and keeps fewer predictors. Smaller is better
    for both, which is why ``maximize`` is ``False`` and is reported rather than
    asked for. The charge as it was used is ``parameters["k"]``.

    Whichever one searches, ``profile`` reports both at every step, so a path
    chosen by AIC can be read against what BIC would have said about the same
    models. A criterion is comparable across models only when they were fitted to
    the same rows, which is what the listwise deletion is for. It is not
    comparable across outcomes, across transformations of an outcome, or between a
    linear and a logistic fit, so the numbers in ``profile`` say which of *these*
    models to prefer and nothing more.

    **What the search is not.** The criterion is computed on the rows the model
    was fitted to, and the model was kept because it scored best on them. Three
    things follow, and none of them is a fault of the implementation. The
    criterion is not a validation, so score the selection on the test half of
    :func:`~statassist.split_data`, which the search never saw. The p-values of
    the selected model are no longer honest, since a coefficient kept because it
    was significant enough to survive is being tested against a null it was
    already screened on. And the path is greedy: one move is taken at a time, so a
    pair of predictors worth keeping only together can be dropped one at a time
    and never come back. ``direction="both"`` reconsiders a dropped term at every
    later step, which is what it is for; nothing short of fitting every subset
    removes the problem entirely.

    :func:`~statassist.fit_elastic_net` answers the same question without a path
    at all, by shrinking a coefficient to exactly zero, and
    :func:`~statassist.perform_rfe` answers it with a resampled score rather than
    with a penalty.

    **Where the search starts and which way it moves.** ``"backward"`` starts at
    every candidate and only drops. ``"forward"`` starts at the intercept and only
    adds. ``"both"`` starts at every candidate and may add a term back after
    dropping it, so its path can visit the same size twice; that is the one
    direction whose ``profile`` is not a ladder.

    Every direction is bounded by the same two models: the intercept alone below
    and all of ``predictors`` above. A search that walks back to the intercept has
    kept nothing, which is an answer - no candidate pays for itself at this charge
    - but not one this contract can carry, since ``selected`` would be empty. It
    is an error saying so rather than a result with a hole in it.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The outcome, either the name of a column of ``data`` or a vector
            with one entry per row.
        predictors: Candidate column names, or ``None`` for every column of
            ``data`` except the outcome. A factor is one candidate however many
            dummy columns it becomes.
        outcome_lv: For a two-class outcome, the two classes with the reference
            first. Naming it is also what tells this function that a numeric
            column of zeroes and ones is two classes rather than two numbers.
        control_label: The reference class on its own, for when the other one
            needs no saying. Naming both and disagreeing is an error.
        model: What is fitted at every step: ``"linear"`` or ``"logistic"``.
        criterion: Which criterion the moves are judged by, ``"AIC"`` or
            ``"BIC"``. The search is the same one either way; what changes is the
            charge per parameter, 2 against ``log(n)``.
        direction: ``"backward"``, ``"both"`` or ``"forward"``.

    Returns:
        A :class:`~statassist.core.SaSelection` whose ``analysis`` is
        ``"stepwise"``.

        * ``candidates`` - the predictors that were offered, most important
          first, which is the row order ``ranking`` follows.
        * ``selected`` - the predictors of the model the search stopped at, most
          important first.
        * ``ranking`` - ``candidates``, ``estimate``, ``rank`` and ``selected``.
          The estimate is what leaving that one predictor out of the selected
          model costs the criterion, so it is positive for a predictor worth
          keeping and negative for one worth leaving out.
        * ``profile`` - one row per step of the path: ``n_vars``, both ``AIC`` and
          ``BIC`` of the model at that step, the ``step`` that was taken to reach
          it, and ``chosen``, which is ``True`` on the last row.
        * ``resampling`` - ``None``. Nothing was resampled.

    Raises:
        SaValueError: If ``model`` and the outcome disagree, if the search walked
            back to the intercept, or if an argument is unusable.

    Examples:
        Six candidates and one continuous outcome. The path says what was dropped
        and when, and ``selected`` is where the search stopped.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=80, n_pred=6, n_factor_pred=0,
        ...                           p_missing=0, seed=7)
        >>> res = perform_stepwise(**sim.args)
        >>> res["analysis"]
        'stepwise'
        >>> list(res["profile"])
        ['n_vars', 'AIC', 'BIC', 'step', 'chosen']
        >>> res["resampling"] is None
        True

        The first row of the path is the model the search started at, which no
        move reached, and the last is the one it kept.

        >>> res["profile"]["step"].iloc[0]
        ''
        >>> bool(res["profile"]["chosen"].iloc[-1])
        True

        A heavier charge per parameter keeps fewer predictors, and a search that
        drops every candidate is refused rather than answered with an empty set.

        >>> heavy = perform_stepwise(**sim.args, criterion="BIC")
        >>> len(heavy["selected"]) < len(res["selected"])
        True
        >>> set(heavy["selected"]) <= set(res["selected"])
        True
    """
    model = check_choice(model, STEPWISE_MODELS, "model")
    criterion = check_choice(criterion, CRITERIA, "criterion")
    direction = check_choice(direction, DIRECTIONS, "direction")

    input_ = resolve_model_input(data, outcome, predictors)
    resolved = resolve_search_outcome(input_, model, outcome_lv, control_label, _WORDS)

    charge = math.log(input_.n_used) if criterion == CRITERIA[1] else AIC_CHARGE
    label = search_label(model, resolved.classify)

    # One grouped note for the whole procedure rather than one per stage. A
    # condition of the data - a logistic regression whose predictors separate its
    # classes perfectly, say - is raised by every model the path fits and again by
    # the ones refitted afterwards to price each candidate, and it is the same
    # condition every time.
    with quiet_engine(label):
        path = _walk(input_, resolved, model, direction, charge, label)
        kept = list(path[-1].terms)
        priced = _price(input_, resolved, model, kept, charge, label)

    if not kept:
        raise SaValueError(
            f"the search walked back to the intercept: at a charge of {fmt_est(charge)} "
            f"per parameter, none of the {len(input_.predictors)} candidate(s) lowers "
            f"{criterion} by more than it costs. That is an answer rather than a "
            "failure, but not one this function can return, since `selected` would be "
            "empty. "
            + (
                '`criterion = "AIC"` charges 2 per parameter instead. '
                if criterion == "BIC"
                else ""
            )
            + "Read it as none of these predictors being worth a coefficient on these rows."
        )

    ranking = ranking_table(priced, input_.predictors, kept)
    return new_selection(
        analysis="stepwise",
        # Most important first, which is the ranking's order rather than the
        # order the model happens to hold its terms in.
        candidates=[str(name) for name in ranking["candidates"]],
        design=search_design(input_, resolved),
        # `maximize` is recorded and not asked for: a criterion is a cost, so a
        # caller who could say otherwise could ask for the worst model by
        # accident. This is the rule `rfe_metric()` follows for a resampled
        # metric.
        parameters={
            "model": model,
            "criterion": criterion,
            "maximize": False,
            "k": charge,
            "direction": direction,
        },
        selected=[
            str(name) for name in ranking.loc[ranking["selected"].astype(bool), "candidates"]
        ],
        ranking=ranking,
        profile=_profile(path),
        resampling=None,
        engine={
            "package": "scikit-learn",
            "method": "stepwise",
            "label": label,
            "metrics": list(CRITERIA),
            "importance": f"{criterion} increase when the predictor is left out",
            "overridden": list(_OVERRIDDEN),
        },
        fit=path,
    )


@dataclass(frozen=True)
class Step:
    """One model the path visited, and the move that reached it.

    Both criteria are recorded whichever one is searching, since they are two
    charges against the same likelihood and the fit that has one has the other.

    Attributes:
        terms: The candidates in the model, in the order they arrived.
        move: What was done to reach it, ``"- wt"`` or ``"+ wt"``, empty on the
            model the search started at.
        aic: :func:`stats::AIC` of the model.
        bic: :func:`stats::BIC` of the model.
        criterion: The same likelihood at the charge the search is moving by.
    """

    terms: tuple[str, ...]
    move: str
    aic: float
    bic: float
    criterion: float


def _walk(
    input_: ModelInput,
    resolved: SearchOutcome,
    model: str,
    direction: str,
    charge: float,
    label: str,
) -> list[Step]:
    """Take one move at a time until no single move lowers the criterion.

    The path is bounded by the same two models in every direction: the intercept
    alone below and all of the candidates above. ``"forward"`` starts at the
    lower bound and only adds, the other two start at the upper bound, and
    ``"both"`` is the one that may put a dropped term back.
    """
    candidates = [str(name) for name in input_.predictors]
    drops = direction in ("backward", "both")
    adds = direction in ("forward", "both")

    current = () if direction == "forward" else tuple(candidates)
    path = [_step(input_, resolved, model, current, "", charge, label)]

    while True:
        moves: list[tuple[str, tuple[str, ...]]] = []
        if drops:
            moves += [
                (f"- {name}", tuple(term for term in current if term != name)) for name in current
            ]
        if adds:
            outside = [name for name in candidates if name not in set(current)]
            moves += [
                (f"+ {name}", tuple(term for term in candidates if term in {*current, name}))
                for name in outside
            ]
        if not moves:
            break

        tried = [
            _step(input_, resolved, model, terms, move, charge, label) for move, terms in moves
        ]
        best = min(tried, key=lambda step: step.criterion)
        if not best.criterion < path[-1].criterion - STEP_TOL:
            break
        path.append(best)
        current = best.terms
    return path


def _step(
    input_: ModelInput,
    resolved: SearchOutcome,
    model: str,
    terms: Sequence[str],
    move: str,
    charge: float,
    label: str,
) -> Step:
    """Fit one model of the path and record what it costs."""
    log_lik, n_params = _likelihood(input_, resolved, model, terms, label)
    n = input_.n_used
    return Step(
        terms=tuple(str(name) for name in terms),
        move=move,
        aic=-2 * log_lik + AIC_CHARGE * n_params,
        bic=-2 * log_lik + math.log(n) * n_params,
        criterion=-2 * log_lik + charge * n_params,
    )


def _likelihood(
    input_: ModelInput,
    resolved: SearchOutcome,
    model: str,
    terms: Sequence[str],
    label: str,
) -> tuple[float, int]:
    """The log likelihood of one subset at its maximum, and its parameter count.

    The two together are what every criterion is: the likelihood with a charge
    levied per parameter. A linear model's parameters are its coefficients plus
    the residual spread, which is what :func:`stats::logLik` counts and why the
    count is a term higher than the rank; a logistic model estimates no spread,
    so its count is the rank.

    A subset the data fits exactly has no finite likelihood, and the infinity is
    passed on rather than smoothed over: such a model wins every comparison, which
    is the right answer to a question that should not have been asked.
    """
    frame = input_.x[[str(name) for name in terms]]
    matrix = design_matrix(frame, _levels(frame))
    fitted = (
        logistic_fit(matrix, resolved.y, label)
        if resolved.classify
        else least_squares(matrix, resolved.y, label)
    )

    if resolved.classify:
        probability = 1 / (1 + np.exp(-fitted.fitted))
        return -logistic_scores(resolved.y, probability)["residual_deviance"] / 2, fitted.rank

    n = len(resolved.y)
    rss = float(np.sum((resolved.y - fitted.fitted) ** 2))
    if rss <= 0:
        return math.inf, fitted.rank + 1
    log_lik = -n / 2 * (math.log(2 * math.pi) + math.log(rss / n) + 1)
    return log_lik, fitted.rank + 1


def _levels(x: pd.DataFrame) -> dict[str, list[str]]:
    """The levels of the factors among these columns, as ``design_matrix`` takes them.

    Passed explicitly so that every model on the path codes a factor against the
    same levels, which is what makes the criteria of two steps comparable.
    """
    return {
        str(name): [str(level) for level in x[name].cat.categories]
        for name in x.columns
        if isinstance(x[name].dtype, pd.CategoricalDtype)
    }


def _price(
    input_: ModelInput,
    resolved: SearchOutcome,
    model: str,
    kept: Sequence[str],
    charge: float,
    label: str,
) -> dict[str, float]:
    """What each candidate is worth to the model the search stopped at.

    One number for both groups of candidates, which is what makes the ranking
    readable as a single column: what the criterion would be with this predictor
    left out of the selected model, minus what it is with it in. A predictor the
    search kept is worth the rise that dropping it would cause, so it is positive;
    a predictor the search left out would raise the criterion by being added, so
    its number is the same difference and comes out negative. The sign is the
    search's own decision about it and the size is by how much, which puts the
    selection at the top of the table with no separate sort.
    """
    inside = list(kept)
    at = _step(input_, resolved, model, inside, "", charge, label).criterion

    priced: dict[str, float] = {}
    for name in inside:
        without = [term for term in inside if term != name]
        priced[name] = _step(input_, resolved, model, without, "", charge, label).criterion - at
    for name in (str(value) for value in input_.predictors):
        if name in priced:
            continue
        with_it = [term for term in input_.predictors if term in {*inside, name}]
        priced[name] = at - _step(input_, resolved, model, with_it, "", charge, label).criterion
    return priced


def _profile(path: Sequence[Step]) -> pd.DataFrame:
    """One row per step of the path.

    ``n_vars`` counts predictors rather than coefficients, so a factor counts
    once, and it is the field name the contract shares with
    :func:`~statassist.perform_rfe`, where the row axis is a subset size instead
    of a step. ``chosen`` is the last row: the search stops when no move improves
    the criterion, so where it stopped is what it chose.
    """
    table = pd.DataFrame(
        {
            "n_vars": [len(step.terms) for step in path],
            "AIC": [step.aic for step in path],
            "BIC": [step.bic for step in path],
            "step": [step.move for step in path],
            "chosen": [position == len(path) - 1 for position in range(len(path))],
        }
    )
    return table[stepwise_profile_columns()]
