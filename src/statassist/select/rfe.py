"""Recursive feature elimination: rank, drop the weakest, score, repeat.

The port of ``R/perform_rfe.R``. R hands the whole procedure to
``caret::rfe()``; here the elimination is written out for the reason
:func:`~statassist.fit._shared.resample_grid` is written out - ``scikit-learn``
owns the splitters and not the summary - and for one more that is specific to
this function. ``sklearn.feature_selection.RFECV``, which is the routine this
would otherwise be, eliminates *columns* of the matrix it is handed and re-ranks
after every drop. Neither is the procedure R runs: a factor is one candidate
however many dummy columns it becomes, and the ranking is computed once per
resample at the full set. Those two disagreements are what the walk below exists
to settle, and both are declared in ``engine["overridden"]``.

What is kept from R is what makes a ranking worth standing behind.

``caret``'s ``lmFuncs$rank`` ranks by ``abs(coef(object))``. A coefficient is an
effect per unit of its predictor, so ranking by its size ranks by the units the
predictors happen to be measured in: the same model with a predictor in
millimetres rather than metres eliminates in a different order. What replaces it
is the absolute t statistic, which is the coefficient divided by its own standard
error and so has no units left, and which is what ``caret::lrFuncs`` already
ranks a logistic regression by. The two models therefore rank on the same scale.

Neither fit sees the columns that were passed in. A factor with ``k`` levels is
``k - 1`` coefficients, so a ranking read off the coefficients is a ranking of
dummy columns and cannot be matched back to the candidates. The ranking here
folds the dummies back into the column they came from and keeps the largest
statistic among them, so what is eliminated is always a column of the input. That
is what makes ``selected`` something that can be handed straight back to
``fit_rf(predictors=...)``, which is the whole point of running a selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaInternalError, SaValueError
from ..core.result import SaSelection, new_selection
from ..core.validate import check_count, check_num_vector
from ..fit._shared import (
    CLASSIFICATION_METRICS,
    CV_METHODS,
    INTERCEPT,
    REGRESSION_METRICS,
    ModelInput,
    ResampleControl,
    check_cv_method,
    classification_scores,
    design_matrix,
    design_source,
    least_squares,
    logistic_fit,
    model_frame,
    permutation_scores,
    quiet_engine,
    regression_scores,
    resample_labels,
    resample_mean,
    resample_spread,
    resolve_model_input,
    rf_grid,
    rollup,
    search_label,
    train_control,
)
from ._shared import (
    SearchOutcome,
    SearchWords,
    check_choice,
    ranking_table,
    resolve_search_outcome,
    search_design,
)

__all__ = ["perform_rfe"]

#: What can do the ranking inside the search, in the order R lists them.
RFE_MODELS = ("linear", "logistic", "rf")

#: The metrics that are better when they are smaller.
#:
#: ``maximize`` follows from the metric and is not an argument. An error is better
#: when it is smaller and a rate is better when it is larger, and letting a caller
#: say otherwise would let them ask for the worst subset by accident.
MINIMIZED_METRICS = ("RMSE", "MAE")

#: The subset sizes scored when the caller names none.
#:
#: Dense at the small end because that is where the answer usually is: the
#: difference between three predictors and four is a different model, while the
#: difference between sixty and seventy is the same model with noise in it.
DEFAULT_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100)

#: Rows a leaf may hold when the caller names none, by outcome kind.
#:
#: ``randomForest``'s own defaults, the ones :func:`~statassist.fit_rf` takes.
_NODESIZE = {True: 1, False: 5}

#: Shuffles per term when a forest does the ranking.
#:
#: The same count :func:`~statassist.fit_rf` uses, so that a forest's ranking here
#: and its importance table there are the same measurement.
_N_PERMUTE = 10

#: What ``ranking["estimate"]`` measures, by ranking model.
_IMPORTANCE = {
    "linear": "absolute t statistic",
    "logistic": "absolute Wald z",
    "rf": "permutation importance",
}

#: What this port changed about the procedure, for ``engine["overridden"]``.
_OVERRIDDEN = (
    "elimination on the candidate axis, not the design matrix columns",
    "ranking computed once per resample, at the full candidate set",
    "permutation importance on the fitted rows",
)

#: The sentences that name this search where a message has to name one.
_WORDS = SearchWords(
    procedure="a recursive feature elimination",
    linear_refusal=(
        '`model = "linear"` ranks by the coefficients of a straight line through a '
        'number, and `outcome` is a set of class labels. Use `model = "logistic"` or '
        '`model = "rf"` for a two-class outcome.'
    ),
    logistic_refusal='Use `model = "linear"` or `model = "rf"` for a continuous outcome.',
    numeric_note='`model = "logistic"` or `model = "rf"`',
    non_finite="which a model fitted inside the search cannot be scored against.",
)


def perform_rfe(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    model: str = RFE_MODELS[0],
    subset_sizes: Any = None,
    metric: Any = None,
    ntree: Any = 500,
    nodesize: Any = None,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    seed: int | None = None,
) -> SaSelection:
    """Select the predictors worth keeping.

    Runs a recursive feature elimination: the predictors are ranked, the weakest
    is dropped, and the model is scored again, over and over, so that every
    subset size gets a resampled score. What comes back is the size that scored
    best, the predictors of that size, and the whole profile the search walked, so
    that a choice of two predictors over eight can be read against what the other
    six were worth.

    The input is the wide format the model functions take, one row per observation
    with one column as the outcome, and is normally the training half of a
    :func:`~statassist.split_data` result.

    **What is resampled.** The elimination is inside the resampling, not before
    it. Each fold ranks the predictors on its own training rows, peels them down
    to each size in turn, and scores every size on rows it did not rank on, so a
    predictor that looks useful only on the rows that chose it is caught. Ranking
    once on all the rows and cross-validating afterwards is the mistake this
    ordering exists to avoid, and it is why there is no ``cv`` argument: an
    elimination with nothing held out has no score to choose a size by, so it is
    not a shorter version of this function but a different and wrong one.

    What the resampling cannot do is make the reported score an honest estimate of
    the selected model. The size was chosen because it scored best, so ``profile``
    reads high at the size it picked, for the same reason a maximum of noisy
    numbers is above their mean. Score the selection on the test half of
    :func:`~statassist.split_data`, which the search never saw, and read
    ``profile`` as the shape of the search rather than as a performance claim.

    The ranking is computed once per resample, at the full set of predictors, and
    the peeling follows it. Re-ranking after every drop is one refit per remaining
    predictor per fold, a far slower procedure, so it is not what this function
    does.

    **Which model does the ranking.** ``model`` names what is fitted inside the
    search, and the outcome has to agree with it. ``"linear"`` is a continuous
    outcome, ``"logistic"`` a two-class one, and ``"rf"`` is either, following the
    outcome the way :func:`~statassist.fit_rf` does. A disagreement is an error
    naming the model that would have fitted rather than a silently different
    analysis.

    What each ranks by is reported in ``engine["importance"]``. The two
    regressions rank by the absolute t or Wald z statistic of each coefficient,
    with a factor ranked as one column by the largest statistic among its levels
    and a term the fit could not estimate ranking at zero, since a predictor the
    others already span costs nothing to drop. The forest ranks by permutation
    importance, which is the measure :func:`~statassist.fit_rf` reports as
    ``estimate``, so a forest's ranking here and its importance table there can be
    read together.

    A forest inside the search grows at the rule of thumb for each subset - the
    square root of the term count for a classification and a third of it for a
    regression - rather than at one value throughout. A fixed count would exceed
    the terms available at the small end of the profile, where the whole question
    is what a handful of predictors can do.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The outcome, either the name of a column of ``data`` or a vector
            with one entry per row.
        predictors: Candidate column names, or ``None`` for every column of
            ``data`` except the outcome.
        outcome_lv: For a two-class outcome, the two classes with the reference
            first. Naming it is also what tells this function that a numeric
            column of zeroes and ones is two classes rather than two numbers.
        control_label: The reference class on its own, for when the other one
            needs no saying. Naming both and disagreeing is an error.
        model: What is fitted inside the search: ``"linear"``, ``"logistic"`` or
            ``"rf"``.
        subset_sizes: Subset sizes to score, or ``None`` for a ladder that is
            dense where the answer usually is - every size up to ten, then 15,
            20, 30, 50 and 100 - capped at the number of candidates. The full set
            is always scored, since keeping everything is the option a selection
            is being compared against.
        metric: Which resampled number chooses the size: ``"RMSE"``,
            ``"Rsquared"`` or ``"MAE"`` for a regression, ``"Accuracy"`` or
            ``"Kappa"`` for a classification, or ``None`` for the first of each.
            Whether it is maximised follows from the metric and is reported in
            ``parameters``.
        ntree: Trees to grow, used by ``model="rf"`` and ignored otherwise.
        nodesize: Rows a leaf may hold, used by ``model="rf"``. ``None`` leaves it
            at 1 for a classification and 5 for a regression.
        cv_method: Resampling scheme: ``"repeated_kfold"``, ``"kfold"`` or
            ``"loocv"``.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        seed: Seed for the fold assignment, and for a forest's own randomness.

    Returns:
        A :class:`~statassist.core.SaSelection` whose ``analysis`` is ``"rfe"``.

        * ``candidates`` - the predictors that were offered, most important
          first, which is the row order ``ranking`` follows.
        * ``selected`` - the predictors of the winning size, most important
          first. These are the names to hand to ``predictors=`` in a ``fit_*``
          call.
        * ``ranking`` - ``candidates``, the ``estimate`` it was ranked by averaged
          over the resamples, its ``rank``, and whether it was ``selected``.
        * ``profile`` - one row per subset size that was scored: ``n_vars``, one
          column per metric with its standard deviation over the resamples, and
          ``chosen``, which is ``True`` on exactly one row.
        * ``resampling`` - one row per resample at the chosen size, absent under
          leave-one-out, which is scored on the pooled predictions.

    Raises:
        SaValueError: If ``model`` and the outcome disagree, if ``metric`` belongs
            to the other kind of outcome, or if an argument is unusable.

    Examples:
        Four candidates and a continuous outcome, scored over three folds.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=90, n_pred=4, n_factor_pred=0,
        ...                           p_missing=0, seed=3)
        >>> res = perform_rfe(**sim.args, cv_method="kfold", n_fold=3, seed=1)
        >>> res["analysis"]
        'rfe'
        >>> list(res["ranking"])
        ['candidates', 'estimate', 'rank', 'selected']
        >>> set(res["selected"]) <= set(res["candidates"])
        True

        The size marked in the profile is the size that was kept.

        >>> chosen = res["profile"].loc[res["profile"]["chosen"], "n_vars"]
        >>> int(chosen.iloc[0]) == len(res["selected"])
        True

        The predictor the simulator planted the largest coefficient on survives.

        >>> planted = sim.truth.loc[sim.truth["beta"].abs().idxmax(), "predictors"]
        >>> planted in res["selected"]
        True
    """
    model = check_choice(model, RFE_MODELS, "model")
    cv_method = check_cv_method(cv_method)
    trees = check_count(ntree, "ntree", 1)

    input_ = resolve_model_input(data, outcome, predictors)
    resolved = resolve_search_outcome(input_, model, outcome_lv, control_label, _WORDS)

    # Validated whether or not the chosen model reads it, for the reason
    # `train_control()` validates the folds of a scheme that has none.
    leaf = check_count(
        _NODESIZE[resolved.classify] if nodesize is None else nodesize, "nodesize", 1
    )

    sizes = rfe_sizes(subset_sizes, len(input_.predictors))
    chosen_metric, maximize = rfe_metric(metric, resolved.classify)
    label = search_label(model, resolved.classify)

    # There is no `cv` argument to pass on: an elimination with nothing held out
    # has no score to choose a size by, so the scheme is always a resampling one.
    control = train_control(
        True, cv_method, n_fold, n_repeat, input_.n_used, classify=resolved.classify, seed=seed
    )
    walked = _walk(input_, resolved, model, trees, leaf, sizes, control, label, seed)

    metrics = list(CLASSIFICATION_METRICS if resolved.classify else REGRESSION_METRICS)
    pooled = control.n_fold is None
    profile = _profile(walked.scored, sizes, metrics, pooled)
    best = _best_size(profile, chosen_metric, maximize)
    profile["chosen"] = profile["n_vars"] == best

    averaged = {
        name: resample_mean(np.array(values, dtype=float)) for name, values in walked.ranked.items()
    }
    ranking = ranking_table(averaged, input_.predictors, [])
    # The winning subset is taken off the averaged ranking rather than off any one
    # resample's, so that the top of `ranking` and `selected` are the same
    # predictors in the same order rather than two answers that usually agree.
    selected = [str(name) for name in ranking["candidates"].iloc[:best]]
    ranking["selected"] = [name in set(selected) for name in ranking["candidates"]]

    resampling = None
    if not pooled:
        resampling = pd.DataFrame(walked.scored[best])
        resampling["Resample"] = resample_labels(control, len(resampling.index))

    return new_selection(
        analysis="rfe",
        candidates=[str(name) for name in ranking["candidates"]],
        design=search_design(input_, resolved),
        # No `subset_sizes` here: `profile` holds one row per size that was
        # scored, so recording the ladder as well would say the same thing twice
        # and leave two places for it to be wrong.
        parameters={
            "model": model,
            "metric": chosen_metric,
            "maximize": maximize,
            **({"ntree": trees, "nodesize": leaf} if model == "rf" else {}),
            "cv_method": control.cv_method,
            "n_fold": control.n_fold,
            "n_repeat": control.n_repeat,
            "seed": seed,
        },
        selected=selected,
        ranking=ranking,
        profile=profile,
        resampling=model_frame(resampling),
        engine={
            "package": "scikit-learn",
            "method": "rfe",
            "label": label,
            "metrics": metrics,
            "importance": _IMPORTANCE[model],
            "overridden": list(_OVERRIDDEN),
        },
        fit=walked,
    )


def rfe_sizes(subset_sizes: Any, p: int) -> list[int]:
    """The subset sizes to score, and the ladder that is scored by default.

    Port of ``sa_rfe_sizes()``, with the full set added the way ``caret`` adds it:
    whichever sizes are asked for, keeping everything is what a selection is being
    compared against, so it is always among them.
    """
    if subset_sizes is None:
        return sorted({min(p, size) for size in DEFAULT_SIZES})

    values = check_num_vector(subset_sizes, "subset_sizes", 1, p)
    fractional = np.unique(values[values != np.trunc(values)])
    if fractional.size > 0:
        raise SaValueError(
            "`subset_sizes` counts predictors, so it must hold whole numbers, but holds "
            + ", ".join(str(value) for value in fractional)
            + "."
        )
    return sorted({int(value) for value in values} | {p})


def rfe_metric(metric: Any, classify: bool) -> tuple[str, bool]:
    """The metric the size is chosen by, and which way it is read.

    Port of ``sa_rfe_metric()``. A metric belongs to a kind of outcome: there is
    no accuracy of a continuous prediction and no root mean squared error of a
    class label. Naming one from the other list is refused here rather than after
    the folds have been scored and the column turns out not to be among them.
    """
    available = list(CLASSIFICATION_METRICS if classify else REGRESSION_METRICS)
    if metric is None:
        chosen = available[0]
    elif not isinstance(metric, str) or metric not in available:
        raise SaValueError(
            "`metric` must be one of "
            + ", ".join(available)
            + " for a "
            + ("classification" if classify else "regression")
            + "."
        )
    else:
        chosen = metric
    return chosen, chosen not in MINIMIZED_METRICS


@dataclass
class Elimination:
    """The walk over the resamples, as the handle the result keeps.

    This is what ``res.fit`` holds, and it stands where R keeps the ``rfe``
    object. It carries the two things the tables were built from rather than a
    fitted engine, because there is no single engine here: one fit per size per
    resample, none of which outlives the fold it was scored in.

    Attributes:
        ranked: One importance per resample, per candidate.
        scored: One metric mapping per resample, per subset size.
    """

    ranked: dict[str, list[float]]
    scored: dict[int, list[dict[str, float]]]

    def __repr__(self) -> str:
        return f"<Elimination> {len(self.ranked)} candidate(s) over {len(self.scored)} size(s)"


def _walk(
    input_: ModelInput,
    resolved: SearchOutcome,
    model: str,
    ntree: int,
    nodesize: int,
    sizes: Sequence[int],
    control: ResampleControl,
    label: str,
    seed: int | None,
) -> Elimination:
    """Rank inside every resample, peel to each size, score on the held-out rows.

    The order is the whole point. A fold ranks on its own training rows and is
    scored on rows it did not rank on, so a predictor that looks useful only on
    the rows that chose it is caught here rather than flattered.

    Leave-one-out is scored on the pooled held-out predictions rather than per
    fold, the way :func:`~statassist.fit._shared.resample_grid` scores it and for
    the same reason: a fold of one row has no correlation and no spread of its
    own.

    One note for the whole walk rather than one per fit, since the same condition
    of the data is raised by every fold and by every size within it.
    """
    if control.splitter is None:
        raise SaInternalError(
            "internal error: the elimination was handed a scheme that holds nothing out, "
            "so no subset size has a score to be chosen by."
        )
    splits = list(control.splitter.split(input_.x, resolved.y if resolved.classify else None))
    ranked: dict[str, list[float]] = {str(name): [] for name in input_.predictors}
    held: dict[int, list[np.ndarray]] = {size: [] for size in sizes}
    guessed: dict[int, list[np.ndarray]] = {size: [] for size in sizes}

    with quiet_engine(label):
        for train, test in splits:
            x_train = input_.x.iloc[train].reset_index(drop=True)
            x_test = input_.x.iloc[test].reset_index(drop=True)
            y_train = resolved.y[train]

            importance = _rank(
                x_train, y_train, input_.predictors, model, resolved, ntree, nodesize, label, seed
            )
            for name, value in importance.items():
                ranked[name].append(value)

            order = _peel(importance, input_.predictors)
            for size in sizes:
                keep = order[:size]
                held[size].append(resolved.y[test])
                guessed[size].append(
                    _predict(
                        x_train[keep],
                        y_train,
                        x_test[keep],
                        model,
                        resolved,
                        ntree,
                        nodesize,
                        label,
                        seed,
                    )
                )

    score = classification_scores if resolved.classify else regression_scores
    pooled = control.n_fold is None
    scored: dict[int, list[dict[str, float]]] = {}
    for size in sizes:
        if pooled:
            scored[size] = [score(np.concatenate(held[size]), np.concatenate(guessed[size]))]
        else:
            scored[size] = [
                score(truth, guess) for truth, guess in zip(held[size], guessed[size], strict=True)
            ]
    return Elimination(ranked, scored)


def _levels(x: pd.DataFrame) -> dict[str, list[str]]:
    """The levels of the factors among these columns, as ``design_matrix`` takes them.

    Passed explicitly rather than left to be read off each frame, so that a level
    no fold happens to hold still gets its column of zeroes: the matrix a subset
    predicts on has to have the columns the matrix it was fitted on had.
    """
    return {
        str(name): [str(level) for level in x[name].cat.categories]
        for name in x.columns
        if isinstance(x[name].dtype, pd.CategoricalDtype)
    }


def _rank(
    x: pd.DataFrame,
    y: np.ndarray,
    predictors: Sequence[str],
    model: str,
    resolved: SearchOutcome,
    ntree: int,
    nodesize: int,
    label: str,
    seed: int | None,
) -> dict[str, float]:
    """What each candidate is worth on these rows, on the candidate axis."""
    levels = _levels(x)
    matrix = design_matrix(x, levels)
    source = design_source(x, levels)
    names = [str(name) for name in predictors]

    if model == "rf":
        forest = _forest(matrix, y, resolved.classify, ntree, nodesize, seed)
        permuted = permutation_scores(forest, matrix, y, resolved.classify, _N_PERMUTE, seed)
        totals = rollup(
            dict(zip(matrix.columns, permuted, strict=True)),
            source,
            names,
        )
        return dict(zip(names, (float(value) for value in totals), strict=True))

    fitted = (
        logistic_fit(matrix, y, label) if resolved.classify else least_squares(matrix, y, label)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.abs(fitted.estimate / fitted.stderr)
    # A term the fit could not estimate ranks at zero rather than at missing: it is
    # a column the others already span, so dropping it costs nothing, which is
    # exactly what a rank of zero says.
    statistic = np.nan_to_num(statistic, nan=0.0, posinf=0.0, neginf=0.0)

    # A factor is several coefficients and only the column can be eliminated, so
    # the largest statistic among its levels stands for it: a factor is kept as
    # long as one of its levels is worth keeping.
    largest = dict.fromkeys(names, 0.0)
    for term, value in zip(fitted.terms, statistic, strict=True):
        if term == INTERCEPT:
            continue
        name = source[term]
        largest[name] = max(largest[name], float(value))
    return largest


def _peel(importance: Mapping[str, float], predictors: Sequence[str]) -> list[str]:
    """The candidates in the order the elimination drops them, weakest last.

    Ties keep the order the candidates arrived in, which is what makes the
    elimination of a set of equally worthless columns reproducible.
    """
    names = [str(name) for name in predictors]
    values = np.array([importance[name] for name in names], dtype=float)
    key = np.where(np.isnan(values), math.inf, -values)
    return [names[position] for position in np.argsort(key, kind="stable")]


def _predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    model: str,
    resolved: SearchOutcome,
    ntree: int,
    nodesize: int,
    label: str,
    seed: int | None,
) -> np.ndarray:
    """Fit one subset on the training rows and predict the held-out ones.

    Both matrices are coded against the levels of the training rows and the test
    one is asked for the training columns by name, since a design matrix is read
    by position once it reaches the engine.
    """
    levels = _levels(x_train)
    train = design_matrix(x_train, levels)
    test = design_matrix(x_test, levels, want=list(train.columns))

    if model == "rf":
        forest = _forest(train, y_train, resolved.classify, ntree, nodesize, seed)
        return np.asarray(forest.predict(np.asarray(test, dtype=float)))

    fitted = (
        logistic_fit(train, y_train, label)
        if resolved.classify
        else least_squares(train, y_train, label)
    )
    return np.asarray(fitted.estimator.predict(test))


def _forest(
    matrix: pd.DataFrame,
    y: np.ndarray,
    classify: bool,
    ntree: int,
    nodesize: int,
    seed: int | None,
) -> Any:
    """A forest fitted at the rule-of-thumb ``mtry`` for this many terms.

    Grown at the rule rather than at one value throughout, because a fixed count
    would exceed the terms available at the small end of the profile, where the
    whole question is what a handful of predictors can do.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    grid = rf_grid(None, len(matrix.columns), classify, cv=False)
    maker = RandomForestClassifier if classify else RandomForestRegressor
    forest = maker(
        n_estimators=ntree,
        max_features=int(grid["mtry"].iloc[0]),
        min_samples_leaf=nodesize,
        random_state=seed,
    )
    forest.fit(np.asarray(matrix, dtype=float), y)
    return forest


def _profile(
    scored: Mapping[int, list[dict[str, float]]],
    sizes: Sequence[int],
    metrics: Sequence[str],
    pooled: bool,
) -> pd.DataFrame:
    """One row per subset size, with every metric averaged over the resamples.

    ``n_vars`` counts predictors rather than coefficients, so a factor counts
    once, and it is the field name the contract shares with
    :func:`~statassist.perform_stepwise`, where the row axis is a step of a path
    instead of a subset size.
    """
    rows: list[dict[str, Any]] = []
    for size in sizes:
        folded = scored[size]
        row: dict[str, Any] = {"n_vars": int(size)}
        for metric in metrics:
            values = np.array([fold[metric] for fold in folded], dtype=float)
            row[metric] = resample_mean(values)
            if not pooled:
                row[f"{metric}SD"] = resample_spread(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _best_size(profile: pd.DataFrame, metric: str, maximize: bool) -> int:
    """The size that placed first on the metric the caller chose.

    A profile whose metric no size could answer keeps the smallest size, which is
    the first row: with nothing to choose by there is no reason to keep more
    columns than the fewest that were scored.
    """
    values = profile[metric].to_numpy(dtype=float)
    if np.isnan(values).all():
        at = 0
    else:
        at = int(np.nanargmax(values) if maximize else np.nanargmin(values))
    return int(profile["n_vars"].iloc[at])
