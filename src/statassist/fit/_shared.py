"""Internal helpers shared by the model fitting functions.

The port of ``R/utils_model.R``. A model function's own body should be the part
that is specific to that model: which engine runs and what its summary means.
Everything before that is the same question every time - which columns are the
predictors, which rows can be used, and how the resampling scheme was described -
so it is answered once here.

Two things are done differently from R, and both follow from the engine rather
than from a change of mind.

R hands ``caret`` the predictor frame and lets it build the model frame, so only
the engines that take a matrix - ``glmnet`` and ``kernlab`` - need
``sa_design_matrix()``. Every ``scikit-learn`` estimator takes a matrix, so every
model here is fitted from one and ``engine["x_names"]`` is always present. The
coding is the same coding: a ``k``-level factor becomes the same ``k - 1`` terms
under the same names R's :func:`stats::model.matrix` gives them, which is what
lets a coefficient table be read beside the R one and beside a simulator's
``truth_term``.

R gets its resampled numbers from ``caret``, which owns the grid search, the fold
scoring and the summary table. ``scikit-learn`` splits the two: the splitters are
its own and the summary is not, so :func:`resample_grid` is what stands where
``train()`` stood. The tables it produces keep ``caret``'s column names, so
``performance`` and ``resampling`` mean the same thing on both sides.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.contracts import model_coef_columns, model_inference_columns
from ..core.errors import SaInternalError, SaValueError, notify
from ..core.validate import (
    check_count,
    check_flag,
    check_num_vector,
    resolve_row_vector,
)

__all__ = [
    "CLASSIFICATION_METRICS",
    "CV_METHODS",
    "INTERCEPT",
    "N_CLASSES",
    "PENALTIES",
    "PREDICT_TYPES",
    "REGRESSION_METRICS",
    "SEARCH_OUTCOME",
    "DesignEstimator",
    "EngineFit",
    "LinearFit",
    "ModelInput",
    "Outcome",
    "ResampleControl",
    "Resampled",
    "ScaledFit",
    "check_cv_method",
    "check_penalty",
    "class_scores",
    "classification_scores",
    "design_lv",
    "design_matrix",
    "design_source",
    "encode_outcome",
    "enet_grid",
    "importance_table",
    "inference_table",
    "least_squares",
    "logistic_estimator",
    "logistic_fit",
    "logistic_scores",
    "model_design",
    "model_frame",
    "no_grid",
    "numeric_scores",
    "outcome_levels",
    "permutation_scores",
    "predict_frame",
    "predict_model",
    "quiet_engine",
    "regression_scores",
    "resample_grid",
    "resample_labels",
    "resample_mean",
    "resample_params",
    "resample_scheme",
    "resample_spread",
    "resolve_model_input",
    "resolve_outcome",
    "rf_grid",
    "rollup",
    "scaled_estimator",
    "scaled_fit",
    "search_frame",
    "search_label",
    "svm_grid",
    "train_control",
    "wald_interval",
    "weighted_least_squares",
]

#: The resampling schemes a model function accepts, in the order R lists them.
#:
#: The first is the default everywhere, as in R, since a single k-fold run scores
#: a model on one partition and the spread across partitions is most of what the
#: number is worth.
CV_METHODS = ("repeated_kfold", "kfold", "loocv")

#: The metrics a resampled regression is scored on, under ``caret``'s names.
REGRESSION_METRICS = ("RMSE", "Rsquared", "MAE")

#: The metrics a resampled classification is scored on, under ``caret``'s names.
CLASSIFICATION_METRICS = ("Accuracy", "Kappa")

#: What :func:`predict_model` accepts, in the order R lists them.
PREDICT_TYPES = ("raw", "response", "prob")

#: Relative tolerance the rank of a design matrix is read at.
#:
#: ``lm()``'s own, ``1e-7`` on the QR decomposition. A term the other predictors
#: already span has to be told from one they nearly span, and the cutoff is what
#: decides: below it the term is aliased and its row comes back missing rather
#: than carrying an estimate the data cannot support.
RANK_TOL = 1e-7


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def check_cv_method(cv_method: Any) -> str:
    """Resolve the resampling scheme name, R's ``match.arg()``."""
    if cv_method not in CV_METHODS:
        raise SaValueError("`cv_method` must be one of: " + ", ".join(CV_METHODS) + ".")
    return str(cv_method)


@dataclass(frozen=True)
class ResampleControl:
    """The resampling scheme as it was actually used.

    Counterpart of what ``sa_train_control()`` returns. ``splitter`` is the
    ``scikit-learn`` cross-validator that stands where ``caret``'s
    ``trainControl`` object stood, or ``None`` for ``cv = False``.

    The three recorded fields are ``None`` where the scheme uses none of them,
    which is R's ``NA``: leave-one-out has no fold count and no repeats and plain
    k-fold has no repeats, so reporting ``n_fold = 5`` for a LOOCV run would be a
    plausible-looking record of something that never happened.
    """

    splitter: Any | None
    cv_method: str | None
    n_fold: int | None
    n_repeat: int | None


def resample_scheme(cv_method: str, n_fold: Any, n_repeat: Any, n_obs: int) -> ResampleControl:
    """Resolve the resampling arguments into a scheme, without the splitter.

    Port of ``sa_resample_scheme()``. Shared with the searches that resample an
    elimination rather than one fit, which build a different splitter around the
    same answer to what ``cv_method = "kfold", n_fold = 5`` means.
    """
    folds = check_count(n_fold, "n_fold", 2)
    repeats = check_count(n_repeat, "n_repeat", 1)

    if cv_method != "loocv" and folds > n_obs:
        raise SaValueError(
            f"`n_fold` = {folds} exceeds the {n_obs} usable observation(s), so a fold "
            'would be empty. Lower `n_fold` or use `cv_method = "loocv"`.'
        )

    if cv_method == "repeated_kfold":
        return ResampleControl(None, cv_method, folds, repeats)
    if cv_method == "kfold":
        return ResampleControl(None, cv_method, folds, None)
    return ResampleControl(None, cv_method, None, None)


def train_control(
    cv: Any,
    cv_method: str,
    n_fold: Any,
    n_repeat: Any,
    n_obs: int,
    classify: bool = False,
    seed: int | None = None,
) -> ResampleControl:
    """Turn the resampling arguments into a cross-validator.

    Port of ``sa_train_control()``. ``cv = False`` is the fourth scheme, the one
    with no folds and no repeats at all: a model can be fitted once and reported
    without a resampled score.

    Every argument is validated whether or not the chosen scheme reads it. A
    rejected value would otherwise depend on ``cv_method``, which is the kind of
    conditional strictness that is impossible to guess from the outside.

    Args:
        cv: Whether to resample at all.
        cv_method: Scheme name, already resolved by :func:`check_cv_method`.
        n_fold: Folds per run.
        n_repeat: Number of runs.
        n_obs: Rows available, used to reject more folds than observations.
        classify: Whether the outcome is a class, in which case the folds are
            stratified on it. That is ``caret``'s default and it is what keeps a
            fold from holding one class only.
        seed: Seed for the fold assignment, or ``None`` to draw from the
            operating system's entropy.
    """
    from sklearn.model_selection import (
        KFold,
        LeaveOneOut,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        StratifiedKFold,
    )

    resample = check_flag(cv, "cv")
    scheme = resample_scheme(cv_method, n_fold, n_repeat, n_obs)
    if not resample:
        return ResampleControl(None, None, None, None)

    if scheme.cv_method == "repeated_kfold":
        maker = RepeatedStratifiedKFold if classify else RepeatedKFold
        splitter: Any = maker(
            n_splits=scheme.n_fold,
            n_repeats=scheme.n_repeat,
            random_state=seed,
        )
    elif scheme.cv_method == "kfold":
        folder = StratifiedKFold if classify else KFold
        splitter = folder(n_splits=scheme.n_fold, shuffle=True, random_state=seed)
    else:
        splitter = LeaveOneOut()
    return ResampleControl(splitter, scheme.cv_method, scheme.n_fold, scheme.n_repeat)


def resample_labels(control: ResampleControl, n_splits: int) -> list[str]:
    """Name each resample the way ``caret`` names it.

    ``Fold1`` for a single k-fold run and ``Fold1.Rep1`` for a repeated one, so
    that a row of ``resampling`` says which partition it came from and a repeated
    scheme can be summarised per run.
    """
    if control.n_fold is None:
        return [f"Resample{position + 1:03d}" for position in range(n_splits)]
    if control.n_repeat is None:
        return [f"Fold{position + 1}" for position in range(n_splits)]
    return [
        f"Fold{position % control.n_fold + 1}.Rep{position // control.n_fold + 1}"
        for position in range(n_splits)
    ]


def regression_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score a held-out regression prediction, under ``caret``'s metric names.

    ``Rsquared`` is the squared correlation between the two vectors rather than
    the share of variance a model explains. That is ``caret``'s definition and it
    is the one that makes sense out of sample: a held-out fold has no fitted
    intercept of its own, so ``1 - RSS / TSS`` there can go below zero and is not
    the same quantity as the ``r_squared`` a fit reports.
    """
    error = observed - predicted
    scores = {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Rsquared": math.nan,
        "MAE": float(np.mean(np.abs(error))),
    }
    if observed.size > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        scores["Rsquared"] = float(np.corrcoef(observed, predicted)[0, 1] ** 2)
    return scores


def classification_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score a held-out class prediction, under ``caret``'s metric names."""
    from sklearn.metrics import cohen_kappa_score

    accuracy = float(np.mean(observed == predicted))
    # Kappa is undefined when one of the two vectors takes a single value, since
    # the agreement expected by chance is then total. `cohen_kappa_score` reports
    # 0 there, which reads as chance agreement rather than as no answer.
    if np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        kappa = math.nan
    else:
        kappa = float(cohen_kappa_score(observed, predicted))
    return {"Accuracy": accuracy, "Kappa": kappa}


@dataclass
class Resampled:
    """What the resampling had to say, in the two tables a model reports.

    Attributes:
        results: One row per hyperparameter combination, holding the tuned
            parameters and every metric with its standard deviation across
            resamples, or ``None`` when nothing was resampled.
        resampling: One row per resample at the chosen combination, or ``None``.
        best: The combination that placed first, as a mapping the result's
            ``parameters`` splices in.
        metrics: The metric names, in the order the engine scored them.
    """

    results: pd.DataFrame | None
    resampling: pd.DataFrame | None
    best: dict[str, Any]
    metrics: list[str] = field(default_factory=list)


def resample_grid(
    build: Callable[[Mapping[str, Any]], Any],
    x: pd.DataFrame,
    y: np.ndarray,
    grid: pd.DataFrame,
    control: ResampleControl,
    classify: bool,
    label: str = "the engine",
) -> Resampled:
    """Score every candidate over the resamples, and say which one placed first.

    What ``caret::train()`` does with ``tuneGrid`` and ``trControl``, written out
    because ``scikit-learn`` owns the splitters and not the summary. The tables
    keep ``caret``'s column names so that a result reads the same on both sides.

    Leave-one-out is scored on the pooled held-out predictions rather than per
    fold, and reports no ``resampling`` table. That is ``caret``'s behaviour and
    it is the only reading that works: a fold of one row has no correlation and
    no spread, so a per-fold ``Rsquared`` would be missing throughout and its
    standard deviation would describe the folds rather than the model.

    Args:
        build: Makes an unfitted estimator from one row of ``grid``.
        x: Design matrix of the usable rows.
        y: The outcome as the estimator takes it.
        grid: One row per candidate, columns named after the tuned parameters.
        control: The scheme, as :func:`train_control` resolved it.
        classify: Whether the outcome is a class.
        label: What to call the model in the note collecting the engine's own.
    """
    metrics = list(CLASSIFICATION_METRICS if classify else REGRESSION_METRICS)
    score = classification_scores if classify else regression_scores
    # The first metric is the one a candidate is chosen on, and the direction is
    # what it means rather than a setting: a smaller RMSE is better and a larger
    # accuracy is.
    maximize = classify

    if control.splitter is None:
        return Resampled(None, None, dict(grid.iloc[0]), metrics)

    splits = list(control.splitter.split(x, y if classify else None))
    pooled = control.n_fold is None

    rows: list[dict[str, Any]] = []
    per_resample: list[list[dict[str, float]]] = []
    # One note for the whole search rather than one per fold, since fitting the
    # same model on 25 resamples raises the same condition of the data 25 times.
    with quiet_engine(label):
        for position in range(len(grid.index)):
            params = dict(grid.iloc[position])
            folded = _score_folds(build, params, x, y, splits, score, pooled)
            per_resample.append(folded)
            summary = dict(params)
            for metric in metrics:
                values = np.array([fold[metric] for fold in folded], dtype=float)
                summary[metric] = resample_mean(values)
                if not pooled:
                    summary[f"{metric}SD"] = resample_spread(values)
            rows.append(summary)

    results = pd.DataFrame(rows)
    ranked = results[metrics[0]].to_numpy(dtype=float)
    if np.isnan(ranked).all():
        at = 0
    else:
        at = int(np.nanargmax(ranked) if maximize else np.nanargmin(ranked))

    resampling = None
    if not pooled:
        resampling = pd.DataFrame(per_resample[at])
        resampling["Resample"] = resample_labels(control, len(splits))

    return Resampled(results, resampling, dict(grid.iloc[at]), metrics)


def resample_mean(values: np.ndarray) -> float:
    """The mean over the resamples that could be scored.

    A metric no resample could answer - ``Kappa`` where every fold predicted one
    class - averages to missing rather than to a warning from the mean of an
    empty selection.
    """
    usable = values[~np.isnan(values)]
    return float(usable.mean()) if usable.size > 0 else math.nan


def resample_spread(values: np.ndarray) -> float:
    """The spread across the resamples that could be scored.

    Two are needed for a spread, which is why one usable resample gives missing
    rather than 0: a single number does not disagree with itself.
    """
    usable = values[~np.isnan(values)]
    return float(usable.std(ddof=1)) if usable.size > 1 else math.nan


def _score_folds(
    build: Callable[[Mapping[str, Any]], Any],
    params: Mapping[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    score: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    pooled: bool,
) -> list[dict[str, float]]:
    """Fit one candidate in every fold and score what it predicted."""
    values = np.asarray(x, dtype=float)
    held_out: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for train, test in splits:
        estimator = build(params)
        estimator.fit(values[train], y[train])
        held_out.append(np.asarray(y)[test])
        predicted.append(np.asarray(estimator.predict(values[test])))

    if pooled:
        return [score(np.concatenate(held_out), np.concatenate(predicted))]
    return [score(truth, guess) for truth, guess in zip(held_out, predicted, strict=True)]


def model_frame(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop a result table that came back with its columns and no rows.

    Port of ``sa_model_frame()``. An empty table reads as a result that is
    missing something rather than as one for which the question does not arise,
    which is the same reason ``posthoc`` is absent from a two-group comparison
    instead of present and empty.
    """
    if table is None or len(table.index) == 0:
        return None
    return table.reset_index(drop=True)


@contextmanager
def quiet_engine(label: str) -> Iterator[None]:
    """Run the engine, and report its notes once rather than per fold.

    Port of ``sa_quiet_engine()``. Cross-validation fits the same model many
    times, so a condition of the data such as a perfectly separated logistic
    regression is raised once per fold. The warnings are collected and re-emitted
    as one note with a count. They are not discarded: a model that did not
    converge has to say so.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    if caught:
        grouped = Counter(str(entry.message) for entry in caught)
        notify(
            f"{label}: engine note(s) while fitting:\n"
            + "\n".join(f"  [{count} time(s)] {text}" for text, count in grouped.items())
        )


# --------------------------------------------------------------------------- #
# Tuning grids
# --------------------------------------------------------------------------- #

#: The penalty corners :func:`enet_grid` names, in the order R lists them.
PENALTIES = ("elastic_net", "lasso", "ridge")


def enet_grid(penalty: str, alpha: Any, lambda_: Any, cv: bool) -> pd.DataFrame:
    """Turn the penalty arguments into a tuning grid.

    Port of ``sa_enet_grid()``. ``penalty`` is a name for a corner of the same
    model: the elastic net penalty is a mixture of the L1 and L2 ones and
    ``alpha`` is the mixing weight, so a lasso is ``alpha = 1`` and a ridge is
    ``alpha = 0``. Naming the corner rather than the number means the two cases
    cannot be asked for wrongly.

    ``alpha`` is validated even when ``penalty`` fixes it, for the same reason
    :func:`train_control` validates the fold count of a scheme with no folds.
    """
    weights = check_num_vector(alpha, "alpha", 0, 1)
    sizes = check_num_vector(lambda_, "lambda_", 0)

    if penalty == "lasso":
        weights = np.array([1.0])
    elif penalty == "ridge":
        weights = np.array([0.0])
    elif penalty == "elastic_net":
        weights = _unique_in_order(weights)
    else:
        raise SaInternalError(f"internal error: unhandled `penalty` {penalty}.")

    grid = _expand_grid(alpha=weights, lambda_=_unique_in_order(sizes))
    if not cv and len(grid.index) > 1:
        extra = " and a single `alpha`" if penalty == "elastic_net" else ""
        raise SaValueError(
            "`cv = False` fits one model, so the grid must hold one candidate, but "
            f"`alpha` and `lambda_` give {len(grid.index)}. Name a single `lambda_`"
            f"{extra}, or leave `cv = True` so that the resampling can choose."
        )
    return grid


def rf_grid(mtry: Any, p: int, classify: bool, cv: bool) -> pd.DataFrame:
    """Turn the forest arguments into a tuning grid.

    Port of ``sa_rf_grid()``. ``mtry`` is the one argument of a random forest
    that is tuned, so it is the one that becomes a grid; ``ntree`` and
    ``nodesize`` are passed through to the engine and are the same for every
    candidate.

    ``None`` resolves to the rule of thumb rather than to a grid, which is where
    this departs from :func:`enet_grid`. A penalty has no default size, so the
    elastic net has to search for one; ``mtry`` does have a default worth
    fitting - the square root of the predictor count for a classification and a
    third of it for a regression - so ``cv = False`` is a complete call.

    A value above the predictor count is refused rather than passed on, since a
    forest fitted at a clamped ``mtry`` is a forest at a different ``mtry`` from
    the one the result would record.
    """
    if mtry is None:
        default = math.floor(math.sqrt(p)) if classify else math.floor(p / 3)
        values = np.array([max(1, default)], dtype=float)
    else:
        values = check_num_vector(mtry, "mtry", 1)
        fractional = _unique_in_order(values[values != np.trunc(values)])
        if fractional.size > 0:
            raise SaValueError(
                "`mtry` counts predictors, so it must hold whole numbers, but holds "
                + ", ".join(str(value) for value in fractional)
                + "."
            )
        above = _unique_in_order(values[values > p])
        if above.size > 0:
            raise SaValueError(
                f"`mtry` cannot exceed the {p} predictor(s) the model has, but holds "
                + ", ".join(str(int(value)) for value in above)
                + ". A forest fitted at a clamped `mtry` is not the forest the result "
                "would report."
            )

    grid = pd.DataFrame({"mtry": _unique_in_order(values).astype(int)})
    if not cv and len(grid.index) > 1:
        raise SaValueError(
            "`cv = False` fits one forest, so the grid must hold one candidate, but "
            f"`mtry` gives {len(grid.index)}. Name a single `mtry`, or leave "
            "`cv = True` so that the resampling can choose."
        )
    return grid


def svm_grid(C: Any, sigma: Any, cv: bool) -> pd.DataFrame:  # noqa: N803 - R's argument name
    """Turn the machine's arguments into a tuning grid.

    Port of ``sa_svm_grid()``. Both arguments of a radial-kernel machine are
    tuned, so both become the grid.

    Zero is rejected by name rather than by the bound, since neither argument is
    answerable there: a machine at ``C = 0`` pays nothing for violating its
    margin and fits a flat surface, and a kernel at ``sigma = 0`` reports the
    same distance between every pair of rows. The engine takes both and returns a
    fit, so the refusal has to be here.
    """
    specs = (
        (C, "C", "a machine that pays nothing for violating its margin fits a flat surface"),
        (sigma, "sigma", "a kernel of no width reports the same distance between every pair"),
    )
    resolved = []
    for value, arg, reason in specs:
        numbers = check_num_vector(value, arg, 0)
        if bool((numbers == 0).any()):
            raise SaValueError(
                f"`{arg}` must be above 0, but holds 0: {reason}, and the engine fits it "
                "without complaint."
            )
        resolved.append(_unique_in_order(numbers))

    grid = _expand_grid(sigma=resolved[1], C=resolved[0])
    if not cv and len(grid.index) > 1:
        raise SaValueError(
            "`cv = False` fits one machine, so the grid must hold one candidate, but "
            f"`C` and `sigma` give {len(grid.index)}. Name a single `C` and a single "
            "`sigma`, or leave `cv = True` so that the resampling can choose."
        )
    return grid


def no_grid() -> pd.DataFrame:
    """The grid of a model that tunes nothing: one candidate, no parameters.

    ``caret`` reports one row of ``results`` for a model with nothing to tune, so
    the resampled score of a linear regression is reached the same way as the
    score of the elastic net that was chosen from fifty candidates.
    """
    return pd.DataFrame(index=pd.RangeIndex(1))


def _unique_in_order(values: np.ndarray) -> np.ndarray:
    """R's ``unique()``: first appearance order rather than sorted."""
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def _expand_grid(**columns: np.ndarray) -> pd.DataFrame:
    """R's ``expand.grid()``: the first column varies fastest."""
    names = list(columns)
    mesh = np.meshgrid(*(columns[name] for name in names), indexing="ij")
    ordered = [axis.reshape(-1, order="F") for axis in mesh]
    return pd.DataFrame(dict(zip(names, ordered, strict=True)))


# --------------------------------------------------------------------------- #
# Input resolution and coding
# --------------------------------------------------------------------------- #


@dataclass
class ModelInput:
    """The outcome and the predictors, told apart and reduced to usable rows.

    Attributes:
        x: The predictor columns, with character columns turned into categoricals
            and their unused levels dropped.
        y: The outcome of those rows.
        outcome: What to call the outcome in ``design``.
        predictors: The columns kept, in the order they were named.
        dropped_predictors: Those left out for taking a single value.
        predictor_lv: The levels of every predictor that has them.
        n_obs: Rows passed in.
        n_used: Rows complete across the outcome and the predictors.
        n_dropped: The difference.
    """

    x: pd.DataFrame
    y: pd.Series
    outcome: str
    predictors: list[str]
    dropped_predictors: list[str]
    predictor_lv: dict[str, list[str]]
    n_obs: int
    n_used: int
    n_dropped: int


def resolve_model_input(data: Any, outcome: Any, predictors: Any = None) -> ModelInput:
    """Resolve the outcome and the predictors out of one data frame.

    Port of ``sa_resolve_model_input()``. The model functions take the same wide
    frame the comparison functions take, which here is the training half
    :func:`~statassist.split_data` handed back. What differs is that one of its
    columns is the outcome and the rest are candidates for predictors.

    Rows with a missing value anywhere in the model are dropped here rather than
    inside the engine. Left to the engine, deletion happens once per fold on
    whatever each fold happens to hold, so the folds would be scored on different
    subsets of the data and the resampled numbers would not be comparable.

    A predictor that takes one value cannot contribute, and leaving it in makes
    the engine return an estimate that reads like a failure rather than like a
    column with nothing in it. It is dropped with a note instead.
    """
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise SaValueError("`data` must be a data.frame or a matrix.")
    n_obs = len(data.index)
    if n_obs == 0:
        raise SaValueError("`data` has zero rows.")
    data = data.reset_index(drop=True)

    # A missing outcome is allowed through the resolver: it marks a row the
    # listwise deletion below removes and counts, not a call that cannot proceed.
    resolved = resolve_row_vector(outcome, "outcome", data, allow_na=True)
    if resolved.value is None:
        raise SaValueError(
            "`outcome` must name a column of `data` or hold one entry per row of it."
        )
    y = resolved.value
    label = str(resolved.label)

    names = _resolve_predictor_names(data, predictors, label)
    x = data[names].copy()

    unsupported = [name for name in names if not _is_usable_predictor(x[name])]
    if unsupported:
        raise SaValueError(
            "`predictors` must be numeric, logical, factor or character columns. "
            "Not usable: " + ", ".join(unsupported) + "."
        )

    keep = x.notna().all(axis=1).to_numpy() & ~y.isna().to_numpy()
    n_used = int(keep.sum())
    if n_used < 2:
        raise SaValueError(
            f"only {n_used} row(s) of `data` are complete across `outcome` and "
            "`predictors`; at least 2 are needed."
        )
    x = x.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    # Character columns become categoricals here rather than inside the engine,
    # so that the levels are fixed before the folds are drawn. Levels left over
    # from the row filtering go for the same reason a constant column does: an
    # all-zero dummy column is not a predictor.
    for name in names:
        column = x[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            x[name] = column.cat.remove_unused_categories()
        elif column.dtype == object or pd.api.types.is_string_dtype(column):
            x[name] = pd.Categorical(column.astype(str))

    constant = [name for name in names if x[name].nunique(dropna=False) < 2]
    if constant:
        x = x.drop(columns=constant)
        notify(
            "predictor(s) with a single value cannot contribute and were left out: "
            + ", ".join(constant)
            + "."
        )
    kept = [name for name in names if name not in constant]
    if not kept:
        raise SaValueError(
            "every predictor takes a single value over the usable rows, so there is nothing to fit."
        )

    return ModelInput(
        x=x,
        y=y,
        outcome=label,
        predictors=kept,
        dropped_predictors=constant,
        predictor_lv={name: levels for name in kept if (levels := _levels_of(x[name])) is not None},
        n_obs=n_obs,
        n_used=n_used,
        n_dropped=n_obs - n_used,
    )


def _resolve_predictor_names(data: pd.DataFrame, predictors: Any, outcome: str) -> list[str]:
    """Which columns are the predictors, checked the way R checks them."""
    if predictors is None:
        # An outcome passed as a vector leaves no column to exclude, so every
        # column is a candidate. Passed as a name, that column is the one thing a
        # predictor must not be.
        return [str(name) for name in data.columns if str(name) != outcome]

    if isinstance(predictors, str) or not hasattr(predictors, "__iter__"):
        raise SaValueError(
            "`predictors` must be a non-empty sequence of column names, or None for "
            "every column except `outcome`."
        )
    names = [None if value is None else str(value) for value in predictors]
    if not names or any(name is None for name in names):
        raise SaValueError(
            "`predictors` must be a non-empty sequence of column names, or None for "
            "every column except `outcome`."
        )
    kept = [name for name in names if name is not None]

    seen: dict[str, int] = Counter(kept)
    duplicated = [name for name in dict.fromkeys(kept) if seen[name] > 1]
    if duplicated:
        raise SaValueError("`predictors` contains duplicated names: " + ", ".join(duplicated) + ".")
    known = {str(name) for name in data.columns}
    unknown = [name for name in kept if name not in known]
    if unknown:
        raise SaValueError("`predictors` not found in `data`: " + ", ".join(unknown) + ".")
    if outcome in kept:
        raise SaValueError(
            f"`predictors` contains the outcome column `{outcome}`, which would let the "
            "model predict from the answer."
        )
    return kept


def _is_usable_predictor(column: pd.Series) -> bool:
    """Whether a column is one of the four kinds a model can read."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
        return True
    return bool(column.dtype == object or pd.api.types.is_string_dtype(column))


def _levels_of(column: pd.Series) -> list[str] | None:
    """The levels a predictor is coded against, or ``None`` for a numeric one.

    A logical column has levels in R's sense - ``model.matrix()`` codes it as
    ``nameTRUE`` - and none that have to be recorded, since ``False`` and
    ``True`` are not something a held-out half can be missing.
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        return [str(level) for level in column.cat.categories]
    return None


def design_lv(predictor_lv: Mapping[str, list[str]]) -> dict[str, Any]:
    """The levels entry of ``design``, left out when nothing has levels.

    Port of ``sa_design_lv()``. ``design`` reports what the model saw, and a set
    of models sees no factor at all. An empty mapping there would be an entry
    that says "these are the levels" about nothing, which is the same reason
    ``outcome_lv`` is absent from a regression rather than present and empty.
    """
    if not predictor_lv:
        return {}
    return {"predictor_lv": dict(predictor_lv)}


def design_matrix(
    x: pd.DataFrame,
    xlev: Mapping[str, list[str]] | None = None,
    want: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Dummy code the predictor frame into the matrix an engine takes.

    Port of ``sa_design_matrix()``. The coding is
    :func:`stats::model.matrix`'s, the same one ``lm()`` applies, so a ``k``-level
    factor becomes the same ``k - 1`` terms under the same names and two models'
    coefficient tables can be read side by side. The intercept column is not
    among them: a model that has one fits it separately.

    Handing a factor to an estimator as its integer codes is what this exists to
    prevent. It fits without complaint and is wrong: a three-level factor arrives
    as one evenly spaced numeric predictor, so the model assumes an order and a
    spacing between the levels that nobody stated.

    Predicting on new rows codes them here as well. ``xlev`` fixes the levels, so
    a level no row of ``newdata`` happens to take still gets its column of
    zeroes, and ``want`` fixes the order by name, since a design matrix is read
    by position once it reaches the engine. Rows are kept whatever they hold: a
    missing cell has to reach the caller as a missing prediction rather than as a
    row that quietly went absent.

    Args:
        x: Predictor frame, or the same columns of the rows to predict.
        xlev: Levels to code against, as ``design["predictor_lv"]`` holds them,
            or ``None`` to read them off ``x``.
        want: Column names the result must have, in the order it must have them.

    Returns:
        One column per model term, no intercept.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"a": [1.0, 2.0], "g": pd.Categorical(["lo", "hi"])})
        >>> list(design_matrix(frame).columns)
        ['a', 'glo']
    """
    columns = {term: values for term, _, values in _design_columns(x, xlev) if values is not None}
    matrix = pd.DataFrame(columns, index=pd.RangeIndex(len(x.index)))
    if want is not None:
        absent = [name for name in want if name not in matrix.columns]
        if absent:
            raise SaInternalError(
                "internal error: the coding of `newdata` is missing term(s) the model "
                "has: " + ", ".join(absent) + "."
            )
        matrix = matrix[list(want)]
    return matrix


def design_source(x: pd.DataFrame, xlev: Mapping[str, list[str]] | None = None) -> dict[str, str]:
    """Which predictor each column of :func:`design_matrix` came from.

    A factor becomes several columns, so anything reported per column has to be
    brought back to the predictor before it can be read as a statement about the
    data: three dummies of one factor are three shares of one predictor's worth.

    Built by the same walk over the frame that does the coding, rather than by
    reading the names apart afterwards. There is no reading them apart: a
    predictor called ``x`` and a factor called ``x`` with a level ``1`` would both
    claim the column ``x1``.
    """
    return {term: source for term, source, _ in _design_columns(x, xlev)}


def _design_columns(
    x: pd.DataFrame, xlev: Mapping[str, list[str]] | None
) -> Iterator[tuple[str, str, np.ndarray | None]]:
    """Walk the predictor frame, naming and coding one model term at a time.

    Yields the term name, the predictor it came from, and its column.
    """
    for name in x.columns:
        values = x[name]
        levels = None if xlev is None else xlev.get(str(name))
        if levels is None:
            levels = _levels_of(values)

        if levels is None:
            if pd.api.types.is_bool_dtype(values):
                yield f"{name}TRUE", str(name), values.astype(float).to_numpy()
            else:
                coded = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                yield str(name), str(name), coded
            continue

        # `NA` reaches every dummy of the factor it came from rather than being
        # read as "not this level", which is what `na.action = na.pass` does.
        codes = values.astype(object).map(lambda entry: None if pd.isna(entry) else str(entry))
        missing = codes.isna().to_numpy()
        for level in levels[1:]:
            dummy = (codes == level).to_numpy(dtype=float)
            dummy[missing] = np.nan
            yield f"{name}{level}", str(name), dummy


def predict_frame(newdata: Any, design: Mapping[str, Any]) -> pd.DataFrame:
    """Reduce new rows to the predictors the model was fitted on.

    Port of ``sa_predict_frame()``. What a model needs of ``newdata`` is its own
    predictors, coded the way they were coded when it was fitted, and nothing
    else. Columns it never saw are ignored rather than refused, since the rows to
    predict usually arrive as the other half of the same frame the fit was given,
    outcome column and all. A predictor that is not there at all is an error
    naming it.

    The levels are put back rather than read off ``newdata``, and that is the
    whole reason this exists. A factor in a held-out half may be missing a level,
    or carry its levels in another order, and either would code to a different
    matrix from the one the model was fitted to. A level the fit never saw is an
    error instead, since there is no coefficient to apply to it.
    """
    if isinstance(newdata, np.ndarray) and newdata.ndim == 2:
        newdata = pd.DataFrame(newdata)
    if not isinstance(newdata, pd.DataFrame):
        raise SaValueError("`newdata` must be a data.frame or a matrix.")
    if len(newdata.index) == 0:
        raise SaValueError("`newdata` has zero rows.")

    predictors = [str(name) for name in design["predictors"]]
    absent = [name for name in predictors if name not in newdata.columns]
    if absent:
        raise SaValueError(
            "`newdata` is missing predictor column(s) the model was fitted on: "
            + ", ".join(absent)
            + "."
        )

    levels_by_name = design.get("predictor_lv") or {}
    frame = newdata[predictors].reset_index(drop=True).copy()
    for name in predictors:
        column = frame[name]
        levels = levels_by_name.get(name)

        if levels is None:
            if not (pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column)):
                raise SaValueError(
                    f"`{name}` was a numeric predictor when the model was fitted, and "
                    f"`newdata` holds it as {column.dtype}. The coding of a column "
                    "cannot change between fitting and predicting."
                )
            continue

        # The missing entries are marked off before the values are read rather
        # than looked for among them: `map()` hands a `None` back as a `NaN` in an
        # object column, and a float among the levels of a factor sorts against
        # nothing.
        missing = column.isna().to_numpy()
        as_text = pd.Series(
            [None if gone else str(entry) for entry, gone in zip(column, missing, strict=True)],
            dtype=object,
        )
        unseen = sorted({value for value in as_text[~missing]} - set(levels))
        if unseen:
            raise SaValueError(
                f"`newdata` holds level(s) of `{name}` the model was not fitted on, so "
                "there is no coefficient for them: "
                + ", ".join(unseen)
                + ". Fitted on: "
                + ", ".join(levels)
                + "."
            )
        frame[name] = pd.Categorical(as_text, categories=levels)
    return frame


#: How many classes a model of this family reads. Two, and it is the whole model.
N_CLASSES = 2


def outcome_levels(
    y: pd.Series,
    outcome_lv: Any = None,
    control_label: Any = None,
    model: str = "a logistic regression",
) -> list[str]:
    """Put a binary outcome in the order that fixes the direction.

    Port of ``sa_outcome_levels()``. The first level is the reference at every
    group count in this package, so it is the level a fold change divides by and
    the one a post-hoc contrast subtracts. A logistic regression follows the same
    rule: with ``outcome_lv = ("control", "case")`` every coefficient is the
    change in the log odds of ``case``, and its odds ratio is above 1 for a
    predictor that raises the chance of ``case``.

    A third level is an error rather than a dropped set of rows. Two levels are
    what this model is, so silently fitting a different subset of the data than
    was passed in would answer a question nobody asked.

    ``control_label`` names the same level ``outcome_lv[0]`` names, and exists
    because most calls have nothing to say about the other one. Naming the
    reference twice and disagreeing is an error rather than a precedence rule:
    either argument is a complete answer on its own.

    Returns:
        The two levels, reference first.
    """
    named = outcome_lv is not None
    as_text = [None if pd.isna(entry) else str(entry) for entry in y]
    present = list(dict.fromkeys(value for value in as_text if value is not None))
    if len(present) < N_CLASSES:
        raise SaValueError(
            "`outcome` takes a single value over the usable rows, so there is nothing to classify."
        )

    if not named:
        if len(present) > N_CLASSES:
            raise SaValueError(
                f"`outcome` holds {len(present)} classes, but {model} models "
                f"{N_CLASSES}: "
                + ", ".join(sorted(present))
                + ". Name the two to model with `outcome_lv`, or reduce `data` to them "
                "first."
            )
        levels = sorted(present)
    else:
        if isinstance(outcome_lv, str) or not hasattr(outcome_lv, "__iter__"):
            raise SaValueError(
                "`outcome_lv` must be two distinct level names, the reference first."
            )
        levels = [str(value) for value in outcome_lv]
        if len(levels) != N_CLASSES or len(set(levels)) != N_CLASSES:
            raise SaValueError(
                "`outcome_lv` must be two distinct level names, the reference first."
            )
        absent = [level for level in levels if level not in present]
        if absent:
            raise SaValueError(
                "`outcome_lv` level(s) absent from `outcome`: "
                + ", ".join(absent)
                + ". Present: "
                + ", ".join(sorted(present))
                + "."
            )

    # A named pair that leaves classes out would fit the model on a subset of the
    # rows that were passed in, which is a different data set from the one the
    # call describes.
    extra = sorted(set(present) - set(levels))
    if extra:
        raise SaValueError(
            f"`outcome` holds {len(present)} classes and `outcome_lv` names "
            f"{N_CLASSES} of them, so {len(extra)} would be silently left out: "
            + ", ".join(extra)
            + ". Reduce `data` to the two classes first."
        )

    if control_label is not None:
        reference = str(control_label)
        if reference not in levels:
            raise SaValueError(
                f"`control_label` names a class `outcome` does not hold: {reference}. "
                "Present: " + ", ".join(sorted(present)) + "."
            )
        if named and reference != levels[0]:
            raise SaValueError(
                f"`control_label` names {reference} as the reference and `outcome_lv` "
                f"puts {levels[0]} first, so the two disagree about which class the "
                "other one is compared against. Pass one of them."
            )
        levels = [reference] + [level for level in levels if level != reference]
    return levels


def encode_outcome(y: pd.Series, outcome_lv: Sequence[str]) -> np.ndarray:
    """The outcome as the engine takes it: 1 for the class being modelled.

    ``outcome_lv[1]`` is the class every coefficient and odds ratio describes, so
    it is the one that becomes the positive label. Encoding it here rather than
    letting the estimator sort the labels is what keeps the direction from
    depending on how the two classes happen to be spelled.
    """
    event = str(outcome_lv[1])
    return np.array([1 if str(entry) == event else 0 for entry in y], dtype=int)


@dataclass
class Outcome:
    """The outcome, read as one kind of thing rather than the other.

    Attributes:
        classify: Whether it was read as a class.
        y: The outcome as the engine takes it, floats for a regression and 0/1
            for a classification.
        levels: The two classes, reference first, or ``None``.
        n_events: Rows in ``levels[1]``, or ``None``.
    """

    classify: bool
    y: np.ndarray
    levels: list[str] | None
    n_events: int | None


def resolve_outcome(
    y: pd.Series,
    outcome_lv: Any,
    control_label: Any,
    model: str,
    non_finite: str,
) -> Outcome:
    """Decide whether the model is a regression or a classification.

    The three models that fit either kind - the elastic net, the forest and the
    machine - decide it the same way, so they decide it here. Naming the classes
    with ``outcome_lv`` or ``control_label`` asks for a classification, and so
    does an outcome that is not numeric; everything else is a regression.

    A numeric column taking two values is the one case where both readings are
    plausible. It is fitted as a regression, since that is what the column says
    it is, and a note says so and says how to ask for the other reading. Guessing
    from the value count would make a regression on a rounded outcome silently
    become a classification.

    Args:
        y: The outcome of the usable rows.
        outcome_lv: The two classes, reference first, or ``None``.
        control_label: The reference class on its own, or ``None``.
        model: What to call the model where a message names it.
        non_finite: Why a non-finite outcome cannot be fitted, in this model's
            own terms. Reached only on the regression path.
    """
    named = outcome_lv is not None or control_label is not None
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)

    if not named and numeric:
        if int(y.nunique()) == N_CLASSES:
            notify(
                "`outcome` is numeric and takes two values, so it was fitted as a "
                "regression. Pass `outcome_lv` or `control_label`, or a categorical "
                "column, to model it as a classification."
            )
        values = y.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SaValueError(f"`outcome` holds non-finite value(s), {non_finite}")
        return Outcome(classify=False, y=values, levels=None, n_events=None)

    levels = outcome_levels(y, outcome_lv, control_label, model)
    encoded = encode_outcome(y, levels)
    return Outcome(classify=True, y=encoded, levels=levels, n_events=int(encoded.sum()))


def model_design(input_: ModelInput, outcome: Outcome) -> dict[str, Any]:
    """The ``design`` slot: what the model saw, in the order every model reports it.

    Written once because it is the same answer every time. The classification
    entries sit next to ``outcome_type`` rather than at the end, since which
    class is being modelled is part of saying what the outcome was.
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


def resample_params(cv: Any, control: ResampleControl, seed: Any) -> dict[str, Any]:
    """The resampling entries of ``parameters``, as they were actually used.

    Read off the resolved scheme rather than off the arguments, so that a
    ``cv=False`` call records no fold count instead of the one it was passed and
    ignored.
    """
    return {
        "cv": bool(cv),
        "cv_method": control.cv_method,
        "n_fold": control.n_fold,
        "n_repeat": control.n_repeat,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def wald_interval(
    estimate: np.ndarray,
    stderr: np.ndarray,
    conf_level: float,
    df: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The confidence interval that agrees with the standard error beside it.

    Port of ``sa_wald_interval()``. Built from the standard error rather than
    profiled, for two reasons. A profile likelihood interval is a better interval
    but a different quantity from the Wald standard error and z value in the same
    row, so the three numbers would not agree. And a term the fit could not
    estimate comes back missing here like any other unanswerable question, rather
    than making the whole interval fail.

    Args:
        estimate: The coefficients.
        stderr: Their standard errors.
        conf_level: Two-sided confidence level.
        df: Residual degrees of freedom for a t interval, or ``None`` for the
            normal approximation.
    """
    upper_tail = (1 - conf_level) / 2
    if df is None or not math.isfinite(df):
        crit = float(sp_stats.norm.ppf(1 - upper_tail))
    else:
        crit = float(sp_stats.t.ppf(1 - upper_tail, df))
    return estimate - crit * stderr, estimate + crit * stderr


def inference_table(
    terms: Sequence[str],
    estimate: np.ndarray,
    stderr: np.ndarray,
    conf_level: float,
    df: float | None = None,
) -> pd.DataFrame:
    """Assemble the coefficient table of a model that reports inference.

    Port of ``sa_coef_table()``, with the statistic and its p-value computed here
    rather than read off a summary object. ``df`` decides which distribution the
    statistic is referred to and therefore what the column means: a number gives
    a t on that many degrees of freedom, as ``lm()`` reads, and ``None`` a Wald z
    referred to the normal, as ``glm()`` reads.

    A term the fit could not estimate keeps its row with everything about it
    missing. Dropping it would make the table quietly shorter than the model it
    describes.
    """
    estimate = np.asarray(estimate, dtype=float)
    stderr = np.asarray(stderr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = estimate / stderr
    if df is None:
        pval = 2 * sp_stats.norm.sf(np.abs(statistic))
    else:
        pval = 2 * sp_stats.t.sf(np.abs(statistic), df)
    lower, upper = wald_interval(estimate, stderr, conf_level, df)

    table = pd.DataFrame(
        {
            "terms": [str(name) for name in terms],
            "estimate": estimate,
            "stderr": stderr,
            "statistic": statistic,
            "df": np.full(len(estimate), math.nan if df is None else float(df)),
            "pval": np.asarray(pval, dtype=float),
            "lower_conf": lower,
            "upper_conf": upper,
        }
    )
    return table[model_coef_columns() + model_inference_columns()]


@dataclass
class LinearFit:
    """An unpenalized fit, with the terms the data could not support named.

    Attributes:
        estimator: The fitted engine object.
        terms: Every term the model has, intercept first.
        estimate: One coefficient per term, missing for an aliased one.
        stderr: Their standard errors, missing the same way.
        aliased: The terms the other predictors already span.
        rank: How many terms the data could support.
        fitted: The linear predictor on the rows that were fitted.
    """

    estimator: Any
    terms: list[str]
    estimate: np.ndarray
    stderr: np.ndarray
    aliased: list[str]
    rank: int
    fitted: np.ndarray


#: What the intercept term is called, in R's spelling.
INTERCEPT = "(Intercept)"


def _estimable(matrix: np.ndarray) -> np.ndarray:
    """Which columns of a design matrix the data can tell apart.

    The columns are taken left to right and one is kept when it carries
    something the kept ones do not: the part of it orthogonal to them, measured
    against its own length so that the answer does not depend on the units the
    column is in.

    The order is the whole point, and it is why a fully pivoted decomposition
    will not do here. Pivoting reaches for the longest column first, so of two
    predictors that alias each other it keeps whichever happens to be on the
    larger scale, and the term that comes back estimated then depends on the
    units rather than on the model. ``lm()`` pivots only far enough to push the
    deficient columns to the end and leaves the rest in the order they were
    given, so the intercept and the first-named predictors are the terms that get
    estimated. That is the behaviour reproduced here.
    """
    n_terms = matrix.shape[1]
    keep = np.zeros(n_terms, dtype=bool)
    basis: list[np.ndarray] = []
    for position in range(n_terms):
        column = np.asarray(matrix[:, position], dtype=float)
        scale = float(np.linalg.norm(column))
        if scale == 0:
            continue
        residual = column
        # Twice, because one pass loses accuracy exactly where the answer is
        # decided: a column that nearly lies in the span of the kept ones.
        for _ in range(2):
            for vector in basis:
                residual = residual - float(vector @ residual) * vector
        size = float(np.linalg.norm(residual))
        if size > RANK_TOL * scale:
            keep[position] = True
            basis.append(residual / size)
    return keep


@dataclass
class DesignEstimator:
    """An engine fitted on an intercept column plus the estimable terms.

    :func:`predict_model` hands over the design matrix as the model's own
    ``x_names`` order fixes it: no intercept column, and every term present
    including the aliased ones. This puts the two back, so that a caller predicts
    from the terms the model has rather than from the columns the engine happened
    to be handed.
    """

    estimator: Any
    keep: np.ndarray

    def _prepare(self, x: Any) -> np.ndarray:
        matrix = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
        return matrix[:, self.keep]

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._prepare(x)))

    def predict_proba(self, x: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self._prepare(x)), dtype=float)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimator.classes_)


def least_squares(x: pd.DataFrame, y: np.ndarray, label: str) -> LinearFit:
    """Fit ordinary least squares, and name the terms it could not estimate.

    The solve is the engine's. What is added is the rank check: a term the other
    predictors already span has no coefficient of its own, and the engine would
    answer with one anyway - a minimum-norm solution spreads the estimate over
    the columns that alias each other. Here such a term comes back missing and
    named, which is what ``lm()`` reports and what makes the table say that the
    term was in the model and could not be estimated.
    """
    from sklearn.linear_model import LinearRegression

    terms = [INTERCEPT] + [str(name) for name in x.columns]
    matrix = np.column_stack([np.ones(len(x.index)), np.asarray(x, dtype=float)])
    keep = _estimable(matrix)

    engine = LinearRegression(fit_intercept=False)
    with quiet_engine(label):
        engine.fit(matrix[:, keep], y)
    coefficients = np.asarray(engine.coef_, dtype=float).reshape(-1)

    estimate = np.full(len(terms), math.nan)
    estimate[keep] = coefficients
    fitted = matrix[:, keep] @ coefficients

    rank = int(keep.sum())
    residual_df = len(y) - rank
    stderr = np.full(len(terms), math.nan)
    if residual_df > 0:
        sigma_sq = float(np.sum((y - fitted) ** 2) / residual_df)
        stderr[keep] = np.sqrt(np.diag(_xtx_inverse(matrix[:, keep])) * sigma_sq)

    return LinearFit(
        estimator=DesignEstimator(engine, keep),
        terms=terms,
        estimate=estimate,
        stderr=stderr,
        aliased=[term for term, kept in zip(terms, keep, strict=True) if not kept],
        rank=rank,
        fitted=fitted,
    )


#: Iterations the unpenalized logistic solver is allowed.
#:
#: Well above the default, because there is no penalty to keep the estimates
#: finite: a predictor that separates the two classes exactly sends its
#: coefficient off to infinity, and the solver has to be allowed to get far
#: enough for the standard error beside it to say so. It stops and reports rather
#: than converging, which :func:`quiet_engine` turns into one counted note.
LOGIT_MAX_ITER = 1000

#: How far the logistic solver is asked to go, four orders below the default.
#:
#: The default stops while the gradient of the log likelihood is still around
#: ``1e-2``, which is close enough for a classifier and not close enough for a
#: regression: the Wald standard errors, z statistics and p-values beside the
#: coefficients are the curvature *at the maximum*, so a fit that stopped short
#: of it reports uncertainty about a point that is not the estimate. At this
#: tolerance the gradient reaches the floor the solver can achieve at all.
LOGIT_TOL = 1e-8

#: What "no penalty" is called for the engine's logistic regression.
#:
#: An infinite budget rather than ``penalty=None``, which the engine deprecated:
#: the two fit the same model, and only this one does it without a warning per
#: fold.
LOGIT_NO_PENALTY = math.inf


def logistic_estimator(fit_intercept: bool = True) -> Any:
    """An unfitted logistic regression with no penalty on its coefficients.

    Shared by the final fit and by the fold fits, so that the model scored by
    the resampling is the model the coefficient table describes. The engine
    penalizes by default, and a penalized fit is a different model: its
    estimates are shrunk and its Wald standard errors are not the curvature of
    the likelihood at the maximum.

    Args:
        fit_intercept: Whether the engine adds the intercept. ``False`` for the
            fit that reports coefficients, since there the intercept is a column
            of the design matrix and so gets a row in the table like any other
            term; ``True`` for a fold, which is handed the matrix as the model's
            own ``x_names`` order fixes it and has no such column.
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=LOGIT_NO_PENALTY,
        fit_intercept=fit_intercept,
        solver="lbfgs",
        max_iter=LOGIT_MAX_ITER,
        tol=LOGIT_TOL,
    )


def logistic_fit(x: pd.DataFrame, y: np.ndarray, label: str) -> LinearFit:
    """Fit a binomial logistic regression, unpenalized.

    The absent penalty is what makes this a maximum likelihood fit rather than
    the ridge the engine applies by default, which is the model
    :func:`~statassist.fit_logistic_regression` documents and the only one whose
    Wald standard errors mean what they say.

    ``fitted`` on the result is the linear predictor, not the probability, so
    that the caller decides which scale to read it on.
    """
    terms = [INTERCEPT] + [str(name) for name in x.columns]
    matrix = np.column_stack([np.ones(len(x.index)), np.asarray(x, dtype=float)])
    keep = _estimable(matrix)

    engine = logistic_estimator(fit_intercept=False)
    with quiet_engine(label):
        engine.fit(matrix[:, keep], y)
    coefficients = np.asarray(engine.coef_, dtype=float).reshape(-1)

    estimate = np.full(len(terms), math.nan)
    estimate[keep] = coefficients
    eta = matrix[:, keep] @ coefficients
    probability = 1 / (1 + np.exp(-eta))

    stderr = np.full(len(terms), math.nan)
    stderr[keep] = weighted_least_squares(matrix[:, keep], probability * (1 - probability))

    return LinearFit(
        estimator=DesignEstimator(engine, keep),
        terms=terms,
        estimate=estimate,
        stderr=stderr,
        aliased=[term for term, kept in zip(terms, keep, strict=True) if not kept],
        rank=int(keep.sum()),
        fitted=eta,
    )


def weighted_least_squares(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Standard errors from the information matrix of a weighted fit.

    What a logistic regression's are: the observed information is
    ``X' W X`` with ``W`` the variance of each fitted probability, so the square
    roots of the diagonal of its inverse are the Wald standard errors reported
    beside the z statistics.
    """
    scaled = matrix * np.sqrt(weights)[:, None]
    return np.sqrt(np.diag(_xtx_inverse(scaled)))


def _xtx_inverse(matrix: np.ndarray) -> np.ndarray:
    """``inv(X' X)``, through the pseudo-inverse so a singular fit still answers."""
    return np.asarray(np.linalg.pinv(matrix.T @ matrix), dtype=float)


def logistic_scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """The deviances a fitted logistic regression is summarised by.

    The residual deviance is ``-2`` times the log likelihood of the fit and the
    null deviance the same for the intercept alone, so the difference is the
    likelihood ratio statistic of the model against nothing.
    """
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    residual = float(-2 * np.sum(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    rate = float(np.mean(y))
    if rate in (0.0, 1.0):
        null = 0.0
    else:
        null = float(-2 * (np.sum(y) * math.log(rate) + np.sum(1 - y) * math.log(1 - rate)))
    return {"null_deviance": null, "residual_deviance": residual}


# --------------------------------------------------------------------------- #
# Scoring a fit on the rows it was fitted to
# --------------------------------------------------------------------------- #


def numeric_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """How close a predicted outcome came, for a model that reports no inference.

    ``r_squared`` is the share of the variance of the outcome the prediction
    accounts for, which is the in-sample reading and not the held-out
    ``Rsquared`` of :func:`regression_scores`.
    """
    error = observed - predicted
    total = float(np.sum((observed - observed.mean()) ** 2))
    residual = float(np.sum(error**2))
    return {
        "r_squared": 1 - residual / total if total > 0 else math.nan,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def class_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """How well a predicted class did, with 1 as the event.

    ``sensitivity`` is the share of the events the model found and
    ``specificity`` the share of the non-events it left alone, so both are
    statements about ``outcome_lv[1]`` being the class of interest. Either is
    missing rather than 0 when its denominator is empty: a model asked about no
    events did not fail to find them.
    """
    from sklearn.metrics import cohen_kappa_score

    event = observed == 1
    accuracy = float(np.mean(observed == predicted))
    if np.unique(observed).size < N_CLASSES or np.unique(predicted).size < N_CLASSES:
        kappa = math.nan
    else:
        kappa = float(cohen_kappa_score(observed, predicted))
    return {
        "accuracy": accuracy,
        "error": 1 - accuracy,
        "kappa": kappa,
        "sensitivity": float(np.mean(predicted[event] == 1)) if event.any() else math.nan,
        "specificity": float(np.mean(predicted[~event] == 0)) if (~event).any() else math.nan,
    }


# --------------------------------------------------------------------------- #
# Shared by the models that report importance rather than coefficients
# --------------------------------------------------------------------------- #


def importance_table(
    terms: Sequence[str],
    estimate: np.ndarray,
    impurity: np.ndarray | None = None,
) -> pd.DataFrame:
    """The coefficient table of a model that has no coefficients.

    A forest and a machine answer with how much each term was worth to them
    rather than with a coefficient, so the table is sorted by that: the term at
    the top is the one the model leaned on hardest. Which is the opposite
    convention from an unpenalized fit, whose rows are in the order of ``terms``
    because the intercept has to come first there.
    """
    table = pd.DataFrame(
        {
            "terms": [str(name) for name in terms],
            "estimate": np.asarray(estimate, dtype=float),
        }
    )
    if impurity is not None:
        table["impurity"] = np.asarray(impurity, dtype=float)
    return table.sort_values("estimate", ascending=False, kind="stable").reset_index(drop=True)


def rollup(
    values: Mapping[str, float], source: Mapping[str, str], predictors: Sequence[str]
) -> np.ndarray:
    """Add up what was reported per model term into one number per predictor.

    A three-level factor reaches the engine as two dummy columns, so the engine
    reports two numbers about one predictor. Their sum is what the predictor was
    worth, which is the quantity R reports directly: its engine splits on the
    factor itself and never divides it into columns.

    Adding rather than averaging is the reading that survives the difference. The
    total decrease in error attributable to a predictor does not depend on how
    many columns it was spread over, while a mean would report a factor as less
    important the more levels it has.
    """
    totals = dict.fromkeys((str(name) for name in predictors), 0.0)
    for term, value in values.items():
        name = source[term]
        totals[name] = totals[name] + float(value)
    return np.array([totals[str(name)] for name in predictors], dtype=float)


def permutation_scores(
    estimator: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    classify: bool,
    n_permute: int,
    seed: int | None,
) -> np.ndarray:
    """What each column was worth, measured by taking it away.

    A column is shuffled and the model is scored again; what the score lost is
    what the column was carrying. The engine's own routine does the shuffling,
    and the scorer decides the sign: accuracy and negative error both go down
    when something is taken away, so a larger number is a more important column
    either way.

    Args:
        estimator: The fitted model, or the pipeline that ends in it.
        x: The columns it was fitted on.
        y: The outcome it was fitted to.
        classify: Whether the outcome is a class.
        n_permute: Shuffles per column. The average over them is reported, since
            one shuffle of one column is a draw rather than a measurement.
        seed: Seed for the shuffling.
    """
    from sklearn.inspection import permutation_importance

    scored = permutation_importance(
        estimator,
        np.asarray(x, dtype=float),
        y,
        scoring="accuracy" if classify else "neg_root_mean_squared_error",
        n_repeats=n_permute,
        random_state=seed,
    )
    return np.asarray(scored.importances_mean, dtype=float)


@dataclass
class ScaledFit:
    """A penalized fit, with its coefficients back on the scale they came in on.

    Attributes:
        estimator: The pipeline that scales and then fits, so that predicting
            from the raw design matrix goes through the same scaling.
        intercept: The intercept on the original scale.
        coefficient: One coefficient per column of the design matrix, on the
            original scale.
    """

    estimator: Any
    intercept: float
    coefficient: np.ndarray


def scaled_estimator(engine: Any) -> Any:
    """The engine with the standardizing in front of it.

    Kept as one object so that the columns are centred and scaled by the training
    rows of whatever it is fitted to. Inside a fold that means the fold's own
    training rows, which is the point: scaling by the whole data set first would
    let each fold's held-out rows contribute to the numbers used to fit it.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), engine)


def scaled_fit(engine: Any, x: pd.DataFrame, y: np.ndarray, label: str) -> ScaledFit:
    """Fit a penalized model on standardized columns, and undo the standardizing.

    Both halves matter and they are separate decisions.

    Standardizing first is what ``glmnet`` does and it is not a preference: a
    penalty divides one budget between the terms, so the same predictor measured
    in millimetres rather than metres would be charged a thousandth as much for
    the same effect and would survive the penalty for no reason but its units.

    Undoing it afterwards is so the table can be read. A coefficient per standard
    deviation is a different quantity from the one
    :func:`~statassist.fit_linear_regression` reports, and the point of a shared
    coefficient contract is that two models' tables answer the same question.
    """
    pipeline = scaled_estimator(engine)
    with quiet_engine(label):
        pipeline.fit(np.asarray(x, dtype=float), y)

    scaler, fitted = pipeline[0], pipeline[-1]
    scaled = np.asarray(fitted.coef_, dtype=float).reshape(-1)
    coefficient = scaled / np.asarray(scaler.scale_, dtype=float)
    intercept = float(np.ravel(fitted.intercept_)[0]) - float(
        coefficient @ np.asarray(scaler.mean_, dtype=float)
    )
    return ScaledFit(estimator=pipeline, intercept=intercept, coefficient=coefficient)


def check_penalty(penalty: Any) -> str:
    """Resolve the penalty name, R's ``match.arg()``."""
    if penalty not in PENALTIES:
        raise SaValueError("`penalty` must be one of: " + ", ".join(PENALTIES) + ".")
    return str(penalty)


# --------------------------------------------------------------------------- #
# Predicting
# --------------------------------------------------------------------------- #


@dataclass
class EngineFit:
    """The engine object a model keeps so that it can predict.

    R keeps ``caret``'s ``train`` object, which carries the fitted model together
    with the data it saw. The same two things are kept here, plus what the engine
    itself does not record: whether the outcome was a class, and which class the
    positive label stood for.

    Attributes:
        estimator: The fitted engine object.
        x: The design matrix it was fitted to.
        y: The outcome as it saw it.
        classify: Whether the outcome was a class.
        outcome_lv: The two class labels, reference first, or ``None``.
    """

    estimator: Any
    x: pd.DataFrame
    y: np.ndarray
    classify: bool
    outcome_lv: list[str] | None = None


def predict_model(model: Any, newdata: Any = None, type: str = "raw") -> Any:  # noqa: A002
    """Predict from a fitted model on rows it was not fitted to.

    Port of ``predict.sa_model()``, which is the method to use rather than
    reaching for the estimator: it knows the names of the columns it was given
    and nothing about where they came from, so a factor predictor handed to it
    raw is dropped whole and a set of numeric predictors in another order is
    silently matched to the wrong coefficients. Here the terms are rebuilt from
    the levels the fit recorded and put in the model's own order by name.

    There is one prediction per row of ``newdata`` whatever the row holds, and a
    row with a missing value among the predictors gets a missing prediction. That
    is the rule the fit already follows in reverse: those are the rows
    ``design["n_dropped"]`` counted.
    """
    if type not in PREDICT_TYPES:
        raise SaValueError("`type` must be one of: " + ", ".join(PREDICT_TYPES) + ".")
    fit = model.fit
    if not isinstance(fit, EngineFit):
        raise SaInternalError("internal error: the model carries no engine object to predict with.")

    if newdata is None:
        # No rows to code: the rows the coefficients were estimated from are the
        # ones the engine already holds.
        return _engine_predict(fit, fit.x, type)

    frame = predict_frame(newdata, model["design"])
    usable = frame.notna().all(axis=1).to_numpy()
    if not usable.any():
        raise SaValueError(
            "no row of `newdata` is complete across the predictor(s) the model was "
            "fitted on, so there is nothing to predict from."
        )

    # The incomplete rows are held back rather than passed in and patched up
    # afterwards, so that the engine is asked only what it can answer and the
    # answer is put back where the row was.
    ready = design_matrix(
        frame.loc[usable],
        xlev=model["design"].get("predictor_lv"),
        want=model["engine"]["x_names"],
    )
    return _scatter(_engine_predict(fit, ready, type), usable)


def _engine_predict(fit: EngineFit, matrix: pd.DataFrame, type: str) -> Any:  # noqa: A002
    """Ask the engine for one kind of prediction, in the shape the contract fixes."""
    values = np.asarray(matrix, dtype=float)
    if not fit.classify:
        if type == "prob":
            raise SaValueError(
                '`type = "prob"` names one column per class, which only a classification has.'
            )
        return np.asarray(fit.estimator.predict(values), dtype=float)

    levels = list(fit.outcome_lv or [])
    if type == "raw":
        codes = np.asarray(fit.estimator.predict(values), dtype=int)
        return np.array([levels[code] for code in codes], dtype=object)

    probability = np.asarray(fit.estimator.predict_proba(values), dtype=float)
    # The engine's own column order is read by name rather than assumed, so that
    # `outcome_lv[1]` is the class reported whichever column it landed in.
    at = list(fit.estimator.classes_)
    if type == "response":
        return probability[:, at.index(1)]
    return pd.DataFrame(
        {level: probability[:, at.index(code)] for code, level in enumerate(levels)}
    )


def _scatter(value: Any, usable: np.ndarray) -> Any:
    """Put predictions back beside the rows they were asked about.

    The engine was given the complete rows only, so what comes back is shorter
    than ``newdata`` whenever a row had a missing predictor. It is scattered back
    to full length here, missing where the row could not be read.
    """
    if usable.all():
        return value
    if isinstance(value, pd.DataFrame):
        full = pd.DataFrame(
            math.nan, index=pd.RangeIndex(len(usable)), columns=value.columns, dtype=float
        )
        full.loc[usable] = value.to_numpy()
        return full
    scattered = (
        np.full(len(usable), None, dtype=object)
        if value.dtype == object
        else np.full(len(usable), math.nan)
    )
    scattered[usable] = value
    return scattered


# --------------------------------------------------------------------------- #
# Shared by the searches that fit a model at every step
# --------------------------------------------------------------------------- #

#: What the outcome column of a search frame is called.
#:
#: R uses ``caret``'s own name for it so that a predictor called ``y`` is a
#: predictor rather than a collision with the formula. The name is kept here for
#: the same reason.
SEARCH_OUTCOME = ".outcome"


def search_frame(x: pd.DataFrame, y: Any = None) -> pd.DataFrame:
    """The model frame a search fits on.

    Port of ``sa_search_frame()``. Both searches use it, so that the two build
    the same frame out of the same predictors.
    """
    frame = x.reset_index(drop=True).copy()
    if y is not None:
        frame[SEARCH_OUTCOME] = np.asarray(y)
    return frame


def search_label(model: str, classify: bool) -> str:
    """What to call the model inside a search, in a note and in ``engine["label"]``.

    Port of ``sa_search_label()``.
    """
    if model == "linear":
        return "Linear regression"
    if model == "logistic":
        return "Binomial logistic regression"
    if model == "rf":
        return "Random forest " + ("classification" if classify else "regression")
    raise SaInternalError(f"internal error: unhandled `model` {model}.")
