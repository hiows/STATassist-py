"""A support vector machine with a radial kernel.

The port of ``R/fit_svm.R``. The kernel is parameterised the same way on both
sides - both engines write the radial kernel as ``exp(-sigma * ||x - x'||^2)`` -
so ``sigma`` means the same number here as it does in R and needs no translating.

Two things do differ, and both are declared in ``engine["overridden"]``. R takes
its default ``sigma`` from a quantile of the pairwise distances; the engine here
has no such routine, so the default is the one it does offer: one over the
number of terms, on standardized columns. And a machine reports no probability
unless it is asked to fit one, so a classification asks, which fits the extra
calibration R's engine fits for the same reason.

The columns are centred and scaled either way. That is not a preference: the
kernel reads one distance over all the terms at once, so a term on a larger scale
contributes more to every distance and would be the only thing the machine could
see.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.result import SaModel, new_model
from ..core.validate import check_count, check_num_vector
from ._shared import (
    CV_METHODS,
    EngineFit,
    ModelInput,
    Outcome,
    check_cv_method,
    class_scores,
    design_matrix,
    importance_table,
    model_design,
    model_frame,
    numeric_scores,
    permutation_scores,
    quiet_engine,
    resample_grid,
    resample_params,
    resolve_model_input,
    resolve_outcome,
    scaled_estimator,
    svm_grid,
    train_control,
)

__all__ = ["fit_svm"]

#: What this model is called where a message names it.
_MODEL = "a support vector machine"

#: The only kernel offered, and the one both defaults are stated for.
#:
#: Radial rather than a choice, because it is the one that needs no decision from
#: the caller: it has one width and it can represent any smooth surface. A
#: polynomial kernel needs a degree, and a linear one is a model that
#: :func:`~statassist.fit_linear_regression` already fits with inference attached.
_KERNEL = "radial"

#: Costs searched when the caller names none, over five orders of magnitude.
#:
#: ``C`` is what the machine pays for letting a row sit on the wrong side of its
#: margin, so the path runs from a machine that would rather be smooth to one
#: that would rather be right, and the resampling decides which the data wants.
_C_PATH = 2.0 ** np.arange(-5, 11, 2)

#: What the port changed about the engine, for ``engine["overridden"]``.
_OVERRIDDEN_ALWAYS = ("columns centred and scaled",)

#: Declared on top of those when the outcome is a class.
_OVERRIDDEN_CLASSIFY = ("probability = True",)

#: Declared on top of those when the caller names no kernel width.
_OVERRIDDEN_SIGMA = ("sigma from the engine's scale heuristic",)


def fit_svm(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    C: Any = None,  # noqa: N803 - the argument is called this everywhere
    sigma: Any = None,
    n_permute: Any = 10,
    cv: bool = True,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    seed: int | None = None,
) -> SaModel:
    """Fit a support vector machine.

    Fits a surface that separates the classes, or tracks the outcome, with as
    much margin around it as the data allows. The radial kernel is what makes the
    surface curved: distances between rows are read through it, so a machine can
    follow a boundary that no straight line would.

    The two arguments are a trade-off against each other and the resampling is
    what settles them. ``C`` is what a row on the wrong side of the margin costs,
    so a large ``C`` bends the surface to accommodate individual rows and a small
    one keeps it smooth and accepts the mistakes. ``sigma`` is how quickly the
    kernel stops caring about distance, so a large ``sigma`` lets each row
    influence only its immediate neighbourhood - a surface that can follow
    anything, including the noise - and a small one makes the machine nearly
    linear.

    What a machine answers with is importance rather than coefficients, and there
    is no coefficient to be had: the surface lives in the space the kernel
    implies rather than in the terms. ``estimate`` is therefore permutation
    importance, measured by shuffling a term and seeing what the fit loses. It is
    measured on the rows the machine was fitted to, so it says which terms this
    fit leaned on rather than which terms would generalise.

    ``fit_stats`` is in-sample and includes ``support_vector_rate``, which is
    worth reading first. The support vectors are the rows that define the
    surface; a machine that needed most of its rows to describe its own boundary
    has found no simple boundary, whatever its accuracy on those rows says.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The outcome, either the name of a column of ``data`` or a vector
            with one entry per row.
        predictors: Column names to fit on, or ``None`` for every column of
            ``data`` except the outcome.
        outcome_lv: The two classes, reference first, to fit a classification.
        control_label: The reference class on its own. Either argument asks for a
            classification.
        C: Costs to search over. ``None`` searches eight values from ``2**-5`` to
            ``2**9``.
        sigma: Kernel widths to search over. ``None`` takes the engine's
            heuristic, one over the number of terms, which is the width that
            makes the kernel read a distance of the order the standardized
            columns supply.
        n_permute: Shuffles per term when measuring importance. One shuffle of
            one term is a draw rather than a measurement.
        cv: Whether to cross-validate. ``False`` fits one machine, so ``C`` and
            ``sigma`` must then name one value each.
        cv_method: Resampling scheme: ``"repeated_kfold"``, ``"kfold"`` or
            ``"loocv"``.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        seed: Seed for the probability calibration, the shuffling and the fold
            assignment.

    Returns:
        A :class:`~statassist.core.SaModel` whose ``analysis`` is ``"svm"``.

        * ``terms`` - the model terms, most important first.
        * ``coefficients`` - ``terms`` and ``estimate`` only. No intercept row
          and no second kind of importance: a machine has no impurity to report.
        * ``fit_stats`` - in-sample ``r_squared``, ``rmse`` and ``mae`` for a
          regression, ``accuracy``, ``error``, ``kappa``, ``sensitivity`` and
          ``specificity`` for a classification, both with
          ``n_support_vector`` and ``support_vector_rate``.
        * ``parameters`` - ``kernel``, the ``C`` and ``sigma`` that were chosen,
          ``n_candidates`` and ``n_permute``.
        * ``engine["overridden"]`` - what this port changed about the engine.

    Raises:
        SaValueError: If ``C`` or ``sigma`` holds zero, if ``cv=False`` is given
            more than one candidate, or if an argument is unusable.

    Examples:
        One machine at one cost, with nothing to choose and so nothing to
        resample.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=120, n_pred=4, n_factor_pred=0,
        ...                           p_missing=0, seed=8)
        >>> fit = fit_svm(**sim.args, C=1, cv=False, seed=1)
        >>> list(fit.coefficients)
        ['terms', 'estimate']
        >>> fit.parameters["kernel"], fit.parameters["C"]
        ('radial', 1.0)
        >>> 0 < fit.fit_stats["support_vector_rate"] <= 1
        True

        The width defaults to one over the number of terms.

        >>> fit.parameters["sigma"] == 1 / len(fit.terms)
        True
    """
    cv_method = check_cv_method(cv_method)
    shuffles = check_count(n_permute, "n_permute", 1)

    input_ = resolve_model_input(data, outcome, predictors)
    resolved = resolve_outcome(
        input_.y,
        outcome_lv,
        control_label,
        _MODEL,
        "which the loss of a support vector regression has no residual for.",
    )
    matrix = design_matrix(input_.x)

    costs = _C_PATH if C is None else check_num_vector(C, "C", 0)
    widths = _sigma(sigma, len(matrix.columns))
    grid = svm_grid(costs, widths, cv)
    control = train_control(
        cv, cv_method, n_fold, n_repeat, input_.n_used, classify=resolved.classify, seed=seed
    )
    label = (
        "Support vector machine "
        + ("classification" if resolved.classify else "regression")
        + " (radial basis kernel)"
    )

    def build(params: Mapping[str, Any]) -> Any:
        return scaled_estimator(
            _engine(float(params["C"]), float(params["sigma"]), resolved.classify, seed)
        )

    resampled = resample_grid(
        build, matrix, resolved.y, grid, control, resolved.classify, label=label
    )

    chosen = resampled.best
    machine = build(chosen)
    with quiet_engine(label):
        machine.fit(np.asarray(matrix, dtype=float), resolved.y)

    permuted = permutation_scores(machine, matrix, resolved.y, resolved.classify, shuffles, seed)
    coefficients = importance_table(list(matrix.columns), permuted)

    overridden = list(_OVERRIDDEN_ALWAYS)
    if resolved.classify:
        overridden += list(_OVERRIDDEN_CLASSIFY)
    if sigma is None:
        overridden += list(_OVERRIDDEN_SIGMA)

    return new_model(
        analysis="svm",
        terms=coefficients["terms"].tolist(),
        design=model_design(input_, resolved),
        # The pair that ran, not the grids that were asked for: `performance`
        # holds every candidate, so recording the grids here as well would say the
        # same thing twice and leave two places for it to be wrong.
        parameters={
            "kernel": _KERNEL,
            "C": float(chosen["C"]),
            "sigma": float(chosen["sigma"]),
            "n_candidates": len(grid.index),
            "n_permute": shuffles,
            **resample_params(cv, control, seed),
        },
        coefficients=coefficients,
        fit_stats=_fit_stats(machine, matrix, input_, resolved),
        performance=model_frame(resampled.results),
        resampling=model_frame(resampled.resampling),
        engine={
            "package": "scikit-learn",
            "method": "SVC" if resolved.classify else "SVR",
            "kernel": _KERNEL,
            "label": label,
            "metrics": resampled.metrics,
            "x_names": list(matrix.columns),
            "overridden": overridden,
        },
        fit=EngineFit(
            estimator=machine,
            x=matrix,
            y=resolved.y,
            classify=resolved.classify,
            outcome_lv=resolved.levels,
        ),
    )


def _sigma(sigma: Any, n_terms: int) -> np.ndarray:
    """The kernel widths to search, resolved to numbers before the grid is built.

    The default is resolved here rather than left to the engine, which would
    accept the word ``"scale"`` and never say what it meant by it. ``parameters``
    has to report the width the machine ran at, so the width has to be a number
    before the machine is asked for.

    One over the term count is that heuristic on standardized columns: each term
    contributes about one to the squared distance between two rows, so the
    exponent of the kernel is of order one however many terms there are.
    """
    if sigma is None:
        return np.array([1 / n_terms], dtype=float)
    return check_num_vector(sigma, "sigma", 0)


def _engine(cost: float, width: float, classify: bool, seed: int | None) -> Any:
    """An unfitted machine at one cost and one kernel width.

    ``probability`` is on for a classification because a class label is not
    enough: ``predict(type="response")`` is a probability, and
    :func:`~statassist.evaluate_classification_models` scores probabilities
    rather than labels. The engine fits it by calibrating on held-out folds of
    the training rows, which is why the seed reaches this.
    """
    from sklearn.svm import SVC, SVR

    if classify:
        return SVC(C=cost, gamma=width, kernel="rbf", probability=True, random_state=seed)
    return SVR(C=cost, gamma=width, kernel="rbf")


def _fit_stats(
    machine: Any, matrix: pd.DataFrame, input_: ModelInput, resolved: Outcome
) -> dict[str, float]:
    """How the machine did on the rows it was fitted to, and how many it needed.

    In-sample throughout, and the support vector count is what keeps that
    readable: a machine that used most of its rows to define its own boundary has
    not found a boundary, and no in-sample score will say so.
    """
    values = np.asarray(matrix, dtype=float)
    if resolved.classify:
        predicted = np.asarray(machine.predict(values), dtype=int)
        scored = class_scores(resolved.y, predicted)
    else:
        predicted = np.asarray(machine.predict(values), dtype=float)
        scored = numeric_scores(resolved.y, predicted)

    n_support = len(np.asarray(machine[-1].support_))
    return {
        **scored,
        "n_support_vector": float(n_support),
        "support_vector_rate": n_support / input_.n_used,
    }
