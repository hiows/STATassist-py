"""The penalized linear model, fitted at the penalty the resampling chose.

The port of ``R/fit_elastic_net.R``. R fits through ``glmnet``, which owns two
things this side has to arrange: the standardizing that makes a penalty a
statement about the predictors rather than about their units, and the path over
``lambda`` that a grid search would otherwise refit from scratch.

The first is done by :func:`~statassist.fit._shared.scaled_fit`, which
standardizes, fits and puts the estimates back. The second is not done at all: a
candidate is fitted per grid point per resample, which is slower and gives the
same answer.

One naming difference is worth stating plainly, since getting it backwards would
be silent. ``glmnet`` and this function call the mixing weight ``alpha`` and the
penalty size ``lambda``; the engine underneath calls the mixing weight
``l1_ratio`` and the penalty size ``alpha``. The R names are the ones the API
keeps, because they are the ones the statistical literature uses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.result import SaModel, new_model
from ._shared import (
    CV_METHODS,
    INTERCEPT,
    LOGIT_MAX_ITER,
    PENALTIES,
    EngineFit,
    Outcome,
    ScaledFit,
    check_cv_method,
    check_penalty,
    design_matrix,
    enet_grid,
    logistic_scores,
    model_design,
    model_frame,
    numeric_scores,
    resample_grid,
    resample_params,
    resolve_model_input,
    resolve_outcome,
    scaled_estimator,
    scaled_fit,
    train_control,
)

__all__ = ["fit_elastic_net"]

#: What this model is called where a message names it.
_MODEL = "an elastic net"

#: Terms a penalized model needs before there is a budget to divide.
_MIN_TERMS = 2


def fit_elastic_net(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    penalty: str = PENALTIES[0],
    alpha: Any = None,
    lambda_: Any = None,
    cv: bool = True,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    seed: int | None = None,
) -> SaModel:
    """Fit an elastic net, a lasso or a ridge.

    Fits a linear or logistic model whose coefficients are charged for their own
    size, and picks how much to charge them by cross-validation. What the penalty
    buys is a model that generalises: an unpenalized fit on many predictors will
    take whatever the noise offers, and shrinking every coefficient towards zero
    trades a little bias for less of that.

    ``penalty`` names a corner of one model rather than three models. The penalty
    is a mixture of the sum of the absolute coefficients and the sum of their
    squares, and ``alpha`` is the mixing weight, so a lasso is ``alpha=1`` and a
    ridge is ``alpha=0``. Which corner matters: only the lasso sets coefficients
    to exactly zero, so only a lasso or a mixture selects predictors. A ridge
    keeps every term and shrinks them all, which is what to use when the
    predictors are correlated and dropping one of a pair would be arbitrary.

    ``lambda_`` is the size of the penalty and is the argument worth resampling
    over. Its default is a wide path from almost nothing to a penalty that leaves
    only the intercept, so the search covers the whole range of models between
    the unpenalized fit and no fit at all.

    The coefficients are on the scale the predictors came in on, and there is a
    step behind that. A penalty divides one budget between the terms, so a
    predictor measured in millimetres rather than metres would be charged a
    thousandth as much for the same effect; the columns are therefore
    standardized before fitting and the estimates put back afterwards. This is
    ``glmnet``'s behaviour and it is what lets this table be read beside
    :func:`~statassist.fit_linear_regression`'s.

    There is no standard error, p-value or confidence interval, and that is a
    property of the model rather than an omission. A penalized estimate is
    deliberately biased towards zero, so the sampling distribution the usual
    interval describes is not the one it came from. ``selected`` is what the
    table carries instead: whether the term survived the penalty.

    A two-class outcome is fitted as a penalized logistic regression, with the
    direction rule the rest of the package follows: ``outcome_lv[0]`` is the
    reference and every coefficient and odds ratio describes ``outcome_lv[1]``.
    A numeric outcome taking two values is fitted as a regression with a note
    saying how to ask for the other reading.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The outcome, either the name of a column of ``data`` or a vector
            with one entry per row.
        predictors: Column names to fit on, or ``None`` for every column of
            ``data`` except the outcome. At least two model terms are needed:
            a penalty divides a budget between terms, and a single term has
            nothing to divide.
        outcome_lv: The two classes, reference first, to fit a classification.
        control_label: The reference class on its own, for when the other one
            needs no saying. Either argument asks for a classification.
        penalty: ``"elastic_net"`` for a mixture, ``"lasso"`` for the L1 corner
            or ``"ridge"`` for the L2 one.
        alpha: Mixing weights to search over, in ``[0, 1]``. Ignored by the two
            corners, which fix it, and validated anyway. ``None`` searches a
            grid of eleven from ridge to lasso.
        lambda_: Penalty sizes to search over. ``None`` searches a path of fifty
            from almost nothing to a penalty that leaves only the intercept.
        cv: Whether to cross-validate. ``False`` fits one model, so the grid must
            then hold one candidate: there is nothing to choose with.
        cv_method: Resampling scheme: ``"repeated_kfold"``, ``"kfold"`` or
            ``"loocv"``.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        seed: Seed for the fold assignment, or ``None`` to draw from the
            operating system's entropy.

    Returns:
        A :class:`~statassist.core.SaModel` whose ``analysis`` is
        ``"elastic_net"``.

        * ``coefficients`` - ``terms``, ``estimate`` and ``selected``, plus
          ``odds_ratio`` for a classification. No inference columns.
        * ``parameters`` - ``penalty`` and the ``alpha`` and ``lambda`` that were
          chosen rather than the grid that was asked for, with ``n_candidates``.
        * ``fit_stats`` - in-sample: ``r_squared``, ``rmse`` and ``mae`` for a
          regression, the deviances and ``mcfadden_r2`` for a classification,
          both with ``n_selected`` and ``n_zero``.
        * ``performance`` - one row per candidate, or ``None`` when ``cv=False``.
          The chosen row is the one matching ``parameters``.

    Raises:
        SaValueError: If the model has fewer than two terms, if the grid holds
            more than one candidate with ``cv=False``, or if an argument is
            unusable.

    Examples:
        A lasso at one penalty, with nothing to choose and so nothing to resample.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=150, n_pred=6, n_factor_pred=0,
        ...                           p_missing=0, seed=2)
        >>> fit = fit_elastic_net(**sim.args, penalty="lasso", lambda_=0.5, cv=False)
        >>> fit.parameters["penalty"], fit.parameters["alpha"]
        ('lasso', 1.0)
        >>> list(fit.coefficients)
        ['terms', 'estimate', 'selected']

        A larger penalty keeps fewer terms, which is the whole content of the L1
        corner.

        >>> light = fit_elastic_net(**sim.args, penalty="lasso", lambda_=0.05, cv=False)
        >>> heavy = fit_elastic_net(**sim.args, penalty="lasso", lambda_=2.0, cv=False)
        >>> heavy.fit_stats["n_selected"] < light.fit_stats["n_selected"]
        True

        A ridge sets nothing to zero, however hard it shrinks.

        >>> ridge = fit_elastic_net(**sim.args, penalty="ridge", lambda_=2.0, cv=False)
        >>> ridge.fit_stats["n_zero"]
        0
    """
    penalty = check_penalty(penalty)
    cv_method = check_cv_method(cv_method)
    weights = _ALPHA_PATH if alpha is None else alpha
    sizes = _LAMBDA_PATH if lambda_ is None else lambda_

    input_ = resolve_model_input(data, outcome, predictors)
    resolved = resolve_outcome(
        input_.y,
        outcome_lv,
        control_label,
        _MODEL,
        "which least squares has no residual for.",
    )
    matrix = design_matrix(input_.x)
    if len(matrix.columns) < _MIN_TERMS:
        raise SaValueError(
            "a penalty divides its budget between terms, so a model with one term has "
            f"nothing to divide, but this one has {len(matrix.columns)}: "
            + ", ".join(matrix.columns)
            + ". Add a predictor, or use fit_linear_regression() or "
            "fit_logistic_regression(), which fit one predictor as readily as ten."
        )

    grid = enet_grid(penalty, weights, sizes, cv)
    control = train_control(
        cv,
        cv_method,
        n_fold,
        n_repeat,
        input_.n_used,
        classify=resolved.classify,
        seed=seed,
    )
    label = _label(penalty, resolved.classify)
    engine = _builder(resolved.classify, input_.n_used)
    resampled = resample_grid(
        lambda params: scaled_estimator(engine(params)),
        matrix,
        resolved.y,
        grid,
        control,
        resolved.classify,
        label=label,
    )

    chosen = resampled.best
    fit = scaled_fit(engine(chosen), matrix, resolved.y, label)
    coefficients = _coef_table(fit, list(matrix.columns), resolved.classify)

    return new_model(
        analysis="elastic_net",
        terms=coefficients["terms"].tolist(),
        design=model_design(input_, resolved),
        # The penalty that ran, not the grid that was asked for: `performance`
        # holds every candidate, so recording the grid here as well would say the
        # same thing twice and leave two places for it to be wrong.
        parameters={
            "penalty": penalty,
            "alpha": float(chosen["alpha"]),
            "lambda": float(chosen["lambda_"]),
            "n_candidates": len(grid.index),
            **resample_params(cv, control, seed),
        },
        coefficients=coefficients,
        fit_stats=_fit_stats(fit, matrix, resolved, coefficients),
        performance=model_frame(resampled.results),
        resampling=model_frame(resampled.resampling),
        engine={
            "package": "scikit-learn",
            "method": "ElasticNet" if not resolved.classify else "LogisticRegression",
            "family": "binomial" if resolved.classify else "gaussian",
            "label": label,
            "metrics": resampled.metrics,
            "x_names": list(matrix.columns),
        },
        fit=EngineFit(
            estimator=fit.estimator,
            x=matrix,
            y=resolved.y,
            classify=resolved.classify,
            outcome_lv=resolved.levels,
        ),
    )


#: Mixing weights searched when the caller names none, ridge to lasso in tenths.
_ALPHA_PATH = np.round(np.linspace(0, 1, 11), 10)

#: Penalty sizes searched when the caller names none.
#:
#: Geometric rather than even, over five orders of magnitude, because that is how
#: a penalty acts: the difference between 0.001 and 0.002 is the difference
#: between two nearly unpenalized fits, while the difference between 1 and 2 is
#: the difference between two quite different models.
_LAMBDA_PATH = np.logspace(-4, 1, 50)


def _label(penalty: str, classify: bool) -> str:
    """What to call this corner of the model, in ``engine["label"]``."""
    corner = {
        "lasso": "Lasso (L1 penalty)",
        "ridge": "Ridge (L2 penalty)",
        "elastic_net": "Elastic net (L1 and L2 penalties)",
    }[penalty]
    return f"{corner} {'binomial classification' if classify else 'linear regression'}"


def _builder(classify: bool, n_used: int) -> Callable[[Mapping[str, Any]], Any]:
    """Make the function that turns one row of the grid into an unfitted engine.

    The two families take the penalty differently, and the classification one
    takes it inverted. Its objective is the total loss plus the penalty divided
    by ``C``, while ``lambda`` multiplies the penalty against the *mean* loss, so
    the two agree when ``C`` is one over ``n`` times ``lambda``. Getting this
    backwards would be silent: the model would fit, at a penalty nobody asked
    for.
    """

    def engine(params: Mapping[str, Any]) -> Any:
        weight = float(params["alpha"])
        size = float(params["lambda_"])
        if classify:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                l1_ratio=weight,
                C=math.inf if size == 0 else 1 / (n_used * size),
                solver="saga",
                max_iter=LOGIT_MAX_ITER,
            )

        from sklearn.linear_model import ElasticNet, LinearRegression

        # A penalty of zero is the unpenalized fit. The penalized solver reports
        # that as a condition of the call rather than fitting it, so the
        # unpenalized engine is asked instead and answers the same thing.
        if size == 0:
            return LinearRegression()
        return ElasticNet(alpha=size, l1_ratio=weight)

    return engine


def _coef_table(fit: ScaledFit, terms: list[str], classify: bool) -> pd.DataFrame:
    """The coefficient table of a penalized fit: what survived, and how much.

    The intercept is ``selected`` whatever its estimate, since it is not among
    the terms the penalty charges: a model has an intercept the way it has rows.
    """
    estimate = np.concatenate([[fit.intercept], fit.coefficient])
    table = pd.DataFrame(
        {
            "terms": [INTERCEPT] + terms,
            "estimate": estimate,
            "selected": (estimate != 0) | (np.arange(len(estimate)) == 0),
        }
    )
    if classify:
        table["odds_ratio"] = np.exp(table["estimate"])
    return table


def _fit_stats(
    fit: ScaledFit, matrix: pd.DataFrame, resolved: Outcome, coefficients: pd.DataFrame
) -> dict[str, float]:
    """How the chosen model did on the rows it was fitted to, and how sparse it is.

    In-sample throughout, and deliberately so: ``performance`` is where the
    held-out score lives. What these add is the count of terms the penalty kept,
    which is the one thing about a penalized fit that no metric reports.
    """
    penalized = coefficients.loc[coefficients["terms"] != INTERCEPT]
    n_selected = int(penalized["selected"].sum())
    sparsity = {
        "n_selected": n_selected,
        "n_zero": len(penalized.index) - n_selected,
    }

    values = np.asarray(matrix, dtype=float)
    if not resolved.classify:
        predicted = np.asarray(fit.estimator.predict(values), dtype=float)
        return {**numeric_scores(resolved.y, predicted), **sparsity}

    probability = np.asarray(fit.estimator.predict_proba(values), dtype=float)
    at = list(fit.estimator.classes_).index(1)
    deviance = logistic_scores(resolved.y, probability[:, at])
    null = deviance["null_deviance"]
    return {
        **deviance,
        "mcfadden_r2": 1 - deviance["residual_deviance"] / null if null > 0 else math.nan,
        **sparsity,
    }
