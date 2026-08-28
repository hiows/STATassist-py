"""Ordinary least squares, fitted through the engine and scored by resampling.

The port of ``R/fit_linear_regression.R``. R fits through ``caret`` rather than
through ``lm()`` directly, and the reason carries over: the same resampling that
scores this model scores the elastic net and the random forest, so the numbers in
``performance`` are comparable across models that have nothing else in common.

What R gets from ``lm()`` and this side computes is the inference. The engine
solves the least squares problem and reports coefficients; the standard errors,
the t statistics and the interval beside them are the closed form on the design
matrix the engine was handed, which is the same quantity ``summary.lm()``
reports.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ..core.errors import SaValueError, warn
from ..core.result import SaModel, new_model
from ..core.validate import check_scalar_num
from ._shared import (
    CV_METHODS,
    EngineFit,
    check_cv_method,
    design_lv,
    design_matrix,
    inference_table,
    least_squares,
    model_frame,
    no_grid,
    resample_grid,
    resolve_model_input,
    train_control,
)

__all__ = ["fit_linear_regression"]

#: What the engine and the label call this model.
_ENGINE = {
    "package": "scikit-learn",
    "method": "LinearRegression",
    "label": "Ordinary least squares linear regression",
}


def fit_linear_regression(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    cv: bool = True,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> SaModel:
    """Fit a linear regression.

    Fits an ordinary least squares model of one continuous outcome on a set of
    predictors, and scores it by cross-validation on the data it was fitted to.
    Both halves of that sentence matter: the coefficient table describes the fit
    on every usable row, while ``performance`` describes how the same procedure
    did on rows it had not seen inside each fold.

    Nothing is selected on the outcome here. ``predictors`` is the set the caller
    names, and a predictor that turns out not to matter stays in the table with
    the p-value that says so.

    Two consequences of how the fitting is arranged are worth knowing.
    ``cv`` decides whether the model is scored, not how it is fitted: the final
    model is fitted on all usable rows either way, so the coefficients of
    ``cv=True`` and ``cv=False`` are identical and only ``performance`` and
    ``resampling`` differ. And terms are not predictors: a factor or string
    predictor with ``k`` levels becomes ``k - 1`` terms, named after the level
    each one stands for, so ``terms`` holds the row order of ``coefficients``
    while ``design["predictors"]`` holds the columns that were read.

    Rows with a missing value in the outcome or in any predictor are dropped
    before the folds are drawn rather than inside each fit. Left to the engine,
    deletion would happen once per fold on whatever that fold held, and the folds
    would then be scored on different subsets of the data;
    ``design["n_dropped"]`` reports how many rows went.

    The confidence interval is the t interval on the residual degrees of freedom,
    the one that matches the t statistic and standard error reported beside it. A
    term the fit could not estimate, because another predictor already spans it,
    keeps its row with its estimate and inference missing and is named in a
    warning.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The continuous outcome, either the name of a column of ``data``
            or a vector with one entry per row.
        predictors: Column names to fit on, or ``None`` for every column of
            ``data`` except the outcome. Numeric, logical, categorical and string
            columns are accepted; a column that takes a single value is left out
            with a note, since it cannot contribute.
        cv: Whether to cross-validate. ``False`` fits the model once and reports
            no resampled performance.
        cv_method: Resampling scheme: ``"repeated_kfold"`` for ``n_repeat`` runs
            of ``n_fold``-fold cross-validation, ``"kfold"`` for one, or
            ``"loocv"`` for leave-one-out.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        conf_level: Confidence level of the coefficient intervals.
        seed: Seed for the fold assignment, or ``None`` to draw from the
            operating system's entropy. Only the folds are drawn, so the
            coefficients do not depend on it.

    Returns:
        A :class:`~statassist.core.SaModel` whose ``analysis`` is
        ``"linear_regression"``.

        * ``coefficients`` - one row per term: ``estimate``, ``stderr``,
          ``statistic`` (t), ``df``, ``pval``, ``lower_conf`` and ``upper_conf``.
        * ``fit_stats`` - the model as a whole: ``r_squared``,
          ``adj_r_squared``, ``sigma``, the overall F test, ``aic`` and ``bic``.
        * ``performance`` - resampled ``RMSE``, ``Rsquared`` and ``MAE`` with
          their standard deviations across resamples, or ``None`` when
          ``cv=False``.
        * ``resampling`` - one row per resample, or ``None``.

    Raises:
        SaValueError: If the outcome is not numeric and finite, if an argument is
            unusable, or if no two rows are complete across the model.
        SaWarning: If a term could not be estimated because the other predictors
            already span it.

    Examples:
        The simulator plants the coefficients, so the fit has something to be
        scored against.

        >>> from statassist import simulate_regression
        >>> sim = simulate_regression(n_samples=120, n_pred=3, n_factor_pred=0,
        ...                           p_missing=0, seed=1)
        >>> fit = fit_linear_regression(**sim.args, cv=False)
        >>> fit.analysis
        'linear_regression'
        >>> fit.terms
        ['(Intercept)', 'x_1', 'x_2', 'x_3']
        >>> fit.performance is None
        True

        A categorical predictor becomes one term per level beyond the first.

        >>> import pandas as pd
        >>> frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...                       "g": ["lo", "hi", "lo", "hi", "lo", "hi"]})
        >>> fit_linear_regression(frame, outcome="y", cv=False).terms
        ['(Intercept)', 'glo']
    """
    cv_method = check_cv_method(cv_method)
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)

    input_ = resolve_model_input(data, outcome, predictors)
    y = input_.y
    if not np.issubdtype(y.to_numpy().dtype, np.number) or y.dtype == bool:
        raise SaValueError(
            f"`outcome` must be a numeric column for a linear regression, but is "
            f"{y.dtype}. Use fit_logistic_regression() for an outcome with two classes."
        )
    values = y.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SaValueError(
            "`outcome` holds non-finite value(s), which least squares has no residual for."
        )

    matrix = design_matrix(input_.x)
    control = train_control(cv, cv_method, n_fold, n_repeat, input_.n_used, seed=seed)
    resampled = resample_grid(
        _build, matrix, values, no_grid(), control, classify=False, label=_ENGINE["label"]
    )

    fit = least_squares(matrix, values, _ENGINE["label"])
    residual_df = input_.n_used - fit.rank
    coefficients = inference_table(fit.terms, fit.estimate, fit.stderr, conf_level, df=residual_df)
    if fit.aliased:
        warn(
            "term(s) could not be estimated because the other predictors already span "
            "them; their rows are missing: " + ", ".join(fit.aliased) + "."
        )

    return new_model(
        analysis="linear_regression",
        terms=fit.terms,
        design={
            "outcome": input_.outcome,
            "outcome_type": "continuous",
            "n_obs": input_.n_obs,
            "n_used": input_.n_used,
            "n_dropped": input_.n_dropped,
            "predictors": input_.predictors,
            "dropped_predictors": input_.dropped_predictors,
            **design_lv(input_.predictor_lv),
        },
        parameters={
            "cv": bool(cv),
            "cv_method": control.cv_method,
            "n_fold": control.n_fold,
            "n_repeat": control.n_repeat,
            "conf_level": conf_level,
            "seed": seed,
        },
        coefficients=coefficients,
        fit_stats=_fit_stats(values, fit.fitted, fit.rank),
        performance=model_frame(resampled.results),
        resampling=model_frame(resampled.resampling),
        engine={**_ENGINE, "metrics": resampled.metrics, "x_names": list(matrix.columns)},
        fit=EngineFit(estimator=fit.estimator, x=matrix, y=values, classify=False, outcome_lv=None),
    )


def _build(params: Any) -> Any:
    """One candidate of the grid, which for this model is the only one."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _fit_stats(y: np.ndarray, fitted: np.ndarray, rank: int) -> dict[str, float]:
    """The goodness-of-fit scalars ``summary.lm()`` reports.

    ``r_squared`` here is the share of the variance of the outcome the fit
    accounts for, which is not the same quantity as the ``Rsquared`` in
    ``performance``: that one is a held-out correlation, computed on rows the fit
    had not seen and with no intercept of its own.
    """
    n = len(y)
    residual_df = n - rank
    rss = float(np.sum((y - fitted) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1 - rss / tss if tss > 0 else math.nan
    stats: dict[str, float] = {
        "r_squared": r_squared,
        "adj_r_squared": math.nan,
        "sigma": math.nan,
        "f_stat": math.nan,
        "df1": math.nan,
        "df2": math.nan,
        "pval": math.nan,
        "aic": math.nan,
        "bic": math.nan,
    }
    if residual_df <= 0 or rss <= 0:
        return stats

    sigma_sq = rss / residual_df
    stats["adj_r_squared"] = 1 - (1 - r_squared) * (n - 1) / residual_df
    stats["sigma"] = math.sqrt(sigma_sq)
    # An intercept-only model has nothing to test against itself, which is the
    # `NULL` R's `summary()$fstatistic` returns there.
    if rank > 1 and tss > rss:
        f_stat = ((tss - rss) / (rank - 1)) / sigma_sq
        stats["f_stat"] = f_stat
        stats["df1"] = float(rank - 1)
        stats["df2"] = float(residual_df)
        stats["pval"] = float(sp_stats.f.sf(f_stat, rank - 1, residual_df))

    # The likelihood of a normal model at its own maximum, which is what R's
    # `logLik.lm()` reports; the parameter count is the coefficients plus sigma.
    log_lik = -n / 2 * (math.log(2 * math.pi) + math.log(rss / n) + 1)
    n_params = rank + 1
    stats["aic"] = -2 * log_lik + 2 * n_params
    stats["bic"] = -2 * log_lik + math.log(n) * n_params
    return stats
