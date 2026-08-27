"""Binomial logistic regression, the classification counterpart of the linear fit.

The port of ``R/fit_logistic_regression.R``. The two are deliberately
near-identical below the documentation: the same input resolution, the same
resampling control, the same coefficient table. What is specific to this one is
the direction rule. A classification has a class of interest, and which of the
two levels that is decides the sign of every coefficient in the table.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ..core.errors import warn
from ..core.result import SaModel, new_model
from ..core.validate import check_scalar_num
from ._shared import (
    CV_METHODS,
    EngineFit,
    check_cv_method,
    design_lv,
    design_matrix,
    encode_outcome,
    inference_table,
    logistic_fit,
    logistic_scores,
    model_frame,
    no_grid,
    outcome_levels,
    resample_grid,
    resolve_model_input,
    train_control,
)

__all__ = ["fit_logistic_regression"]

#: What the engine and the label call this model.
_ENGINE = {
    "package": "scikit-learn",
    "method": "LogisticRegression",
    "family": "binomial",
    "label": "Binomial logistic regression",
}


def fit_logistic_regression(
    data: Any,
    outcome: Any,
    predictors: Any = None,
    outcome_lv: Any = None,
    control_label: Any = None,
    cv: bool = True,
    cv_method: str = CV_METHODS[0],
    n_fold: Any = 5,
    n_repeat: Any = 5,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> SaModel:
    """Fit a logistic regression.

    Fits a binomial logistic regression of one two-class outcome on a set of
    predictors, and scores it by cross-validation on the data it was fitted to.
    The coefficient table describes the fit on every usable row, while
    ``performance`` describes how the same procedure did on rows it had not seen
    inside each fold.

    ``outcome_lv`` fixes the direction, and it does so by the rule the rest of
    the package follows: the first level is the reference. Every coefficient is
    the change in the log odds of ``outcome_lv[1]`` per unit of its predictor,
    and ``odds_ratio`` is above 1 for a predictor that raises the chance of it.
    With ``outcome_lv=("control", "case")`` the table therefore reads as a
    statement about ``case``, and the same pair handed to
    :func:`~statassist.compare_two_groups` as ``group_lv`` would put ``control``
    in the denominator of its fold change, so the two point the same way.

    ``control_label`` states the same direction with one name instead of two.
    Naming both and pointing them at different classes is an error rather than a
    re-pointing: ``outcome_lv`` holds the two classes and nothing else, so there
    is no reading under which a different reference leaves any of it standing.

    The rule reaches the predictions too:
    ``model.predict(newdata, type="response")`` is the probability of
    ``outcome_lv[1]``, the same class the coefficients describe.

    A third class is an error rather than a silently dropped set of rows: two
    classes are what this model is, and quietly fitting a subset of the data that
    was passed in would answer a question nobody asked.

    The fit is unpenalized, which is what makes it a maximum likelihood
    regression and what makes its Wald standard errors mean what they say. The
    interval is the Wald interval, the one matching the z statistic and standard
    error reported beside it, rather than a profile likelihood interval.

    Perfect separation is reported rather than hidden. A predictor that splits the
    two classes exactly gives an estimate that grows until the solver stops, with
    a standard error to match, and the engine says so once per fold; those notes
    come back as one note with a count.

    Args:
        data: A frame in wide format, one row per observation. Typically the
            training half of a :func:`~statassist.split_data` result.
        outcome: The two-class outcome, either the name of a column of ``data`` or
            a vector with one entry per row. Categorical, string, logical and
            numeric columns are all read as class labels.
        predictors: Column names to fit on, or ``None`` for every column of
            ``data`` except the outcome.
        outcome_lv: The two classes, reference first, so that the coefficients
            describe the odds of the second one. ``None`` sorts the classes, which
            puts ``"control"`` before ``"treated"`` and ``0`` before ``1``.
        control_label: The reference class on its own, for when the other one
            needs no saying.
        cv: Whether to cross-validate. ``False`` fits the model once and reports
            no resampled performance.
        cv_method: Resampling scheme: ``"repeated_kfold"``, ``"kfold"`` or
            ``"loocv"``. The folds are stratified on the outcome, so a fold
            cannot come out holding one class only.
        n_fold: Folds per run, used by ``"repeated_kfold"`` and ``"kfold"``.
        n_repeat: Number of runs, used by ``"repeated_kfold"``.
        conf_level: Confidence level of the coefficient intervals.
        seed: Seed for the fold assignment, or ``None`` to draw from the
            operating system's entropy.

    Returns:
        A :class:`~statassist.core.SaModel` whose ``analysis`` is
        ``"logistic_regression"``, shaped as
        :func:`~statassist.fit_linear_regression`'s with these differences:

        * ``design`` also holds ``outcome_lv``, and ``n_events`` with
          ``event_rate``, the number and proportion of rows in
          ``outcome_lv[1]``.
        * ``coefficients`` has ``statistic`` as a Wald z rather than a t and
          ``df`` missing, since the z is not referred to any. ``odds_ratio``,
          ``or_lower_conf`` and ``or_upper_conf`` are added.
        * ``fit_stats`` holds ``null_deviance`` and ``residual_deviance`` with
          their degrees of freedom, ``mcfadden_r2``, the likelihood ratio test of
          the model against the intercept alone, ``aic`` and ``bic``.
        * ``performance`` holds resampled ``Accuracy`` and ``Kappa``.

    Raises:
        SaValueError: If the outcome does not hold two classes, if ``outcome_lv``
            and ``control_label`` disagree, or if an argument is unusable.
        SaWarning: If a term could not be estimated.

    Examples:
        The simulator plants the coefficients and the direction, so the fit has
        something to be scored against.

        >>> from statassist import simulate_classification
        >>> sim = simulate_classification(n_samples=160, n_pred=3,
        ...                               n_factor_pred=0, seed=1)
        >>> fit = fit_logistic_regression(**sim.args, cv=False)
        >>> fit.design["outcome_lv"]
        ['control', 'case']
        >>> fit.design["n_events"] + 0
        48
        >>> "odds_ratio" in fit.coefficients
        True

        The odds ratio of a planted positive coefficient is above 1, which is the
        whole content of the direction rule.

        >>> planted = sim.truth.loc[sim.truth["direction"] == "up", "predictors"]
        >>> rows = fit.coefficients["terms"].isin(planted)
        >>> bool((fit.coefficients.loc[rows, "odds_ratio"] > 1).all())
        True
    """
    cv_method = check_cv_method(cv_method)
    conf_level = check_scalar_num(
        conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True
    )

    input_ = resolve_model_input(data, outcome, predictors)
    levels = outcome_levels(input_.y, outcome_lv, control_label)
    y = encode_outcome(input_.y, levels)

    matrix = design_matrix(input_.x)
    control = train_control(
        cv, cv_method, n_fold, n_repeat, input_.n_used, classify=True, seed=seed
    )
    resampled = resample_grid(_build, matrix, y, no_grid(), control, classify=True)

    fit = logistic_fit(matrix, y, _ENGINE["label"])
    # No df: a Wald z is referred to the normal distribution, so reporting the
    # residual degrees of freedom next to it would suggest a t test.
    coefficients = inference_table(fit.terms, fit.estimate, fit.stderr, conf_level, df=None)
    coefficients["odds_ratio"] = np.exp(coefficients["estimate"])
    coefficients["or_lower_conf"] = np.exp(coefficients["lower_conf"])
    coefficients["or_upper_conf"] = np.exp(coefficients["upper_conf"])
    if fit.aliased:
        warn(
            "term(s) could not be estimated because the other predictors already span "
            "them; their rows are missing: " + ", ".join(fit.aliased) + "."
        )

    n_events = int(y.sum())
    return new_model(
        analysis="logistic_regression",
        terms=fit.terms,
        design={
            "outcome": input_.outcome,
            "outcome_type": "two classes",
            "outcome_lv": levels,
            "n_events": n_events,
            "event_rate": n_events / input_.n_used,
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
        fit_stats=_fit_stats(y, fit.fitted, fit.rank),
        performance=model_frame(resampled.results),
        resampling=model_frame(resampled.resampling),
        engine={**_ENGINE, "metrics": resampled.metrics, "x_names": list(matrix.columns)},
        fit=EngineFit(
            estimator=fit.estimator, x=matrix, y=y, classify=True, outcome_lv=levels
        ),
    )


def _build(params: Any) -> Any:
    """One candidate of the grid, which for this model is the only one."""
    from sklearn.linear_model import LogisticRegression

    from ._shared import LOGIT_MAX_ITER

    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=LOGIT_MAX_ITER)


def _fit_stats(y: np.ndarray, eta: np.ndarray, rank: int) -> dict[str, float]:
    """The deviances and the likelihood ratio test ``summary.glm()`` reports."""
    n = len(y)
    probability = 1 / (1 + np.exp(-eta))
    deviance = logistic_scores(y, probability)
    null = deviance["null_deviance"]
    residual = deviance["residual_deviance"]

    lr_stat = null - residual
    lr_df = rank - 1
    stats = {
        "null_deviance": null,
        "residual_deviance": residual,
        "df_null": float(n - 1),
        "df_residual": float(n - rank),
        "mcfadden_r2": 1 - residual / null if null > 0 else math.nan,
        "lr_stat": lr_stat,
        "lr_df": float(lr_df),
        "lr_pval": float(sp_stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else math.nan,
        # A generalised linear model's likelihood is its deviance, so the two
        # criteria are the deviance penalised by the parameter count rather than
        # anything that has to be computed again.
        "aic": residual + 2 * rank,
        "bic": residual + math.log(n) * rank,
    }
    return stats
