"""Scoring a set of fitted regressions on rows none of them was fitted on.

Port of ``R/evaluate_regression_models.R``. What ``fit_*`` already reports in
``performance`` is a resampled score, measured inside the folds of the data the
model was fitted to; this is the other kind, measured once on a held-out set the
caller drew. The two are different questions and the names are shared on purpose,
so ``rmse`` here and ``RMSE`` there read in the same unit.

There is no test in this module. A difference of held-out errors has no null this
function is in a position to state - the rows are not a sample from anything the
caller described - so what it reports is the two numbers and their difference,
and the reader supplies the judgement. The classification side does carry tests,
because the quantities there are functions of a class label and a probability and
have sampling distributions that do not depend on where the rows came from.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, warn
from ..core.result import SaPerformance, new_performance
from ._shared import (
    check_model_agreement,
    check_model_family,
    collect_predictions,
    evaluate_newdata,
    prediction_table,
    resolve_answer,
    resolve_models,
)

__all__ = ["evaluate_regression_models"]

#: What a model has to have been fitted to before it can be scored here.
_WANTED_FAMILY = "continuous"

#: Where a model of the other kind belongs.
_OTHER_FUNCTION = "evaluate_classification_models()"


def evaluate_regression_models(
    baseline_model: Any,
    new_models: Any = None,
    newdata: Any = None,
    answer: Any = None,
    baseline_label: str = "baseline",
) -> SaPerformance:
    """Score fitted regressions on held-out rows.

    Port of ``evaluate_regression_models()``. Predicts one or more fitted
    regressions on the same rows and reports how each one did, with the
    differences against a baseline where more than one model was passed. Every
    model is read through :meth:`~statassist.core.result.SaModel.predict`, so the
    fitting functions are interchangeable here and a linear model, a penalized
    one, a forest and a machine are scored by the same arithmetic.

    The rows are the intersection rather than the union. A prediction is missing
    for a row that is incomplete across *that model's* predictors, so a baseline
    fitted on nine columns and a reduced model fitted on four disagree about any
    row missing one of the extra five. Scoring each model on whatever it managed
    would put two numbers from two samples in one table and call their difference
    an improvement, so a row that any model cannot predict is left out of all of
    them, and a single note reports how many went and why.

    ``metrics`` reports ``cor`` and ``r_squared`` side by side because they
    answer different questions and agree only for predictions that need no
    calibration. ``r_squared`` is ``1 - SSE/SST`` on these rows, the fraction of
    the variance the predictions actually removed, and it is negative for a model
    that does worse than the mean of the outcome. ``cor**2`` is what that would
    be if the predictions were first rescaled by a line fitted to these same
    rows, so the gap between the two is what ``calib_slope`` and
    ``calib_intercept`` describe.

    Args:
        baseline_model: The reference model. It is the first row of every table
            and what ``comparisons`` subtracts.
        new_models: Mapping of name to further model to hold against it, such as
            ``{"selected": fit_2}``, or ``None`` to score the baseline on its
            own. The names are what the tables and the plot legend call the
            models, so an unnamed collection is refused rather than numbered.
        newdata: The rows to score, typically the test half of a
            :func:`~statassist.split_data` result. Columns no model was fitted on
            are ignored.
        answer: The observed outcome, either the name of a column of ``newdata``
            or a vector with one entry per row. ``None`` reads the column the
            models were fitted to, which is the usual case.
        baseline_label: What to call the baseline in the tables and the legend.

    Returns:
        A :class:`~statassist.core.result.SaPerformance` whose ``analysis`` is
        ``"regression_performance"``.

    Examples:
        >>> from statassist import fit_linear_regression, simulate_regression
        >>> sim = simulate_regression(n_samples=80, n_pred=3, seed=1)
        >>> frame = sim.args["data"]
        >>> train, test = frame.iloc[:60], frame.iloc[60:]
        >>> full = fit_linear_regression(train, outcome=sim.args["outcome"], cv=False)
        >>> res = evaluate_regression_models(full, newdata=test)
        >>> res["analysis"]
        'regression_performance'
        >>> int(res["metrics"]["n_used"].iloc[0]) == len(test.index)
        True
        >>> "comparisons" in res
        False
    """
    newdata = evaluate_newdata(newdata)
    models = resolve_models(baseline_model, new_models, baseline_label)
    check_model_family(models, _WANTED_FAMILY, _OTHER_FUNCTION)
    check_model_agreement(models)

    resolved = resolve_answer(answer, newdata, models[baseline_label])
    series = pd.Series(resolved.value).reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise SaValueError(
            "`answer` must be numeric: a regression is scored against the value it "
            f"predicted, not against a label. Got {series.dtype}. Use "
            f"{_OTHER_FUNCTION} for a two-class outcome."
        )
    observed = series.to_numpy(dtype=float)

    collected = collect_predictions(models, newdata, observed)
    observed = observed[collected.keep]
    predicted = collected.predicted

    # A property of the rows rather than of any one model, so it is measured once
    # and the caller is told once, rather than each model reporting its own NaN.
    var_observed = float(np.var(observed, ddof=1))
    if not math.isfinite(var_observed) or var_observed <= 0:
        warn(
            f"the observed outcome takes a single value over the {observed.size} scored "
            "row(s), so `cor`, `r_squared` and the calibration line are missing. "
            "`rmse`, `mae` and `bias` are still reported."
        )

    names = list(models)
    metrics = pd.DataFrame(
        [
            {"model": name, **_scores(observed, predicted[:, position], var_observed)}
            for position, name in enumerate(names)
        ]
    )
    metrics["n_used"] = metrics["n_used"].astype(int)

    # One warning for the whole run rather than one per model, the way a feature
    # that could not be tested is reported by the comparison functions. A model
    # whose calibration line is missing too is one the rows defeated, which the
    # warning above already covers.
    flat = [
        name
        for position, name in enumerate(names)
        if math.isnan(metrics["cor"].iloc[position])
        and not math.isnan(metrics["calib_slope"].iloc[position])
    ]
    if flat:
        warn(
            f"`cor` is missing for {len(flat)} model(s) that answered a single value "
            "over the scored rows: " + ", ".join(flat) + ". A prediction that does not "
            "vary ranks nothing."
        )

    comparisons = _comparisons(metrics, names)

    return new_performance(
        analysis="regression_performance",
        models=names,
        design={
            "outcome": resolved.label,
            "outcome_type": _WANTED_FAMILY,
            "baseline": baseline_label,
            "n_obs": collected.n_obs,
            "n_used": int(collected.keep.size),
            "n_dropped": collected.n_dropped,
        },
        parameters={},
        predictions=prediction_table(names, predicted, collected.keep, observed),
        metrics=metrics,
        comparisons=comparisons,
    )


def _scores(
    observed: np.ndarray,
    predicted: np.ndarray,
    var_observed: float,
) -> dict[str, Any]:
    """Per-model scores against the observed outcome.

    Port of ``sa_regression_scores()``. The calibration line is
    ``lm(predicted ~ observed)`` written out in closed form, which is the same two
    numbers and leaves no engine object to store.
    """
    n = int(observed.size)
    residual = predicted - observed
    sse = float(np.sum(residual**2))

    # An outcome that takes one value over the scored rows has nothing for a
    # proportion of variance or a calibration slope to be measured against.
    degenerate = not math.isfinite(var_observed) or var_observed <= 0
    if degenerate:
        correlation = math.nan
        r_squared = math.nan
        slope = math.nan
        intercept = math.nan
    else:
        var_predicted = float(np.var(predicted, ddof=1))
        # A model that answers the same value for every row ranks nothing, so it
        # has no correlation. Its calibration line is still defined and is flat,
        # which is the honest description of what it did.
        correlation = (
            float(np.corrcoef(observed, predicted)[0, 1]) if var_predicted > 0 else math.nan
        )
        r_squared = 1 - sse / (var_observed * (n - 1))
        slope = float(np.cov(observed, predicted, ddof=1)[0, 1]) / var_observed
        intercept = float(np.mean(predicted)) - slope * float(np.mean(observed))

    return {
        "n_used": n,
        "cor": correlation,
        "r_squared": r_squared,
        "rmse": math.sqrt(sse / n),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }


def _comparisons(metrics: pd.DataFrame, names: list[str]) -> pd.DataFrame | None:
    """Every score as ``new - baseline``, or ``None`` when nothing was compared.

    A positive ``delta_cor`` and a negative ``delta_rmse`` both say the new model
    did better, which is why the columns are not signed to a common direction:
    each one keeps the direction of the quantity it is a difference of.
    """
    if len(names) < 2:
        return None
    against = metrics.iloc[1:].reset_index(drop=True)
    baseline = metrics.iloc[0]
    return pd.DataFrame(
        {
            "model": against["model"],
            "delta_cor": against["cor"] - baseline["cor"],
            "delta_r_squared": against["r_squared"] - baseline["r_squared"],
            "delta_rmse": against["rmse"] - baseline["rmse"],
            "delta_mae": against["mae"] - baseline["mae"],
        }
    )
