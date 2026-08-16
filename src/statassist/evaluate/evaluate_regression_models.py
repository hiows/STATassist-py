"""Held-out regression model evaluation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.contracts.performance import sa_new_performance
from statassist.utils.evaluate_utils import (
    sa_check_model_agreement,
    sa_check_model_family,
    sa_collect_predictions,
    sa_evaluate_newdata,
    sa_prediction_table,
    sa_resolve_answer,
    sa_resolve_models,
)
from statassist.utils.performance_kernel import sa_regression_scores


def evaluate_regression_models(
    baseline_model: dict[str, Any],
    new_models: dict[str, Any] | None = None,
    newdata: pd.DataFrame | None = None,
    *,
    answer: Any = None,
    baseline_label: str = "baseline",
) -> dict[str, Any]:
    if newdata is None:
        raise ValueError("`newdata` is required.")
    newdata = sa_evaluate_newdata(newdata)
    models = sa_resolve_models(baseline_model, new_models, baseline_label)
    sa_check_model_family(models, "continuous", "evaluate_classification_models()")
    sa_check_model_agreement(models)

    resolved = sa_resolve_answer(answer, newdata, baseline_model)
    observed = np.asarray(resolved["value"], dtype=float)
    if not np.issubdtype(observed.dtype, np.number):
        raise ValueError(
            "`answer` must be numeric for regression evaluation. "
            "Use evaluate_classification_models() for a two-class outcome."
        )

    collected = sa_collect_predictions(models, newdata, observed)
    observed = observed[collected["keep"] - 1]
    predicted = collected["predicted"]
    model_names = collected["model_names"]

    var_observed = float(np.var(observed, ddof=1))
    if not np.isfinite(var_observed) or var_observed <= 0:
        warnings.warn(
            f"the observed outcome takes a single value over the {len(observed)} scored "
            "row(s), so `cor`, `r_squared` and the calibration line are NA.",
            stacklevel=2,
        )

    scores = [
        sa_regression_scores(observed, predicted[:, i], var_observed)
        for i in range(len(model_names))
    ]
    metrics = pd.DataFrame({"model": model_names, **{k: [s[k] for s in scores] for k in scores[0]}})
    metrics["n_used"] = metrics["n_used"].astype(int)

    comparisons = None
    if len(model_names) > 1:
        comparisons = pd.DataFrame(
            {
                "model": model_names[1:],
                "delta_cor": metrics["cor"].iloc[1:].to_numpy() - metrics["cor"].iloc[0],
                "delta_r_squared": metrics["r_squared"].iloc[1:].to_numpy()
                - metrics["r_squared"].iloc[0],
                "delta_rmse": metrics["rmse"].iloc[1:].to_numpy() - metrics["rmse"].iloc[0],
                "delta_mae": metrics["mae"].iloc[1:].to_numpy() - metrics["mae"].iloc[0],
            }
        )

    return sa_new_performance(
        analysis="regression_performance",
        models=model_names,
        design={
            "outcome": resolved["label"],
            "outcome_type": "continuous",
            "baseline": baseline_label,
            "n_obs": collected["n_obs"],
            "n_used": len(collected["keep"]),
            "n_dropped": collected["n_dropped"],
        },
        parameters={},
        predictions=sa_prediction_table(predicted, collected["keep"], observed, model_names),
        metrics=metrics,
        comparisons=comparisons,
    )
