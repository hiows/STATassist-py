"""PerformanceResult (sa_performance) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.contracts.base import _sa_result
from statassist.contracts.repr import repr_sa_performance
from statassist.utils.metadata import sa_metadata

PREDICTION_TABLE_COLUMNS = ("model", "row", "observed", "predicted")
REGRESSION_METRIC_COLUMNS = (
    "model",
    "n_used",
    "cor",
    "r_squared",
    "rmse",
    "mae",
    "bias",
    "calib_slope",
    "calib_intercept",
)
CLASSIFICATION_METRIC_COLUMNS = (
    "model",
    "n_used",
    "n_events",
    "auc",
    "auc_lower_conf",
    "auc_upper_conf",
    "brier",
    "accuracy",
    "sensitivity",
    "specificity",
)
REGRESSION_COMPARISON_COLUMNS = (
    "model",
    "delta_cor",
    "delta_r_squared",
    "delta_rmse",
    "delta_mae",
)
CLASSIFICATION_COMPARISON_COLUMNS = (
    "model",
    "delta_auc",
    "delta_auc_lower_conf",
    "delta_auc_upper_conf",
    "delta_auc_pval",
    "idi",
    "idi_lower_conf",
    "idi_upper_conf",
    "idi_pval",
    "nri",
    "nri_event",
    "nri_nonevent",
    "nri_lower_conf",
    "nri_upper_conf",
    "nri_pval",
)
ROC_CURVE_COLUMNS = ("model", "threshold", "sensitivity", "specificity")


def _check_columns(df: pd.DataFrame, what: str, want: tuple[str, ...]) -> None:
    absent = set(want) - set(df.columns)
    if absent:
        raise ValueError(
            f"internal error: {what} is missing contract column(s): "
            f"{', '.join(sorted(absent))}."
        )


def sa_new_performance(
    *,
    analysis: str,
    models: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame | None = None,
    curves: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not models or any(m is None or m == "" for m in models):
        raise ValueError("internal error: `models` must be a non-empty list.")
    if len(models) != len(set(models)):
        raise ValueError("internal error: `models` must be unique.")
    if analysis not in ("regression_performance", "classification_performance"):
        raise ValueError(f"internal error: unknown analysis `{analysis}`.")
    if design.get("baseline") != models[0]:
        raise ValueError("internal error: `design$baseline` is not the first of `models`.")

    metric_columns = (
        REGRESSION_METRIC_COLUMNS
        if analysis == "regression_performance"
        else CLASSIFICATION_METRIC_COLUMNS
    )
    comparison_columns = (
        REGRESSION_COMPARISON_COLUMNS
        if analysis == "regression_performance"
        else CLASSIFICATION_COMPARISON_COLUMNS
    )

    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("internal error: `metrics` must be a DataFrame.")
    _check_columns(metrics, "`metrics`", metric_columns)
    if metrics["model"].tolist() != list(models):
        raise ValueError("internal error: `metrics` is not aligned with `models`.")

    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("internal error: `predictions` must be a DataFrame.")
    _check_columns(predictions, "`predictions`", PREDICTION_TABLE_COLUMNS)
    if predictions["model"].drop_duplicates().tolist() != list(models):
        raise ValueError(
            "internal error: `predictions` does not hold every model once, in order."
        )

    if comparisons is not None:
        if not isinstance(comparisons, pd.DataFrame):
            raise ValueError("internal error: `comparisons` must be a DataFrame or None.")
        _check_columns(comparisons, "`comparisons`", comparison_columns)
        if comparisons["model"].tolist() != models[1:]:
            raise ValueError(
                "internal error: `comparisons` must hold every model other than "
                "the baseline, once and in order."
            )

    if curves is not None:
        if analysis == "regression_performance":
            raise ValueError("internal error: a regression evaluation has no ROC curve.")
        if not isinstance(curves, pd.DataFrame):
            raise ValueError("internal error: `curves` must be a DataFrame or None.")
        _check_columns(curves, "`curves`", ROC_CURVE_COLUMNS)
        unknown = set(curves["model"]) - set(models)
        if unknown:
            raise ValueError(
                f"internal error: `curves` holds model(s) absent from the evaluation: "
                f"{', '.join(sorted(unknown))}."
            )

    return _sa_result(
        {
            "analysis": analysis,
            "models": list(models),
            "design": design,
            "parameters": parameters,
            "predictions": predictions.reset_index(drop=True),
            "metrics": metrics.reset_index(drop=True),
            "comparisons": comparisons,
            "curves": curves,
            "metadata": sa_metadata(),
            "__class__": ("sa_performance", "sa_result"),
        },
        repr_sa_performance,
    )
