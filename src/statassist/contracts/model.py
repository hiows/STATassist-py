"""ModelResult (sa_model) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata

MODEL_COEF_COLUMNS = ("terms", "estimate")
MODEL_INFERENCE_COLUMNS = (
    "stderr",
    "statistic",
    "df",
    "pval",
    "lower_conf",
    "upper_conf",
)


def sa_new_model(
    *,
    analysis: str,
    terms: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    coefficients: pd.DataFrame,
    fit_stats: dict[str, Any],
    performance: pd.DataFrame | None = None,
    resampling: pd.DataFrame | None = None,
    engine: dict[str, Any],
    fit: Any,
) -> dict[str, Any]:
    if not terms:
        raise ValueError("internal error: `terms` must be a non-empty list.")
    if not isinstance(coefficients, pd.DataFrame):
        raise ValueError("internal error: `coefficients` must be a DataFrame.")
    if not coefficients["terms"].tolist() == list(terms):
        raise ValueError("internal error: `coefficients` is not aligned with `terms`.")
    absent = set(MODEL_COEF_COLUMNS) - set(coefficients.columns)
    if absent:
        raise ValueError(
            f"internal error: `coefficients` is missing contract column(s): "
            f"{', '.join(sorted(absent))}."
        )
    present = [c for c in MODEL_INFERENCE_COLUMNS if c in coefficients.columns]
    if present and len(present) < len(MODEL_INFERENCE_COLUMNS):
        raise ValueError(
            "internal error: `coefficients` carries some inference column(s) and "
            f"not others: {', '.join(present)}."
        )
    if not isinstance(fit_stats, dict) or not fit_stats:
        raise ValueError("internal error: `fit_stats` must be a non-empty dict.")
    for key in ("package", "method", "label", "metrics"):
        if engine.get(key) is None:
            raise ValueError(f"internal error: `engine` is missing `{key}`.")
    if performance is not None and not isinstance(performance, pd.DataFrame):
        raise ValueError("internal error: `performance` must be a DataFrame or None.")
    if resampling is not None and not isinstance(resampling, pd.DataFrame):
        raise ValueError("internal error: `resampling` must be a DataFrame or None.")

    return {
        "analysis": analysis,
        "terms": list(terms),
        "design": design,
        "parameters": parameters,
        "coefficients": coefficients.reset_index(drop=True),
        "fit_stats": fit_stats,
        "performance": performance,
        "resampling": resampling,
        "engine": engine,
        "fit": fit,
        "metadata": sa_metadata(),
        "__class__": ("sa_model", "sa_result"),
    }
