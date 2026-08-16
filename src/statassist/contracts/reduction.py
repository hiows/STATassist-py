"""ReductionResult (sa_reduction) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata

REDUCTION_ANALYSES = ("pca", "tsne", "umap")


def sa_new_reduction(
    *,
    analysis: str,
    points: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    scores: pd.DataFrame,
    variance: pd.DataFrame | None = None,
    loadings: pd.DataFrame | None = None,
    engine: dict[str, Any],
    fit: Any,
) -> dict[str, Any]:
    if analysis not in REDUCTION_ANALYSES:
        raise ValueError(
            f"internal error: `analysis` must be one of {', '.join(REDUCTION_ANALYSES)}."
        )
    if not points:
        raise ValueError("internal error: `points` must be a non-empty list.")
    if design.get("point_type") not in ("sample", "feature"):
        raise ValueError('internal error: `design$point_type` must be "sample" or "feature".')
    if not isinstance(scores, pd.DataFrame) or scores["points"].tolist() != list(points):
        raise ValueError("internal error: `scores` is not aligned with `points`.")
    for key in ("package", "method", "label", "overridden"):
        if engine.get(key) is None:
            raise ValueError(f"internal error: `engine` is missing `{key}`.")
    is_pca = analysis == "pca"
    if is_pca != (variance is not None) or is_pca != (loadings is not None):
        raise ValueError(
            "internal error: `variance` and `loadings` are present exactly when "
            "the analysis is a principal component analysis."
        )
    n_vars = design["n_used"] if design["point_type"] == "feature" else design["n_feats"]
    if loadings is not None and len(loadings) != n_vars:
        raise ValueError(
            f"internal error: `loadings` has {len(loadings)} row(s) for {n_vars} variable(s)."
        )

    result: dict[str, Any] = {
        "analysis": analysis,
        "points": list(points),
        "design": design,
        "parameters": parameters,
        "scores": scores.reset_index(drop=True),
        "engine": engine,
        "fit": fit,
        "metadata": sa_metadata(),
        "__class__": ("sa_reduction", "sa_result"),
    }
    if variance is not None:
        result["variance"] = variance.reset_index(drop=True)
    if loadings is not None:
        result["loadings"] = loadings.reset_index(drop=True)
    return result
