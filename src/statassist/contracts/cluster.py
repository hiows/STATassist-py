"""ClusterResult (sa_cluster) contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from statassist.contracts.base import _sa_result
from statassist.contracts.repr import repr_sa_cluster
from statassist.utils.metadata import sa_metadata

CLUSTER_ANALYSES = ("hclust", "kmeans", "dbscan", "snn")


def sa_new_cluster(
    *,
    analysis: str,
    points: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    assignments: pd.DataFrame,
    clusters: pd.DataFrame,
    engine: dict[str, Any],
    fit: Any,
) -> dict[str, Any]:
    if analysis not in CLUSTER_ANALYSES:
        raise ValueError(
            f"internal error: `analysis` must be one of {', '.join(CLUSTER_ANALYSES)}."
        )
    if not points:
        raise ValueError("internal error: `points` must be a non-empty list.")
    if design.get("point_type") not in ("sample", "feature"):
        raise ValueError('internal error: `design$point_type` must be "sample" or "feature".')
    if not isinstance(assignments, pd.DataFrame) or assignments["points"].tolist() != list(
        points
    ):
        raise ValueError("internal error: `assignments` is not aligned with `points`.")
    if assignments["cluster"].isna().any():
        raise ValueError(
            "internal error: `assignments$cluster` must be integer with no missing value."
        )
    if not isinstance(clusters, pd.DataFrame) or (clusters["cluster"] == 0).any():
        raise ValueError(
            "internal error: `clusters` must be a DataFrame of clusters found, "
            "which never includes noise."
        )
    found = sorted(set(assignments.loc[assignments["cluster"] > 0, "cluster"].astype(int)))
    if clusters["cluster"].tolist() != found:
        raise ValueError(
            f"internal error: `clusters` lists {clusters['cluster'].tolist()} "
            f"but the assignments hold {found}."
        )
    n_noise = int((assignments["cluster"] == 0).sum())
    if design.get("n_noise") != n_noise:
        raise ValueError(
            f"internal error: `design$n_noise` is {design.get('n_noise')} "
            f"but {n_noise} point(s) were left unassigned."
        )
    if design.get("n_clusters") != len(clusters):
        raise ValueError(
            f"internal error: `design$n_clusters` is {design.get('n_clusters')} "
            f"but `clusters` has {len(clusters)} row(s)."
        )
    for key in ("package", "method", "label", "overridden"):
        if engine.get(key) is None:
            raise ValueError(f"internal error: `engine` is missing `{key}`.")
    return _sa_result(
        {
            "analysis": analysis,
            "points": list(points),
            "design": design,
            "parameters": parameters,
            "assignments": assignments.reset_index(drop=True),
            "clusters": clusters.reset_index(drop=True),
            "engine": engine,
            "fit": fit,
            "metadata": sa_metadata(),
            "__class__": ("sa_cluster", "sa_result"),
        },
        repr_sa_cluster,
    )
