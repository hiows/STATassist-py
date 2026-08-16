"""UMAP embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.contracts.reduction import sa_new_reduction
from statassist.utils.reduce import (
    sa_embedding_frame,
    sa_reduce_embedding_matrix,
    sa_reduce_few_points,
    sa_reduce_input,
    sa_reduce_points,
    sa_umap_neighbors,
)
from statassist.utils.validate import sa_check_flag, sa_preserve_seed


def perform_umap(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    embedding_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    n_dim: int = 2,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    method: str = "umap",
    seed: int | None = None,
) -> dict:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "perform_umap() requires the optional `umap-learn` package. "
            "Install with: pip install 'statassist-py[reduce]'"
        ) from exc

    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    input_ = sa_reduce_input(data, feats, scale, "perform_umap")
    pt = sa_reduce_points(input_["x"], input_["samples"], embedding_scale)
    m = sa_reduce_embedding_matrix(input_["x"], embedding_scale, center, scale)
    n_nei = sa_umap_neighbors(n_neighbors, m.shape[0], pt["point_type"])
    sa_reduce_few_points(m.shape[0], pt["point_type"], f"n_neighbors = {n_nei}")

    with sa_preserve_seed(seed):
        fit = umap.UMAP(
            n_components=n_dim,
            n_neighbors=n_nei,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        coords = fit.fit_transform(m)

    return sa_new_reduction(
        analysis="umap",
        points=pt["points"],
        design={
            "point_type": pt["point_type"],
            "n_samples": input_["n_samples"],
            "n_used": input_["x"].shape[0],
            "n_dropped": input_["n_dropped"],
            "n_feats": input_["x"].shape[1],
            "feats": list(input_["feats"]),
            "dropped_feats": input_["dropped_feats"],
        },
        parameters={
            "embedding_scale": embedding_scale,
            "center": center,
            "scale": scale,
            "n_dim": n_dim,
            "n_neighbors": n_nei,
            "min_dist": min_dist,
            "metric": metric,
            "method": method,
            "seed": seed,
        },
        scores=sa_embedding_frame(coords, pt["points"], "UMAP"),
        engine={
            "package": "umap",
            "method": "umap",
            "label": "Uniform Manifold Approximation and Projection",
            "overridden": [],
        },
        fit=fit,
    )
