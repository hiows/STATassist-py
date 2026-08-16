"""Principal component analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from statassist.contracts.reduction import sa_new_reduction
from statassist.utils.reduce import sa_embedding_frame, sa_reduce_input, sa_reduce_points
from statassist.utils.validate import sa_check_flag


def perform_pca(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    embedding_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    if embedding_scale not in ("samples", "features"):
        raise ValueError('`embedding_scale` must be "samples" or "features".')

    input_ = sa_reduce_input(data, feats, scale, "perform_pca")
    x = input_["x"]
    pt = sa_reduce_points(x, input_["samples"], embedding_scale)

    fit = PCA(n_components=min(x.shape), svd_solver="full")
    if center:
        fit.mean_ = x.mean(axis=0)
        xs = x - fit.mean_
    else:
        xs = x.copy()
    if scale:
        sc = xs.std(axis=0, ddof=1)
        sc[sc == 0] = 1.0
        xs = xs / sc
    fit.fit(xs)

    if embedding_scale == "features":
        weights = fit.explained_variance_ ** 0.5 * np.sqrt(x.shape[0] - 1)
        coords = fit.components_.T * weights
        other = fit.transform(xs)
    else:
        coords = fit.transform(xs)
        other = fit.components_.T

    v = fit.explained_variance_
    total = v.sum()
    variance = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(v))],
            "sdev": np.sqrt(v),
            "prop_var": v / total * 100 if total > 0 else np.nan,
            "cum_var": np.cumsum(v) / total * 100 if total > 0 else np.nan,
        }
    )
    loadings_vars = input_["samples"] if embedding_scale == "features" else input_["feats"]
    loadings = pd.DataFrame({"variables": loadings_vars})
    for i in range(other.shape[1]):
        loadings[f"PC{i+1}"] = other[:, i]

    return sa_new_reduction(
        analysis="pca",
        points=pt["points"],
        design={
            "point_type": pt["point_type"],
            "n_samples": input_["n_samples"],
            "n_used": x.shape[0],
            "n_dropped": input_["n_dropped"],
            "n_feats": x.shape[1],
            "feats": list(input_["feats"]) if isinstance(input_["feats"], list) else input_["feats"],
            "dropped_feats": input_["dropped_feats"],
        },
        parameters={"embedding_scale": embedding_scale, "center": center, "scale": scale},
        variance=variance,
        loadings=loadings,
        scores=sa_embedding_frame(coords, pt["points"], "PC"),
        engine={
            "package": "sklearn",
            "method": "pca",
            "label": "Principal component analysis",
            "overridden": [],
        },
        fit=fit,
    )
