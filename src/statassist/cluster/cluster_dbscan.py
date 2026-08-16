"""DBSCAN density clustering."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from statassist.contracts.cluster import sa_new_cluster
from statassist.utils.cluster_utils import (
    sa_cluster_dist,
    sa_cluster_eps,
    sa_cluster_input,
    sa_cluster_tables,
    sa_dbscan_min_pts,
)
from statassist.utils.validate import sa_check_flag, sa_check_scalar_num


def cluster_dbscan(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    cluster_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    eps: float | None = None,
    min_pts: int | None = None,
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    if eps is not None:
        sa_check_scalar_num(eps, "eps", 0, lower_open=True)

    input_ = sa_cluster_input(data, feats, cluster_scale, center, scale, "cluster_dbscan")
    m = input_["m"]
    min_pts = sa_dbscan_min_pts(min_pts, m.shape[0], m.shape[1], input_["point_type"])

    eps_source = "derived" if eps is None else "supplied"
    if eps is None:
        eps = sa_cluster_eps(m, min_pts)
        warnings.warn(
            f"Using eps = {eps:.4g}, the radius reaching the {min_pts - 1}th neighbour of "
            f"95% of the {input_['point_type']}(s). Pass `eps` to set it.",
            stacklevel=2,
        )

    fit = DBSCAN(eps=eps, min_samples=min_pts, metric="euclidean")
    labels = fit.fit_predict(m)
    labels = np.where(labels < 0, 0, labels)

    d = sa_cluster_dist(m, "euclidean")
    tables = sa_cluster_tables(labels, input_["points"], d)

    if tables["n_clusters"] == 0:
        warnings.warn(
            f"No {input_['point_type']} has min_pts = {min_pts} neighbour(s) within eps = "
            f"{eps:.4g}, so every one of them is noise.",
            stacklevel=2,
        )

    return sa_new_cluster(
        analysis="dbscan",
        points=input_["points"],
        design={
            "point_type": input_["point_type"],
            "n_samples": input_["n_samples"],
            "n_used": input_["n_used"],
            "n_dropped": input_["n_dropped"],
            "n_feats": input_["n_feats"],
            "feats": input_["feats"],
            "dropped_feats": input_["dropped_feats"],
            "n_clusters": tables["n_clusters"],
            "n_noise": tables["n_noise"],
        },
        parameters={
            "cluster_scale": cluster_scale,
            "center": center,
            "scale": scale,
            "eps": eps,
            "eps_source": eps_source,
            "min_pts": min_pts,
            "dist_method": "euclidean",
        },
        assignments=tables["assignments"],
        clusters=tables["clusters"],
        engine={
            "package": "sklearn",
            "method": "dbscan",
            "label": "Density-based spatial clustering",
            "overridden": [],
        },
        fit=fit,
    )
