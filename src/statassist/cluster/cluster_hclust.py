"""Hierarchical clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from statassist.contracts.cluster import sa_new_cluster
from statassist.utils.cluster_utils import (
    sa_cluster_dist,
    sa_cluster_input,
    sa_cluster_n_clust,
    sa_cluster_tables,
)
from statassist.utils.validate import sa_check_flag


def cluster_hclust(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    cluster_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    n_clust: int = 2,
    dist_method: str = "euclidean",
    hclust_method: str = "average",
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    if dist_method not in ("euclidean", "manhattan", "correlation"):
        raise ValueError('`dist_method` must be "euclidean", "manhattan" or "correlation".')
    if hclust_method not in ("average", "complete", "ward"):
        hclust_method = {"ward.D2": "ward"}.get(hclust_method, hclust_method)

    input_ = sa_cluster_input(data, feats, cluster_scale, center, scale, "cluster_hclust")
    n_clust = sa_cluster_n_clust(n_clust, input_["m"].shape[0], input_["point_type"])

    d = sa_cluster_dist(input_["m"], dist_method)
    if not np.all(np.isfinite(d)):
        raise ValueError(
            f"`cluster_hclust()` cannot measure the {dist_method} distance between every pair."
        )

    method = "ward" if hclust_method == "ward" else hclust_method
    fit = linkage(d, method=method)
    labels = fcluster(fit, t=n_clust, criterion="maxclust")
    tables = sa_cluster_tables(labels, input_["points"], d)

    return sa_new_cluster(
        analysis="hclust",
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
            "n_clust": n_clust,
            "dist_method": dist_method,
            "hclust_method": hclust_method,
        },
        assignments=tables["assignments"],
        clusters=tables["clusters"],
        engine={
            "package": "scipy",
            "method": "hclust",
            "label": "Hierarchical clustering",
            "overridden": [],
        },
        fit=fit,
    )
