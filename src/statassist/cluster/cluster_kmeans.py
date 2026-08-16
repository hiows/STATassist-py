"""k-means clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from statassist.contracts.cluster import sa_new_cluster
from statassist.utils.cluster_utils import (
    sa_cluster_dist,
    sa_cluster_input,
    sa_cluster_n_clust,
    sa_cluster_tables,
)
from statassist.utils.validate import sa_check_count, sa_check_flag, sa_preserve_seed


def cluster_kmeans(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    cluster_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    n_clust: int = 2,
    n_start: int = 25,
    iter_max: int = 100,
    seed: int | None = None,
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")
    n_start = sa_check_count(n_start, "n_start", 1)
    iter_max = sa_check_count(iter_max, "iter_max", 1)

    input_ = sa_cluster_input(data, feats, cluster_scale, center, scale, "cluster_kmeans")
    m = input_["m"]
    n_clust = sa_cluster_n_clust(n_clust, m.shape[0], input_["point_type"])

    n_distinct = len(np.unique(m, axis=0))
    if n_clust > n_distinct:
        raise ValueError(
            f"`n_clust` is {n_clust} but only {n_distinct} of the {m.shape[0]} "
            f"{input_['point_type']}(s) being clustered are distinct."
        )

    with sa_preserve_seed(seed):
        fit = KMeans(n_clusters=n_clust, n_init=n_start, max_iter=iter_max, random_state=seed)
        fit.fit(m)

    d = sa_cluster_dist(m, "euclidean")
    tables = sa_cluster_tables(fit.labels_ + 0, input_["points"], d)

    return sa_new_cluster(
        analysis="kmeans",
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
            "n_start": n_start,
            "iter_max": iter_max,
            "seed": seed,
            "dist_method": "euclidean",
            "tot_withinss": float(fit.inertia_),
        },
        assignments=tables["assignments"],
        clusters=tables["clusters"],
        engine={
            "package": "sklearn",
            "method": "kmeans",
            "label": "k-means clustering",
            "overridden": ["nstart = 25"],
        },
        fit=fit,
    )
