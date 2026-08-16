"""Shared nearest neighbour clustering."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from statassist.contracts.cluster import sa_new_cluster
from statassist.utils.cluster_utils import (
    sa_cluster_dist,
    sa_cluster_input,
    sa_cluster_tables,
    sa_snn_cluster,
    sa_snn_params,
)
from statassist.utils.validate import sa_check_flag


def cluster_snn(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None = None,
    *,
    cluster_scale: str = "samples",
    center: bool = True,
    scale: bool = True,
    k: int | None = None,
    eps: int | None = None,
    min_pts: int | None = None,
) -> dict:
    sa_check_flag(center, "center")
    sa_check_flag(scale, "scale")

    input_ = sa_cluster_input(data, feats, cluster_scale, center, scale, "cluster_snn")
    m = input_["m"]
    par = sa_snn_params(k, eps, min_pts, m.shape[0], input_["point_type"])

    labels = sa_snn_cluster(m, par["k"], par["eps"], par["min_pts"])

    d = sa_cluster_dist(m, "euclidean")
    tables = sa_cluster_tables(labels, input_["points"], d)

    if tables["n_clusters"] == 0:
        warnings.warn(
            f"No {input_['point_type']} shares eps = {par['eps']} of its k = {par['k']} "
            "neighbour(s), so every one of them is noise.",
            stacklevel=2,
        )

    return sa_new_cluster(
        analysis="snn",
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
            "k": par["k"],
            "eps": par["eps"],
            "min_pts": par["min_pts"],
            "dist_method": "euclidean",
        },
        assignments=tables["assignments"],
        clusters=tables["clusters"],
        engine={
            "package": "statassist",
            "method": "sNNclust",
            "label": "Shared nearest neighbour clustering",
            "overridden": [],
        },
        fit={"labels": labels, **par},
    )
