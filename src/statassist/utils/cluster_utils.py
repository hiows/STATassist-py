"""Clustering utilities (from utils_cluster.R and kernel_cluster.R)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors

from statassist.utils.reduce import sa_reduce_embedding_matrix, sa_reduce_input, sa_reduce_points
from statassist.utils.validate import sa_check_count, sa_check_flag, sa_check_scalar_num


def sa_silhouette(d: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    cluster = cluster.astype(int)
    out = np.full(len(cluster), np.nan, dtype=float)
    assigned = cluster > 0
    ids = sorted(set(cluster[assigned].tolist()))
    if len(ids) < 2:
        return out

    if d.ndim == 1:
        m = squareform(d)
    else:
        m = d

    members = [(assigned & (cluster == g)) for g in ids]
    for i in np.where(assigned)[0]:
        own = ids.index(cluster[i])
        same = members[own].copy()
        same[i] = False
        if not np.any(same):
            out[i] = 0.0
            continue
        a = float(m[i, same].mean())
        b = min(float(m[i, g].mean()) for j, g in enumerate(members) if j != own)
        scale = max(a, b)
        out[i] = (b - a) / scale if scale > 0 else 0.0
    return out


def sa_cluster_tables(
    cluster: np.ndarray,
    points: list[str],
    d: np.ndarray,
) -> dict[str, Any]:
    cluster = cluster.astype(int)
    found = sorted(set(cluster[cluster > 0].tolist()))
    renumbered = np.zeros_like(cluster)
    if found:
        mapping = {old: new for new, old in enumerate(found, start=1)}
        renumbered[cluster > 0] = np.vectorize(mapping.get)(cluster[cluster > 0])

    sil = sa_silhouette(d, renumbered)
    assignments = pd.DataFrame(
        {"points": points, "cluster": renumbered.astype(int), "silhouette": sil}
    )
    ids = list(range(1, len(found) + 1))
    sizes = [int((renumbered == i).sum()) for i in ids]
    cluster_sil = []
    for i in ids:
        vals = sil[renumbered == i]
        cluster_sil.append(float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan)
    clusters = pd.DataFrame({"cluster": ids, "size": sizes, "silhouette": cluster_sil})
    n_noise = int((renumbered == 0).sum())
    return {
        "assignments": assignments,
        "clusters": clusters,
        "n_clusters": len(clusters),
        "n_noise": n_noise,
    }


def sa_cluster_dist(x: np.ndarray, dist_method: str) -> np.ndarray:
    if dist_method == "correlation":
        r = np.corrcoef(x)
        np.fill_diagonal(r, 1.0)
        with np.errstate(invalid="ignore"):
            d = 1.0 - r
        return squareform(d, checks=False)
    return pdist(x, metric=dist_method)


def sa_cluster_input(
    data: pd.DataFrame | np.ndarray,
    feats: list[str] | None,
    cluster_scale: str,
    center: bool,
    scale: bool,
    fn: str,
) -> dict[str, Any]:
    input_ = sa_reduce_input(data, feats, scale, fn)
    pt = sa_reduce_points(input_["x"], input_["samples"], cluster_scale)
    m = sa_reduce_embedding_matrix(input_["x"], cluster_scale, center, scale)
    rownames = pt["points"]
    return {
        "m": m,
        "points": rownames,
        "point_type": pt["point_type"],
        "n_samples": input_["n_samples"],
        "n_used": input_["x"].shape[0],
        "n_dropped": input_["n_dropped"],
        "n_feats": input_["x"].shape[1],
        "feats": input_.get("feats", [str(i) for i in range(input_["x"].shape[1])]),
        "dropped_feats": input_["dropped_feats"],
    }


def sa_cluster_n_clust(n_clust: int, n: int, point_type: str) -> int:
    n_clust = sa_check_count(n_clust, "n_clust", 2)
    if n_clust > n:
        raise ValueError(
            f"`n_clust` must not exceed the {n} usable {point_type}(s) being "
            f"clustered, but is {n_clust}."
        )
    return n_clust


def sa_dbscan_min_pts(
    min_pts: int | None,
    n: int,
    n_var: int,
    point_type: str,
) -> int:
    if min_pts is not None:
        min_pts = sa_check_count(min_pts, "min_pts", 2)
        if min_pts > n:
            raise ValueError(
                f"`min_pts` must not exceed the {n} usable {point_type}(s) being "
                f"clustered, but is {min_pts}."
            )
        return min_pts
    derived = min(n, max(4, min(int(n_var) + 1, n // 2)))
    warnings.warn(
        f"Using min_pts = {derived}, from the {n_var} variable(s) describing each "
        f"{point_type} and the {n} being clustered. Pass `min_pts` to set it.",
        stacklevel=2,
    )
    return derived


def sa_snn_params(
    k: int | None,
    eps: int | None,
    min_pts: int | None,
    n: int,
    point_type: str,
) -> dict[str, int]:
    if k is not None:
        k = sa_check_count(k, "k", 2)
    if eps is not None:
        eps = sa_check_count(eps, "eps", 1)
    if min_pts is not None:
        min_pts = sa_check_count(min_pts, "min_pts", 1)

    if k is None:
        k = min(n - 1, max(3, int(np.ceil(np.sqrt(n)))))
        warnings.warn(
            f"Using k = {k} neighbour(s), from the {n} {point_type}(s) being "
            "clustered. Pass `k` to set it.",
            stacklevel=2,
        )
    elif k > n - 1:
        raise ValueError(
            f"`k` must not exceed one less than the {n} usable {point_type}(s) "
            f"being clustered, which is {n - 1}, but is {k}."
        )

    if eps is None:
        eps = max(1, k // 2)
    elif eps > k:
        raise ValueError(
            f"`eps` must not exceed `k`, which is {k}, but is {eps}. Two points "
            "cannot share more neighbours than they each keep."
        )

    if min_pts is None:
        min_pts = max(2, k // 2)

    return {"k": k, "eps": eps, "min_pts": min_pts}


def sa_cluster_eps(m: np.ndarray, min_pts: int) -> float:
    nn = NearestNeighbors(n_neighbors=min_pts, metric="euclidean")
    nn.fit(m)
    dists, _ = nn.kneighbors(m)
    y = dists[:, -1]
    finite = y[np.isfinite(y)]
    eps = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    if not np.isfinite(eps) or eps <= 0:
        positive = finite[finite > 0]
        eps = float(positive.max()) if positive.size else 1.0
    return eps


def sa_snn_cluster(m: np.ndarray, k: int, eps: int, min_pts: int) -> np.ndarray:
    """Shared nearest neighbour clustering (dbscan sNNclust analogue)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(m)
    _, indices = nn.kneighbors(m)
    n = m.shape[0]
    neighbor_sets = [set(indices[i, 1:].tolist()) for i in range(n)]

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(neighbor_sets[i] & neighbor_sets[j])
            if shared >= eps:
                adj[i].add(j)
                adj[j].add(i)

    labels = np.zeros(n, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        if len(adj[i]) < min_pts:
            visited[i] = True
            continue
        cluster_id += 1
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            labels[node] = cluster_id
            if len(adj[node]) >= min_pts:
                for nb in adj[node]:
                    if not visited[nb]:
                        stack.append(nb)
    return labels
