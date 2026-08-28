"""The first of the two that are not told how many groups to find.

Port of ``R/cluster_dbscan.R``. It is also the first that is allowed to answer
"this point is not in one", and both of those follow from the same change of
question. :func:`~statassist.cluster_hclust` and
:func:`~statassist.cluster_kmeans` are asked to divide the points into
``n_clust`` parts, and a division has no room for a leftover; DBSCAN is asked
which regions are dense, and a point outside all of them is not a small cluster of
one, it is noise.

What it costs is that the two arguments it does take are harder to choose than a
cluster count. ``min_pts`` is a number of points and can be given a rule; ``eps``
is a radius in the units of the data and cannot, so a constant default would be
wrong on the next matrix. Rather than refuse to run without it, ``eps`` is derived
from the neighbour distances the matrix actually has and reported as derived in
``parameters["eps_source"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import notify
from ..core.result import SaCluster, new_cluster
from ..core.validate import check_flag, check_scalar_num, fmt_est
from ..kernel.cluster import cluster_dist
from ..reduce._shared import EMBEDDING_SCALES, check_embedding_scale
from ._shared import cluster_input, cluster_tables, dbscan_min_pts, derive_eps

__all__ = ["cluster_dbscan"]


def cluster_dbscan(
    data: Any,
    feats: Any = None,
    cluster_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
    eps: float | None = None,
    min_pts: int | None = None,
) -> SaCluster:
    """Cluster by finding the dense regions.

    Grows a cluster out of every point that has at least ``min_pts`` neighbours
    within ``eps`` of it, joining the neighbourhoods that overlap, and leaves the
    points that never fell into one as noise. How many clusters there are is the
    answer rather than the question, and it can be none.

    The input is the wide format the comparison functions take: **one row per
    sample and one column per feature**. Which margin of it becomes the thing
    being clustered is ``cluster_scale``, and ``design["point_type"]`` reports the
    answer.

    A point that never joined a neighbourhood gets cluster
    :data:`~statassist.kernel.cluster.NOISE_LABEL`. It has no silhouette, since
    noise is not a cluster to be near or far from, and ``clusters`` has no row for
    it; ``design["n_noise"]`` is the count. All points being noise is a possible
    answer and means the density asked for is not present, which is usually ``eps``
    being too small or ``min_pts`` too large for the data.

    Left as ``None``, ``eps`` is derived from the distances actually present: every
    point's distance to its ``min_pts - 1``th neighbour is collected, and ``eps`` is
    the 95th percentile of those. The rule therefore reads as "assume about one
    point in twenty is noise, and set the radius that leaves that many outside",
    which is a statement worth disagreeing with rather than a number off a curve.
    ``parameters["eps_source"]`` records whether the value was supplied or derived,
    and a derived one is said out loud when it is used.

    Note that this ``eps`` is a distance. :func:`~statassist.cluster_snn` has an
    argument of the same name that counts shared neighbours, because the algorithm
    it comes from named it that; the two are not comparable.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and one
            column per feature.
        feats: Column names to cluster on, or ``None`` for every numeric column of
            ``data``.
        cluster_scale: Which margin becomes the points being clustered, one of
            :data:`~statassist.reduce._shared.EMBEDDING_SCALES`.
        center: Whether to centre each feature first.
        scale: Whether to divide each feature by its standard deviation first.
            Scaling matters here because ``eps`` is one radius applied to all of
            them at once. Both flags always apply to the **columns of** ``data``.
        eps: The radius of a neighbourhood, in the units of the matrix being
            clustered, or ``None`` to derive it.
        min_pts: How many points must be within ``eps`` of a point, itself
            included, before it can be the core of a cluster. ``None`` derives it
            from the number of variables, capped at half the points.

    Returns:
        A :class:`~statassist.core.result.SaCluster` with ``analysis`` ``"dbscan"``,
        and ``design["n_clusters"]`` and ``design["n_noise"]`` both answers rather
        than arguments.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, or if fewer
            than two points or two features survive.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(1)
        >>> blobs = pd.DataFrame(
        ...     np.vstack([rng.normal(-3, 1, (30, 2)), rng.normal(3, 1, (30, 2))]),
        ...     columns=["x", "y"],
        ... )
        >>> res = cluster_dbscan(blobs, scale=False, eps=1.5, min_pts=4)
        >>> res["analysis"], res["design"]["n_clusters"]
        ('dbscan', 2)
    """
    scale_name = check_embedding_scale(cluster_scale, "cluster_scale")
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")
    # Before the input is read and before `min_pts` is derived, so that a call that
    # is going to fail does not first announce a default it never used.
    if eps is not None:
        eps = check_scalar_num(eps, "eps", 0, lower_open=True)

    input_ = cluster_input(data, feats, scale_name, center, scale, "cluster_dbscan")
    m = input_.m
    min_pts = dbscan_min_pts(min_pts, m.shape[0], m.shape[1], input_.point_type)

    eps_source = "supplied" if eps is not None else "derived"
    if eps is None:
        eps = derive_eps(m, min_pts)
        notify(
            f"Using eps = {fmt_est(eps)}, the radius reaching the {min_pts - 1}th "
            f"neighbour of 95% of the {input_.point_type}(s). Pass `eps` to set it."
        )

    from sklearn.cluster import DBSCAN

    fit = DBSCAN(eps=eps, min_samples=min_pts).fit(m)

    d = cluster_dist(m, "euclidean")
    # The engine numbers its clusters from 0 and calls noise -1, so one shift puts
    # noise at 0 and the clusters at 1 upwards, which is what this contract means.
    tables = cluster_tables(np.asarray(fit.labels_, dtype=int) + 1, input_.points, d)

    if tables.n_clusters == 0:
        notify(
            f"No {input_.point_type} has min_pts = {min_pts} neighbour(s) within eps = "
            f"{fmt_est(eps)}, so every one of them is noise. A larger `eps` or a "
            "smaller `min_pts` is what asks for less density."
        )

    return new_cluster(
        analysis="dbscan",
        points=input_.points,
        design=input_.design(tables.n_clusters, tables.n_noise),
        parameters={
            "cluster_scale": scale_name,
            "center": center,
            "scale": scale,
            "eps": float(eps),
            "eps_source": eps_source,
            "min_pts": min_pts,
            "dist_method": "euclidean",
        },
        assignments=tables.assignments,
        clusters=tables.clusters,
        engine={
            "package": "sklearn",
            "method": "DBSCAN",
            "label": "Density-based spatial clustering",
            "overridden": [],
        },
        fit=fit,
    )
