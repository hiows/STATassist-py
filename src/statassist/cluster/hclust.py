"""The clustering that keeps its working.

Port of ``R/cluster_hclust.R``. The other three return a labelling and nothing
that says how it was arrived at; this one returns the tree, and the tree is a
different kind of answer. It holds every cut the data admits, not just the one
that was asked for, so ``n_clust`` is a question put to a fitted object rather
than a parameter of the fit, and cutting ``res.fit`` again asks it without
recomputing anything.

It is also the only one of the four with a choice of distance. k-means minimises
squared Euclidean distance by construction and the two density methods measure a
radius in it, so for those three the distance is the method's and not the
caller's. Here it is genuinely open, and ``"correlation"`` is the reason the choice
is offered: it groups points by the shape of their profile rather than by how high
it sits, which on a feature scale is the difference between "these features move
together" and "these features sit at similar levels".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import SaValueError
from ..core.result import SaCluster, new_cluster
from ..core.validate import check_flag
from ..kernel.cluster import DIST_METHODS, cluster_dist
from ..reduce._shared import EMBEDDING_SCALES, check_embedding_scale
from ._shared import cluster_input, cluster_tables, resolve_n_clust

__all__ = ["HCLUST_METHODS", "cluster_hclust"]

#: The linkages on offer, in the order R lists them.
#:
#: What "close" means for two groups rather than two points. The default
#: ``"average"`` is the one that neither chains clusters into strings the way
#: single linkage does nor insists they come out round the way ``"ward.D2"`` does.
HCLUST_METHODS = ("average", "complete", "ward.D2")

#: What each linkage is called by the engine that computes it.
#:
#: R's ``"ward.D2"`` is SciPy's ``"ward"``: both apply the Lance-Williams update
#: to the distances as given, which is Ward's criterion as Ward stated it. R's
#: other spelling, ``"ward.D"``, applies it to unsquared distances and is not
#: offered by either.
_LINKAGE_METHODS = {"average": "average", "complete": "complete", "ward.D2": "ward"}


def cluster_hclust(
    data: Any,
    feats: Any = None,
    cluster_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
    n_clust: int = 2,
    dist_method: str = DIST_METHODS[0],
    hclust_method: str = HCLUST_METHODS[0],
) -> SaCluster:
    """Cluster by building a tree and cutting it.

    Merges the two closest points, then the two closest groups, and keeps going
    until everything is one group; then cuts the resulting tree so that
    ``n_clust`` groups fall out. What comes back is one cluster label per point,
    beside the linkage the labels were cut from.

    The input is the wide format the comparison functions take: **one row per
    sample and one column per feature**. Which margin of it becomes the thing
    being clustered is ``cluster_scale``, and ``design["point_type"]`` reports the
    answer.

    ``cluster_scale="samples"``, the default, puts one point per row of ``data``,
    and the question is which samples resemble each other. ``"features"`` puts one
    point per column, and the question is which features move together: the
    features are standardised first and the transpose is clustered as it stands.
    Transposing ``data`` by hand instead is a third analysis, since standardising
    then applies to samples, and the picture looks right while answering a
    different question.

    Rows that are not complete and finite across ``feats`` go before the distance
    is measured, and ``design["n_dropped"]`` reports how many. This is the listwise
    deletion the rest of the package uses; nothing is imputed. A feature that takes
    a single value cannot be scaled, so with ``scale=True`` it is left out with a
    message and named in ``design["dropped_feats"]``.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and
            one column per feature. The index is kept as the sample labels,
            repeated ones included.
        feats: Column names to cluster on, or ``None`` for every numeric column of
            ``data``. A non-numeric column is left out with a message, so a frame
            that carries a grouping column alongside the measurements can be
            passed as it is.
        cluster_scale: Which margin becomes the points being clustered, one of
            :data:`~statassist.reduce._shared.EMBEDDING_SCALES`.
        center: Whether to centre each feature first.
        scale: Whether to divide each feature by its standard deviation first.
            On by default because features are not measured on a common scale, and
            without it the feature with the widest units decides which points are
            close. Both flags always apply to the **columns of** ``data``, whatever
            ``cluster_scale`` is.
        n_clust: How many groups to cut the tree into. The tree itself does not
            depend on this, so ``res.fit`` can be cut again at another ``k``.
        dist_method: What "close" means, one of
            :data:`~statassist.kernel.cluster.DIST_METHODS`. ``"correlation"`` is
            ``1 - cor()``, which compares the shape of a profile rather than its
            level.
        hclust_method: Linkage, one of :data:`HCLUST_METHODS`.

    Returns:
        A :class:`~statassist.core.result.SaCluster` with ``analysis`` ``"hclust"``
        and ``design["n_noise"]`` always 0, since a tree places every point.
        ``res.fit`` is the SciPy linkage matrix, so another cut costs nothing.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, if fewer
            than two points or two features survive, or if the chosen distance is
            undefined for some pair of points.

    Examples:
        >>> from statassist import cluster_hclust, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=3)
        >>> res = cluster_hclust(sim.args["data"], n_clust=2)
        >>> res["analysis"], res["design"]["n_clusters"], res["design"]["n_noise"]
        ('hclust', 2, 0)
        >>> int(res["clusters"]["size"].sum()) == len(res["points"])  # all placed
        True
    """
    scale_name = check_embedding_scale(cluster_scale, "cluster_scale")
    if dist_method not in DIST_METHODS:
        raise SaValueError(
            "`dist_method` must be one of " + ", ".join(DIST_METHODS) + f". Got {dist_method}."
        )
    if hclust_method not in HCLUST_METHODS:
        raise SaValueError(
            "`hclust_method` must be one of "
            + ", ".join(HCLUST_METHODS)
            + f". Got {hclust_method}."
        )
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")

    input_ = cluster_input(data, feats, scale_name, center, scale, "cluster_hclust")
    n_clust = resolve_n_clust(n_clust, input_.m.shape[0], input_.point_type)

    d = cluster_dist(input_.m, dist_method)
    if not np.isfinite(d).all():
        # `draw_heatmap()` falls back to the input order here, because the picture
        # is still a picture without a tree. There is no such fallback when the
        # tree is the whole answer.
        raise SaValueError(
            f"`cluster_hclust()` cannot measure the {dist_method} distance between "
            f"every pair of {input_.point_type}s: some of them are undefined, which "
            f"happens when a {input_.point_type} has no variance across the other "
            "margin. A different `dist_method` is not governed by this."
        )

    from scipy.cluster.hierarchy import fcluster, linkage

    fit = linkage(d, method=_LINKAGE_METHODS[hclust_method])
    tables = cluster_tables(fcluster(fit, t=n_clust, criterion="maxclust"), input_.points, d)

    return new_cluster(
        analysis="hclust",
        points=input_.points,
        design=input_.design(tables.n_clusters, tables.n_noise),
        parameters={
            "cluster_scale": scale_name,
            "center": center,
            "scale": scale,
            "n_clust": n_clust,
            "dist_method": dist_method,
            "hclust_method": hclust_method,
        },
        assignments=tables.assignments,
        clusters=tables.clusters,
        engine={
            "package": "scipy",
            "method": "linkage",
            "label": "Hierarchical clustering",
            # R cuts `stats::hclust()` with `cutree()`, which returns exactly `k`
            # groups. SciPy's `maxclust` returns at most `k`, and returns fewer
            # when a tie means no height splits the tree into exactly that many.
            # `design["n_clusters"]` is what came back either way.
            "overridden": [],
        },
        fit=fit,
    )
