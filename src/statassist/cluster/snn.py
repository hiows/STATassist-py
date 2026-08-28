"""The second density method: closeness as overlap rather than as distance.

Port of ``R/cluster_snn.R``. Each point keeps its ``k`` nearest neighbours, two
points are linked when their neighbour lists overlap, and clusters are the dense
parts of that graph. A k nearest neighbour classifier is supervised - it labels a
new point by what its neighbours were labelled - and that is not what this is.

Which is worth having beside :func:`~statassist.cluster_dbscan` rather than
instead of it, because the two disagree about what "close" is in exactly the place
a radius breaks down. DBSCAN measures one radius everywhere, so it cannot find two
clusters of different densities at once: an ``eps`` large enough for the sparse one
swallows the dense one. Sharing neighbours is a relative measure - the neighbours
of a point in a sparse region are far away, but they are still its neighbours - so
this finds both. That is also its known weakness: in high dimensions everything's
neighbour lists start to overlap, and it will happily report structure that is only
the curse of dimensionality. Neither is right, and a grouping both of them find is
worth more than one only one of them does.

The graph and the walk over it are written out here, which is the one place in this
family that departs from wrapping an engine. There is no shared-nearest-neighbour
clustering in scikit-learn to wrap, and the algorithm is a neighbour query - which
*is* wrapped - followed by two set operations and a breadth-first search. Rebuilding
the neighbour search would be reimplementing an engine; counting overlaps between
its answers is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import notify
from ..core.result import SaCluster, new_cluster
from ..core.validate import check_flag
from ..kernel.cluster import NOISE_LABEL, cluster_dist
from ..reduce._shared import EMBEDDING_SCALES, check_embedding_scale
from ._shared import cluster_input, cluster_tables, snn_params

__all__ = ["SnnGraph", "cluster_snn"]


@dataclass(frozen=True)
class SnnGraph:
    """The graph a shared neighbour clustering was read off.

    What ``res.fit`` holds. There is no third-party fitted object here, so this
    carries the working instead: which points were dense enough to grow a cluster
    from, how many links each one had, and the labelling before this contract
    renumbered it.

    Attributes:
        k: How many nearest neighbours each point kept.
        eps: How many of them two points had to share to be linked.
        min_pts: How many links a point needed, itself counted, to be a core.
        neighbours: The ``k`` nearest neighbours of each point, by position.
        links: The surviving links of each point, by position.
        core: Whether each point was dense enough to grow a cluster from.
        cluster: The labelling as this module produced it, 0 for noise.
    """

    k: int
    eps: int
    min_pts: int
    neighbours: list[list[int]]
    links: list[list[int]]
    core: np.ndarray
    cluster: np.ndarray


def cluster_snn(
    data: Any,
    feats: Any = None,
    cluster_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
    k: int | None = None,
    eps: int | None = None,
    min_pts: int | None = None,
) -> SaCluster:
    """Cluster by how many neighbours points have in common.

    Builds a graph in which every point keeps its ``k`` nearest neighbours, links
    two points that share at least ``eps`` of them, and grows a cluster out of
    every point with at least ``min_pts`` such links. Points that never joined one
    are left as noise. How many clusters there are is the answer rather than the
    question.

    The input is the wide format the comparison functions take: **one row per
    sample and one column per feature**. Which margin of it becomes the thing
    being clustered is ``cluster_scale``, and ``design["point_type"]`` reports the
    answer.

    ``eps`` here counts neighbours, and in :func:`~statassist.cluster_dbscan` it is
    a distance. The two functions have an argument of the same name meaning two
    different things, which is the algorithms' doing rather than this package's.
    Here it is a count of shared neighbours, so it is a whole number between 1 and
    ``k``, and a value of ``k / 2`` says "half of what each of you considers close
    is the same points".

    One radius has to be right everywhere, so DBSCAN cannot find a tight cluster
    and a loose one in the same call. Shared neighbours are relative, so clusters
    of different densities come out together. The price is dimension: as the number
    of variables grows, distances concentrate and neighbour lists start to overlap
    for no reason, so this method will report structure that is an artefact of the
    width of the table. ``clusters["silhouette"]`` is the check worth making, and
    agreement with :func:`~statassist.cluster_dbscan` on the same points is worth
    more than either alone.

    A point that never joined a dense part of the graph gets cluster
    :data:`~statassist.kernel.cluster.NOISE_LABEL`, has no silhouette and no row in
    ``clusters``. Every point being noise is a possible answer and means the overlap
    asked for is not there, which is usually ``eps`` or ``min_pts`` being too large
    for the ``k`` in use.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and one
            column per feature.
        feats: Column names to cluster on, or ``None`` for every numeric column of
            ``data``.
        cluster_scale: Which margin becomes the points being clustered, one of
            :data:`~statassist.reduce._shared.EMBEDDING_SCALES`.
        center: Whether to centre each feature first.
        scale: Whether to divide each feature by its standard deviation first. Both
            flags always apply to the **columns of** ``data``.
        k: How many nearest neighbours each point keeps. ``None`` derives it as the
            square root of the number of points and reports what it came out as.
        eps: How many neighbours two points must **share** before they are linked,
            a whole number from 1 to ``k``. Not a distance. ``None`` uses ``k / 2``.
        min_pts: How many links a point needs in that graph, itself counted, before
            it can be the core of a cluster. ``None`` uses ``k / 2``.

    Returns:
        A :class:`~statassist.core.result.SaCluster` with ``analysis`` ``"snn"``,
        ``res.fit`` a :class:`SnnGraph`, and ``design["n_clusters"]`` and
        ``design["n_noise"]`` both answers rather than arguments.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, or if fewer
            than two points or two features survive.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(1)
        >>> blobs = pd.DataFrame(
        ...     np.vstack([rng.normal(-6, 0.4, (40, 2)), rng.normal(6, 2.5, (40, 2))]),
        ...     columns=["x", "y"],
        ... )
        >>> res = cluster_snn(blobs, scale=False, k=10)
        >>> res["analysis"], res["design"]["n_clusters"]
        ('snn', 2)
    """
    scale_name = check_embedding_scale(cluster_scale, "cluster_scale")
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")

    input_ = cluster_input(data, feats, scale_name, center, scale, "cluster_snn")
    m = input_.m
    par = snn_params(k, eps, min_pts, m.shape[0], input_.point_type)

    fit = _snn_graph(m, par["k"], par["eps"], par["min_pts"])

    d = cluster_dist(m, "euclidean")
    tables = cluster_tables(fit.cluster, input_.points, d)

    if tables.n_clusters == 0:
        notify(
            f"No {input_.point_type} shares eps = {par['eps']} of its k = {par['k']} "
            f"neighbour(s) with min_pts = {par['min_pts']} others, so every one of "
            "them is noise. A smaller `eps` or `min_pts`, or a larger `k`, is what "
            "asks for less overlap."
        )

    return new_cluster(
        analysis="snn",
        points=input_.points,
        design=input_.design(tables.n_clusters, tables.n_noise),
        parameters={
            "cluster_scale": scale_name,
            "center": center,
            "scale": scale,
            "k": par["k"],
            "eps": par["eps"],
            "min_pts": par["min_pts"],
            "dist_method": "euclidean",
        },
        assignments=tables.assignments,
        clusters=tables.clusters,
        engine={
            "package": "statassist",
            "method": "snn",
            "label": "Shared nearest neighbour clustering",
            # The neighbour search is scikit-learn's; the graph and the walk over
            # it are not, because there is no shared-neighbour clustering there to
            # call. So the labelling is not expected to match R's `dbscan`
            # point for point, and this says so rather than leaving a reader to
            # infer it from a disagreement.
            "overridden": ["graph and search written here; sklearn has no sNNclust"],
        },
        fit=fit,
    )


def _snn_graph(m: np.ndarray, k: int, eps: int, min_pts: int) -> SnnGraph:
    """Build the shared neighbour graph and walk it.

    Three steps, each of which is one sentence of the method. Every point keeps
    its ``k`` nearest neighbours; two points are linked when their lists overlap in
    at least ``eps`` places; and a cluster is a connected run of points that have
    ``min_pts`` links, plus whatever is linked to one of those.

    The link is symmetric even though the neighbour relation is not. "You are among
    my ten closest and I am not among yours" is a fact about how crowded your
    surroundings are, and the shared count - which is the same number read from
    either end - is what this method measures closeness by, so a one-sided
    neighbourhood is no reason to drop the pair.

    ``min_pts`` counts the point itself among its links, which is the convention
    :func:`~statassist.cluster_dbscan` documents for the same argument.
    """
    from sklearn.neighbors import NearestNeighbors

    n_points = m.shape[0]
    finder = NearestNeighbors(n_neighbors=min(k + 1, n_points))
    finder.fit(m)
    _, found = finder.kneighbors(m)
    # A point is not its own neighbour. It comes back as one because it is at
    # distance 0 from itself, and duplicated points make its position in the row
    # unpredictable, so it is removed by identity rather than by position.
    neighbours = [[int(j) for j in row if int(j) != index][:k] for index, row in enumerate(found)]

    adjacent = np.zeros((n_points, n_points), dtype=bool)
    for index, row in enumerate(neighbours):
        adjacent[index, row] = True
    counts = adjacent.astype(np.int32) @ adjacent.astype(np.int32).T
    linked = (adjacent | adjacent.T) & (counts >= eps)
    np.fill_diagonal(linked, False)

    links = [[int(j) for j in np.flatnonzero(linked[index])] for index in range(n_points)]
    core = np.array([len(row) + 1 >= min_pts for row in links], dtype=bool)

    cluster = np.full(n_points, NOISE_LABEL, dtype=int)
    label = 0
    for start in range(n_points):
        if not core[start] or cluster[start] != NOISE_LABEL:
            continue
        label += 1
        queue = [start]
        cluster[start] = label
        while queue:
            current = queue.pop()
            for neighbour in links[current]:
                if cluster[neighbour] != NOISE_LABEL:
                    continue
                cluster[neighbour] = label
                # A border point joins the cluster that reached it and stops
                # there. Letting it pass the walk on would join two clusters
                # through a point neither of them was dense enough to own.
                if core[neighbour]:
                    queue.append(neighbour)

    return SnnGraph(
        k=k,
        eps=eps,
        min_pts=min_pts,
        neighbours=neighbours,
        links=links,
        core=core,
        cluster=cluster,
    )
