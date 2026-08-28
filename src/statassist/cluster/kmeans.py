"""The one clustering here that is not deterministic.

Port of ``R/cluster_kmeans.R``. k-means starts from a random set of centres and
improves them until they stop moving, which finds a local optimum and not the best
one; running it once and reporting the answer would make the result depend on the
state of the random number generator when the caller happened to call it.
``n_start = 25`` is the answer to that - twenty-five starts, best one kept - and it
is the engine's own default overridden, so it is declared in
``engine["overridden"]``.

The distance is not an argument. k-means does not minimise a distance the caller
chooses; it minimises the sum of squared Euclidean distances to the centres, and a
mean is only the centre of its group under that one. Offering ``"correlation"``
here would be offering an algorithm that does not exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import SaValueError
from ..core.result import SaCluster, new_cluster
from ..core.validate import check_count, check_flag
from ..kernel.cluster import cluster_dist
from ..reduce._shared import EMBEDDING_SCALES, check_embedding_scale
from ._shared import cluster_input, cluster_tables, resolve_n_clust

__all__ = ["KMEANS_N_START", "cluster_kmeans"]

#: How many random starts are tried, keeping the best.
#:
#: Twenty-five, which overrides the engine's own default. It makes a bad local
#: optimum unlikely rather than impossible, and it is declared as an override
#: because a caller comparing this with a bare engine call should be told why the
#: two disagree.
KMEANS_N_START = 25


def cluster_kmeans(
    data: Any,
    feats: Any = None,
    cluster_scale: str = EMBEDDING_SCALES[0],
    center: bool = True,
    scale: bool = True,
    n_clust: int = 2,
    n_start: int = KMEANS_N_START,
    iter_max: int = 100,
    seed: int | None = None,
) -> SaCluster:
    """Cluster by moving centres until they stop.

    Places ``n_clust`` centres, gives every point to its nearest one, moves each
    centre to the mean of the points it was given, and repeats until nothing moves.
    What comes back is one cluster label per point, beside the fitted estimator
    holding the centres.

    The input is the wide format the comparison functions take: **one row per
    sample and one column per feature**. Which margin of it becomes the thing
    being clustered is ``cluster_scale``, and ``design["point_type"]`` reports the
    answer.

    The starting centres are random, so two calls can disagree. Two defences are on
    by default and neither replaces the other. ``n_start`` runs the whole thing
    from that many different starts and keeps the one with the smallest
    within-cluster sum of squares. ``seed`` makes the run exactly repeatable, and
    it seeds a generator of this call's own, so seeding this call does not quietly
    reseed whatever the caller does next.

    Neither is a guarantee that ``n_clust`` is the right number. k-means always
    returns the number of groups it was asked for, and ``clusters["silhouette"]``
    is where to look for whether they are groups: a cluster near zero is one whose
    members are about as close to another cluster as to their own.

    Args:
        data: A DataFrame or a 2-d array in wide format, one row per sample and
            one column per feature.
        feats: Column names to cluster on, or ``None`` for every numeric column of
            ``data``.
        cluster_scale: Which margin becomes the points being clustered, one of
            :data:`~statassist.reduce._shared.EMBEDDING_SCALES`.
        center: Whether to centre each feature first.
        scale: Whether to divide each feature by its standard deviation first.
            Scaling matters more here than anywhere else in this family, since a
            mean is taken in the units the features arrived in. Both flags always
            apply to the **columns of** ``data``, whatever ``cluster_scale`` is.
        n_clust: How many centres to place. Unlike the two density methods, this
            is the number of clusters that will come back.
        n_start: How many random starts to try, keeping the best.
        iter_max: Most iterations one start may take before it is abandoned.
        seed: Seed for the starts, or ``None`` to draw from the operating system's
            entropy.

    Returns:
        A :class:`~statassist.core.result.SaCluster` with ``analysis`` ``"kmeans"``
        and ``design["n_noise"]`` always 0, since every point is given to a centre.
        ``res.fit`` is the fitted estimator, whose ``cluster_centers_`` are in the
        scaled units the clustering ran in.

    Raises:
        SaValueError: If an argument is not of the kind it has to be, if fewer than
            two points or two features survive, or if ``n_clust`` exceeds the number
            of distinct points.

    Examples:
        >>> from statassist import cluster_kmeans, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=3)
        >>> res = cluster_kmeans(sim.args["data"], n_clust=2, seed=1)
        >>> res["analysis"], res["design"]["n_clusters"], res["design"]["n_noise"]
        ('kmeans', 2, 0)
        >>> res["parameters"]["tot_withinss"] > 0
        True
    """
    scale_name = check_embedding_scale(cluster_scale, "cluster_scale")
    center = check_flag(center, "center")
    scale = check_flag(scale, "scale")
    n_start = check_count(n_start, "n_start", 1)
    iter_max = check_count(iter_max, "iter_max", 1)
    seed_used = None if seed is None else check_count(seed, "seed")

    input_ = cluster_input(data, feats, scale_name, center, scale, "cluster_kmeans")
    m = input_.m
    n_clust = resolve_n_clust(n_clust, m.shape[0], input_.point_type)

    # More centres than distinct points cannot be placed, since two centres would
    # have to start in the same place. Said here in terms of the argument the
    # caller passed rather than left to the engine's own wording.
    n_distinct = len(np.unique(m, axis=0))
    if n_clust > n_distinct:
        raise SaValueError(
            f"`n_clust` is {n_clust} but only {n_distinct} of the {m.shape[0]} "
            f"{input_.point_type}(s) being clustered are distinct. A centre cannot be "
            "placed where another one already is."
        )

    from sklearn.cluster import KMeans

    fit = KMeans(
        n_clusters=n_clust,
        n_init=n_start,
        max_iter=iter_max,
        random_state=seed_used,
    ).fit(m)

    d = cluster_dist(m, "euclidean")
    # The engine counts its clusters from 0 and noise is 0 here, so every label is
    # shifted into the 1-based space `cluster_tables()` renumbers within. k-means
    # has no noise, so nothing is left at 0.
    tables = cluster_tables(np.asarray(fit.labels_, dtype=int) + 1, input_.points, d)

    return new_cluster(
        analysis="kmeans",
        points=input_.points,
        design=input_.design(tables.n_clusters, tables.n_noise),
        parameters={
            "cluster_scale": scale_name,
            "center": center,
            "scale": scale,
            "n_clust": n_clust,
            "n_start": n_start,
            "iter_max": iter_max,
            "seed": seed_used,
            # Not an argument, but the silhouettes were measured on it and `repr`
            # says which distance it is quoting.
            "dist_method": "euclidean",
            "tot_withinss": float(fit.inertia_),
        },
        assignments=tables.assignments,
        clusters=tables.clusters,
        engine={
            "package": "sklearn",
            "method": "KMeans",
            "label": "k-means clustering",
            "overridden": [f"n_init = {n_start}"],
        },
        fit=fit,
    )
