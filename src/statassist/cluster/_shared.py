"""Everything the four clustering functions share: the input, and the distance.

Port of ``R/utils_cluster.R`` and of ``sa_cluster_tables()``. The four differ in
what they do with a matrix of points and not at all in how they read one, so the
same rows are dropped for the same reasons in all four and ``design`` describes
the input the same way. That is what makes them comparable, and comparing them is
most of the point of having four: a group that only k-means finds, having been
told to find two things, is a different fact from one that DBSCAN found without
being told how many to look for.

The input is read through :mod:`statassist.reduce._shared` rather than through a
copy of it, so a clustering and a reduction of the same frame are about the same
rows and the assignment can be painted straight onto the scores.

``cluster_scale`` is the ``embedding_scale`` of the reductions under the name that
fits here, and it means the same thing: which margin of the input becomes a point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.validate import check_count
from ..kernel.cluster import NOISE_LABEL, silhouette
from ..reduce._shared import (
    ReduceInput,
    embedding_matrix,
    reduce_input,
    reduce_points,
)

__all__ = [
    "ClusterInput",
    "ClusterTables",
    "MIN_CLUSTERS",
    "cluster_input",
    "cluster_tables",
    "dbscan_min_pts",
    "derive_eps",
    "resolve_n_clust",
    "snn_params",
]

#: Fewest clusters a partitioning method will be asked for.
#:
#: One cluster is not a clustering: every point is in it, and the silhouette that
#: would say how well they fit is undefined because there is no other cluster to
#: be far from.
MIN_CLUSTERS = 2

#: The DBSCAN density floor, and where it comes from.
#:
#: The textbook floor is one more than the number of dimensions: a group in ``d``
#: dimensions needs ``d + 1`` points to be more than a flat piece of one. That rule
#: is unusable on the shape of table this package is usually given, where there are
#: more features than samples and ``d + 1`` is larger than the whole data set, so it
#: is capped at half the points - a threshold above half can only ever return a
#: single cluster, since two groups that size do not fit.
#:
#: It is floored at 4 rather than at ``d + 1`` because ``d + 1`` is 3 on a
#: two-dimensional table, and 3 fragments: Ester et al. (1996), who introduced the
#: method, settled on 4 for two dimensions and found nothing bought by going higher.
_DBSCAN_MIN_PTS_FLOOR = 4

#: Which quantile of the k-distance curve the derived ``eps`` is read off.
#:
#: Not the knee. Reading the knee arithmetically, as the point of the sorted curve
#: furthest below the chord joining its ends, is the textbook translation of the
#: manual procedure and it does not survive contact with a curve that rises
#: gradually. A percentile has no such failure mode and says something the knee
#: never did: ``eps`` at the 95th percentile is the radius that reaches the
#: ``min_pts - 1``th neighbour of all but one point in twenty, so the rule reads as
#: "assume about 5% of the points are noise and set the radius accordingly", which
#: is a statement a caller can disagree with rather than a number off a curve.
_EPS_QUANTILE = 0.95

#: The radius a curve that never leaves zero falls back to.
#:
#: Duplicated points give exactly that, and a zero radius would make every point
#: noise. Used only when no positive distance exists at all.
_EPS_FALLBACK = 1.0

#: The smallest neighbourhood shared nearest neighbour clustering keeps.
#:
#: Three neighbours, floored: below that the shared count is 0 or 1 for nearly
#: every pair and the graph has no structure left to find components in.
_SNN_K_FLOOR = 3


@dataclass(frozen=True)
class ClusterInput:
    """The point-by-variable matrix an engine is handed, and what it came from.

    ``design`` describes the input, so its counts do not turn with
    ``cluster_scale``: ``n_samples`` is always rows of ``data`` and ``feats``
    always the columns kept. ``point_type`` is what says which of the two became
    the points.
    """

    m: np.ndarray
    points: list[str]
    point_type: str
    n_samples: int
    n_used: int
    n_dropped: int
    n_feats: int
    feats: list[str]
    dropped_feats: list[str]

    def design(self, n_clusters: int, n_noise: int) -> dict[str, Any]:
        """The ``design`` slot, once the answer is known."""
        return {
            "point_type": self.point_type,
            "n_samples": self.n_samples,
            "n_used": self.n_used,
            "n_dropped": self.n_dropped,
            "n_feats": self.n_feats,
            "feats": list(self.feats),
            "dropped_feats": list(self.dropped_feats),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }


def cluster_input(
    data: Any,
    feats: Any,
    cluster_scale: str,
    center: bool,
    scale: bool,
    fn: str,
) -> ClusterInput:
    """Read the matrix a clustering is computed on out of the caller's frame.

    Port of ``sa_cluster_input()``. The reduction helpers do the work; this puts
    them in the one order all four clustering functions use, so that none of them
    can assemble the matrix its own way.
    """
    input_: ReduceInput = reduce_input(data, feats, scale, fn)
    points, point_type = reduce_points(input_.feats, input_.samples, cluster_scale)
    m = embedding_matrix(input_.x, cluster_scale, center, scale)
    return ClusterInput(
        m=m,
        points=list(points),
        point_type=point_type,
        n_samples=input_.n_samples,
        n_used=input_.x.shape[0],
        n_dropped=input_.n_dropped,
        n_feats=input_.x.shape[1],
        feats=list(input_.feats),
        dropped_feats=list(input_.dropped_feats),
    )


def resolve_n_clust(n_clust: Any, n: int, point_type: str) -> int:
    """How many clusters to cut to.

    Port of ``sa_cluster_n_clust()``. The two partitioning methods are told the
    count, so the count is theirs to check. One cluster is not a clustering and
    there cannot be more clusters than points, and both ends are refused here
    rather than left to the engine, which says it in terms of its own arguments.
    """
    value = check_count(n_clust, "n_clust", MIN_CLUSTERS)
    if value > n:
        raise SaValueError(
            f"`n_clust` must not exceed the {n} usable {point_type}(s) being "
            f"clustered, but is {value}. There cannot be more groups than there are "
            "things to put in them."
        )
    return value


def dbscan_min_pts(min_pts: Any, n: int, n_var: int, point_type: str) -> int:
    """How dense a neighbourhood has to be before DBSCAN calls it a cluster.

    Port of ``sa_dbscan_min_pts()``. The derived value is said out loud: it is the
    whole behaviour of the method and someone clustering 40 features is the last
    person who would think to check what it came out as.
    """
    if min_pts is not None:
        value = check_count(min_pts, "min_pts", MIN_CLUSTERS)
        if value > n:
            raise SaValueError(
                f"`min_pts` must not exceed the {n} usable {point_type}(s) being "
                f"clustered, but is {value}. No neighbourhood can hold more points "
                "than there are."
            )
        return value
    derived = min(n, max(_DBSCAN_MIN_PTS_FLOOR, min(int(n_var) + 1, n // 2)))
    notify(
        f"Using min_pts = {derived}, from the {n_var} variable(s) describing each "
        f"{point_type} and the {n} being clustered. Pass `min_pts` to set it."
    )
    return derived


def snn_params(
    k: Any,
    eps: Any,
    min_pts: Any,
    n: int,
    point_type: str,
) -> dict[str, int]:
    """The neighbourhood shared nearest neighbour clustering is run at.

    Port of ``sa_snn_params()``. Three arguments that only mean anything together,
    so they are resolved together. ``k`` is how many neighbours each point keeps,
    ``eps`` is how many of them two points must have in common before there is an
    edge between them, and ``min_pts`` is how many such edges a point needs before
    it is a core point. All three are counts of neighbours, which is why ``eps``
    here is nothing like the ``eps`` of :func:`cluster_dbscan`: that one is a
    radius in the units of the data.
    """
    # Everything that was supplied is checked for being the right kind of thing
    # before anything is derived, so that a call that is going to fail does not
    # first announce a default it never used. The one check that cannot come first
    # is `eps` against `k`, since until `k` is resolved there is nothing to check
    # it against.
    k_value = None if k is None else check_count(k, "k", MIN_CLUSTERS)
    eps_value = None if eps is None else check_count(eps, "eps", 1)
    min_pts_value = None if min_pts is None else check_count(min_pts, "min_pts", 1)

    if k_value is None:
        k_value = min(n - 1, max(_SNN_K_FLOOR, int(np.ceil(np.sqrt(n)))))
        notify(
            f"Using k = {k_value} neighbour(s), from the {n} {point_type}(s) being "
            "clustered. Pass `k` to set it."
        )
    elif k_value > n - 1:
        raise SaValueError(
            f"`k` must not exceed one less than the {n} usable {point_type}(s) being "
            f"clustered, which is {n - 1}, but is {k_value}. A {point_type} is not "
            "its own neighbour."
        )

    if eps_value is None:
        eps_value = max(1, k_value // 2)
    elif eps_value > k_value:
        raise SaValueError(
            f"`eps` must not exceed `k`, which is {k_value}, but is {eps_value}. Two "
            "points cannot share more neighbours than they each keep. Note that this "
            "`eps` counts shared neighbours; the one in `cluster_dbscan()` is a radius."
        )

    if min_pts_value is None:
        min_pts_value = max(MIN_CLUSTERS, k_value // 2)

    return {"k": k_value, "eps": eps_value, "min_pts": min_pts_value}


def derive_eps(m: np.ndarray, min_pts: int) -> float:
    """How far a neighbourhood reaches, when the caller has not said.

    Port of ``sa_cluster_eps()``. DBSCAN's ``eps`` is the one argument it cannot be
    given a fixed default for: it is a radius in the units of the data, so any
    constant would be wrong on the next matrix. It is derived from the k-distance
    curve - every point's distance to its ``min_pts - 1``th neighbour - and the
    value taken off it is the :data:`_EPS_QUANTILE` quantile.
    """
    from sklearn.neighbors import NearestNeighbors

    neighbours = min_pts - 1
    # `kneighbors()` on the points it was fitted to counts each point as its own
    # nearest neighbour at distance 0, so one more column is asked for and the
    # first is dropped. That is what R's `kNNdist(k = min_pts - 1)` returns.
    finder = NearestNeighbors(n_neighbors=min(neighbours + 1, m.shape[0]))
    finder.fit(m)
    distances, _ = finder.kneighbors(m)
    curve = distances[:, -1]

    finite = curve[np.isfinite(curve)]
    eps = float(np.quantile(finite, _EPS_QUANTILE)) if finite.size else float("nan")
    if not np.isfinite(eps) or eps <= 0:
        positive = finite[finite > 0]
        eps = float(positive.max()) if positive.size else _EPS_FALLBACK
    return eps


def _mean_width(values: np.ndarray) -> float:
    """Mean silhouette of a cluster, missing when none of its members has one."""
    present = values[~np.isnan(values)]
    return float(present.mean()) if present.size else float("nan")


@dataclass(frozen=True)
class ClusterTables:
    """The two tables a clustering reports, and the two counts of the answer."""

    assignments: pd.DataFrame
    clusters: pd.DataFrame
    n_clusters: int
    n_noise: int


def cluster_tables(cluster: Any, points: list[str], d: Any) -> ClusterTables:
    """The two tables a clustering reports, built from one vector of labels.

    Port of ``sa_cluster_tables()``. Every one of the four methods ends with a
    whole number per point and a distance matrix, and everything the result says
    about the grouping is derived from those two here. Doing it in one place is
    what keeps ``cluster`` meaning the same thing in all four: the labels are
    renumbered from 1 in order of first appearance, so that a k-means run and a
    DBSCAN run on the same points name their groups by the same rule and neither
    inherits whatever the engine happened to count from.

    Args:
        cluster: Labels as the engine returned them, :data:`NOISE_LABEL` for
            noise.
        points: Point labels.
        d: The distances the engine was run on, for the silhouette.
    """
    labels = np.asarray(cluster, dtype=int)
    assigned = labels != NOISE_LABEL
    # Renumbering by first appearance, not by size: a cluster is not more itself
    # for being large, and first appearance is the one order that does not change
    # when the caller reorders their rows for an unrelated reason.
    found = list(dict.fromkeys(int(value) for value in labels[assigned]))
    order = {value: index + 1 for index, value in enumerate(found)}
    renumbered = np.full(labels.shape, NOISE_LABEL, dtype=int)
    renumbered[assigned] = [order[int(value)] for value in labels[assigned]]

    widths = silhouette(d, renumbered)
    assignments = pd.DataFrame(
        {
            "points": list(points),
            "cluster": renumbered,
            "silhouette": widths,
        }
    )

    ids = list(range(1, len(found) + 1))
    # A single cluster has no other cluster to be far from, so its width is
    # undefined rather than zero, and the mean of an all-missing group is missing.
    clusters = pd.DataFrame(
        {
            "cluster": np.asarray(ids, dtype=int),
            "size": np.asarray([int((renumbered == value).sum()) for value in ids], dtype=int),
            "silhouette": np.asarray(
                [_mean_width(widths[renumbered == value]) for value in ids], dtype=float
            ),
        }
    )
    return ClusterTables(
        assignments=assignments,
        clusters=clusters,
        n_clusters=len(clusters.index),
        n_noise=int((renumbered == NOISE_LABEL).sum()),
    )
