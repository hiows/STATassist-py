"""The clustering kernel: a distance matrix and labels in, one width per point out.

Port of ``R/kernel_cluster.R``, written to the same rule as
:mod:`statassist.kernel.performance`: plain input in, an array out, no fitted
object kept anywhere and nothing said to the user. There is one kernel here,
because there is one number all four clustering methods can be read on.

It is written out rather than taken from ``sklearn.metrics.silhouette_samples``
for the reason the R original is written out rather than taken from
``cluster::silhouette()``. The definition is four lines of arithmetic over a
distance matrix this package already has in its hand, and the conventions that
matter here - noise taking no part, a singleton scoring zero, a single cluster
scoring missing - are not what ``silhouette_samples`` does with them: it refuses
fewer than two labels rather than returning missing values, and it has no notion
of a noise label at all, so a DBSCAN result would have to be filtered before and
scattered back after. Writing the arithmetic out is shorter than that, and
``silhouette_samples`` remains the thing to check it against.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.distance import squareform

from ..core.errors import SaValueError

__all__ = ["DIST_METHODS", "NOISE_LABEL", "cluster_dist", "silhouette"]

#: The distances a clustering can be measured on, in the order they are offered.
DIST_METHODS = ("euclidean", "correlation", "manhattan")

#: The cluster label that means "not in any cluster".
#:
#: Density based methods leave points out, and DBSCAN's convention for saying so
#: is 0. That makes it a **data value** rather than an index, so it is not shifted
#: when 1-based labels become 0-based ones anywhere else in this port; a label of
#: 0 means noise here exactly as it does in R.
NOISE_LABEL = 0


def _as_distance_matrix(d: Any, n_points: int) -> np.ndarray:
    """Read a square distance matrix, accepting the condensed form as well.

    R takes a :func:`stats::dist` object, which is the condensed lower triangle
    with the point count in an attribute. There is no such object here, so a
    square matrix is the argument and a condensed vector is accepted for the
    callers that have one, since that is what
    :func:`scipy.spatial.distance.pdist` returns.
    """
    array = np.asarray(d, dtype=float)
    if array.ndim == 1:
        array = squareform(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise SaValueError("`d` must be a square distance matrix, or a condensed one.")
    if array.shape[0] != n_points:
        raise SaValueError(
            f"`d` covers {array.shape[0]} points and `cluster` labels {n_points}; "
            "the distance matrix must be in the order the labels are in."
        )
    return array


def cluster_dist(x: Any, dist_method: str = "euclidean") -> np.ndarray:
    """Distances between the rows of ``x``, in condensed form.

    Port of ``sa_cluster_dist()``. ``"correlation"`` is ``1 - cor()``, which
    measures the shape of a profile rather than how high it sits, and the other
    two are the plain metrics :func:`stats::dist` computes.

    Written out rather than handed to :func:`scipy.spatial.distance.pdist` for
    one reason: missing values. ``pdist`` has no notion of them and returns
    ``NaN`` for a pair with a single gap, where R measures the pair on the
    columns it does share and scales the sum up to the full width, which is what
    lets a heatmap cluster data with holes in it. A pair that shares no column at
    all has no distance, and comes back ``NaN`` here as it does ``NA`` there.

    Args:
        x: Objects to measure between, in rows.
        dist_method: One of :data:`DIST_METHODS`.

    Returns:
        The condensed lower triangle, in the layout
        :func:`scipy.spatial.distance.squareform` reads.

    Raises:
        SaValueError: If ``dist_method`` is not one of :data:`DIST_METHODS`, or
            ``x`` is not a two-dimensional array of at least two rows.
    """
    if dist_method not in DIST_METHODS:
        raise SaValueError(
            "`dist_method` must be one of " + ", ".join(DIST_METHODS) + f". Got {dist_method}."
        )
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise SaValueError("`x` must be a matrix of at least two rows to measure between.")

    n_rows, n_cols = values.shape
    known = np.isfinite(values)
    out = np.empty(n_rows * (n_rows - 1) // 2)

    at = 0
    for i in range(n_rows - 1):
        for j in range(i + 1, n_rows):
            shared = known[i] & known[j]
            n_shared = int(shared.sum())
            left = values[i, shared]
            right = values[j, shared]
            if dist_method == "correlation":
                # A row with no variance has no correlation, which is left to the
                # caller to notice rather than turned into an error here.
                if n_shared < 2:
                    out[at] = np.nan
                elif left.std() == 0 or right.std() == 0:
                    out[at] = np.nan
                else:
                    out[at] = 1.0 - float(np.corrcoef(left, right)[0, 1])
            elif n_shared == 0:
                out[at] = np.nan
            elif dist_method == "euclidean":
                total = float(np.square(left - right).sum())
                out[at] = math.sqrt(total * n_cols / n_shared)
            else:
                total = float(np.abs(left - right).sum())
                out[at] = total * n_cols / n_shared
            at += 1

    return out


def silhouette(d: Any, cluster: Any) -> np.ndarray:
    """Silhouette width of every point.

    Port of ``sa_silhouette()``. The silhouette of a point compares how far it
    sits from its own cluster with how far it sits from the nearest cluster it is
    not in: ``(b - a) / max(a, b)``, where ``a`` is the mean distance to the other
    members of its own cluster and ``b`` is the smallest mean distance to the
    members of another. It runs from 1, meaning the point is far closer to its own
    group than to any other, through 0 at the border between two, to -1 for a
    point that would be better off elsewhere.

    Three conventions, all Rousseeuw's:

    * A point alone in its cluster scores 0. It has no ``a`` to speak of, and the
      alternative - dividing by nothing and calling it 1 - would score a singleton
      as the best-placed point in the data.
    * Noise scores missing, and takes no part in any other point's ``a`` or ``b``.
      It is not a cluster, so a point cannot be near to it in the sense ``b``
      measures.
    * A single cluster scores missing throughout. There is no other cluster for
      ``b`` to be about, and the width is a comparison rather than a measurement.

    Args:
        d: Distances between the points, square or condensed, in the order
            ``cluster`` is in.
        cluster: Cluster label per point, :data:`NOISE_LABEL` for noise.

    Returns:
        One silhouette width per point, ``NaN`` where undefined.

    Raises:
        SaValueError: If the distances do not match the labels in shape.

    References:
        Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
        interpretation and validation of cluster analysis. *Journal of
        Computational and Applied Mathematics*, 20, 53-65.
    """
    labels = np.asarray(cluster).astype(np.int64, copy=False)
    out = np.full(labels.size, np.nan)

    assigned = labels > NOISE_LABEL
    ids = np.unique(labels[assigned])
    matrix = _as_distance_matrix(d, labels.size)
    if ids.size < 2:
        return out

    # One boolean row per cluster, so the membership of each is worked out once
    # rather than once per point.
    members = np.array([assigned & (labels == group) for group in ids])
    sizes = members.sum(axis=1)

    # Mean distance from every point to the members of every cluster, in one
    # product. The diagonal is zero, so a point's own row contributes nothing to
    # its own cluster's total and dividing by one fewer member gives `a` directly.
    totals = matrix @ members.T
    own = np.searchsorted(ids, labels)

    for i in np.flatnonzero(assigned):
        group = own[i]
        if sizes[group] == 1:
            out[i] = 0.0
            continue
        a = totals[i, group] / (sizes[group] - 1)
        others = np.delete(totals[i] / sizes, group)
        b = float(others.min())
        scale = max(a, b)
        # Coincident points give a == b == 0, which is a tie rather than a division.
        out[i] = (b - a) / scale if scale > 0 else 0.0

    return out
