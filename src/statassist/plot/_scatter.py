"""What the reduction scatter reads out of its two result objects.

Port of the ``sa_reduction_scatter()``, ``sa_scatter_clusters()`` and
``sa_scatter_groups()`` helpers of ``R/draw_dim_reduction_plot.R``. A reduction
gives every point a coordinate and a clustering gives every point a label, and
because both read their input through the same helpers the two are about the same
rows in the same order. That is the whole reason the plot can exist without asking
either of them where its rows came from.

Colour and shape are given to two different things on purpose. A clustering is what
the data was found to say and a group is what the caller already knew, and the
interesting question is almost always whether the two agree. Drawing them on one
channel would answer it by hiding it: one would have to be left out, or the two
would have to be crossed into a single set of labels whose count is the product of
theirs. On two channels the agreement is read directly - one colour per marker is a
clustering that recovered the groups, and a marker split across colours is a group
the data does not see as one thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.result import SaCluster, SaReduction
from ..core.validate import check_count, fmt_est
from ..kernel.cluster import NOISE_LABEL
from ._theme import Theme, group_colors

__all__ = [
    "MAX_SCATTER_LEVELS",
    "SCATTER_MARKERS",
    "ScatterClusters",
    "ScatterGroups",
    "ScatterSpace",
    "scatter_clusters",
    "scatter_groups",
    "scatter_space",
]

#: The markers a group is drawn with, in the order they are handed out.
#:
#: R hands out ``pch`` codes and starts with the filled shapes, since a filled
#: marker is the one that reads at a small size. matplotlib has no filled and hollow
#: version of one marker - hollowness is a face colour rather than a shape - so the
#: sequence here is ten shapes that tell each other apart at a small size, which is
#: what R's ten codes were chosen for.
SCATTER_MARKERS = ("o", "^", "s", "D", "*", "v", "P", "X", "<", ">")

#: How many levels a marker channel will tell apart.
#:
#: Ten is where it stops: past that the markers are telling each other apart rather
#: than telling the groups apart, and the caller is asked to name their own or to
#: put the many-levelled channel on the colours instead.
MAX_SCATTER_LEVELS = len(SCATTER_MARKERS)

#: What the point-label column of a score table is called.
_POINT_COLUMN = "points"


@dataclass(frozen=True)
class ScatterSpace:
    """The two coordinates a reduction is drawn on, and what to call their axes."""

    x: np.ndarray
    y: np.ndarray
    xlab: str
    ylab: str
    points: list[str]


def scatter_space(reduction_result: Any, dims: Any) -> ScatterSpace:
    """Read two coordinates out of a reduction, and label their axes.

    Port of ``sa_reduction_scatter()``. Only a rotation has a share of the variance
    to report, and the share is what makes one of its axes wider than another. An
    embedding's coordinates carry no such number, so their labels are the names and
    nothing more.
    """
    # A clustering is the other half of the same pipeline and is easy to reach for
    # first, so it is turned away by name rather than by type.
    if isinstance(reduction_result, SaCluster):
        raise SaValueError(
            "`reduction_result` is a clustering, which gives every point a label and "
            "no coordinate. Reduce the same frame with perform_pca(), perform_tsne() "
            "or perform_umap() and pass the clustering as `cluster_result`."
        )
    if not isinstance(reduction_result, SaReduction):
        raise SaValueError(
            "`reduction_result` must be a reduction, as returned by perform_pca(), "
            "perform_tsne() or perform_umap()."
        )

    scores: pd.DataFrame = reduction_result["scores"]
    coords = [str(name) for name in scores.columns if str(name) != _POINT_COLUMN]

    wanted = [dims] if isinstance(dims, (str, int, float)) else list(dims)
    if len(wanted) != 2:
        raise SaValueError("`dims` must hold two numbers, naming the two coordinates to draw.")
    first = check_count(wanted[0], "dims", 1)
    second = check_count(wanted[1], "dims", 1)
    if first == second:
        raise SaValueError(f"`dims` must name two different coordinates, but names {first} twice.")
    if max(first, second) > len(coords):
        raise SaValueError(
            f"`dims` asks for coordinate {max(first, second)} of a "
            f"{reduction_result['analysis']} that has {len(coords)} "
            "(" + ", ".join(coords) + "). An embedding has as many as `n_dim` asked for."
        )

    drawn = [coords[first - 1], coords[second - 1]]
    labels = list(drawn)
    if reduction_result.get("variance") is not None:
        variance: pd.DataFrame = reduction_result["variance"]
        share = dict(zip(variance["component"].astype(str), variance["prop_var"], strict=True))
        labels = [f"{name} ({fmt_est(share[name])}%)" for name in drawn]

    return ScatterSpace(
        x=scores[drawn[0]].to_numpy(dtype=float),
        y=scores[drawn[1]].to_numpy(dtype=float),
        xlab=labels[0],
        ylab=labels[1],
        points=[str(value) for value in scores[_POINT_COLUMN]],
    )


@dataclass(frozen=True)
class ScatterClusters:
    """One colour per cluster, and grey for what did not join one."""

    cluster: np.ndarray
    col: list[Any]
    palette: list[Any]
    levels: list[str]
    n_noise: int


def scatter_clusters(
    cluster_result: Any,
    points: list[str],
    col: Any,
    cluster_lv: Any,
    palette_theme: Theme,
) -> ScatterClusters:
    """Colour every point by the cluster it was put in.

    Port of ``sa_scatter_clusters()``. The alignment check is the contract both
    objects already promise rather than anything new: a clustering and a reduction
    repeat the same ``points`` in the same order, so a mismatch here means two
    different frames were reduced and clustered rather than one.

    Noise is grey rather than a palette colour. A point left out is the absence of a
    cluster rather than a cluster of its own; giving it a colour beside the others
    would put it in the legend as though it were one.
    """
    if not isinstance(cluster_result, SaCluster):
        raise SaValueError(
            "`cluster_result` must be a clustering, as returned by cluster_hclust(), "
            "cluster_kmeans(), cluster_dbscan() or cluster_snn()."
        )
    assigned: pd.DataFrame = cluster_result["assignments"]
    if [str(value) for value in assigned[_POINT_COLUMN]] != points:
        raise SaValueError(
            "`cluster_result` and `reduction_result` describe different points. Both "
            "read their input through the same function, so this means they were given "
            "different frames, different `feats`, or one of them the sample scale and "
            "the other the feature scale."
        )

    labels = assigned["cluster"].to_numpy(dtype=int)
    n_clust = int(cluster_result["design"]["n_clusters"])
    if col is None:
        colours = group_colors(None, max(n_clust, 2))[:n_clust]
    else:
        held = [col] if isinstance(col, str) else list(col)
        if len(held) not in (1, n_clust):
            raise SaValueError(f"`col` must hold one colour, or one per cluster ({n_clust}).")
        colours = [held[index % len(held)] for index in range(n_clust)]

    point_col: list[Any] = [
        palette_theme.guide if value == NOISE_LABEL else colours[value - 1] for value in labels
    ]

    if cluster_lv is None:
        levels = [f"#{index + 1}" for index in range(n_clust)]
    else:
        levels = [str(value) for value in cluster_lv]
        if len(levels) != n_clust:
            raise SaValueError(f"`cluster_lv` must hold one label per cluster ({n_clust}).")
        if len(set(levels)) != len(levels):
            raise SaValueError("`cluster_lv` must not repeat a level.")

    return ScatterClusters(
        cluster=labels,
        col=point_col,
        palette=colours,
        levels=levels,
        n_noise=int(cluster_result["design"]["n_noise"]),
    )


@dataclass(frozen=True)
class ScatterGroups:
    """One marker per group level, and colours too when ``col`` was named."""

    group: pd.Categorical[str]
    marker: list[str]
    levels: list[str]
    marker_lv: list[str]
    palette: list[Any] | None
    col: list[Any] | None


def scatter_groups(
    group: Any,
    group_lv: Any,
    points: list[str],
    design: Any,
    col: Any = None,
    marker: Any = None,
) -> ScatterGroups:
    """Shape every point by the group it was known to be in.

    Port of ``sa_scatter_groups()``. ``group_lv`` sets the order the levels are
    drawn and read in, and nothing else. It does not select rows, which is where
    this parts company with :func:`~statassist.draw_grouped_boxplot`: over there a
    level left out is a level left out of the analysis, and here the reduction has
    already placed every point, so a point whose group went unlisted would vanish
    from a picture it belongs in.
    """
    values = pd.Series(group)
    if len(values) != len(points):
        raise SaValueError(_group_length_message(len(values), points, design))
    if values.isna().any():
        raise SaValueError(
            "`group` must not hold a missing value. Every point the reduction placed "
            "is drawn, so every one of them needs a marker."
        )
    as_text = [str(value) for value in values]
    # A factor's own level order is kept where there is one, which is what lets a
    # simulator's `group_lv` reach the legend without being named again. A level the
    # factor declares and no point uses is not drawn, since it has no point to draw.
    declared = (
        list(values.cat.categories) if isinstance(values.dtype, pd.CategoricalDtype) else None
    )

    if group_lv is None:
        if declared is not None:
            observed = set(as_text)
            levels = [str(level) for level in declared if str(level) in observed]
        else:
            levels = sorted(set(as_text))
    else:
        levels = [str(value) for value in group_lv]
        if len(set(levels)) != len(levels):
            raise SaValueError("`group_lv` must not repeat a level.")
        unlisted = [value for value in dict.fromkeys(as_text) if value not in set(levels)]
        if unlisted:
            raise SaValueError(
                "`group_lv` leaves out " + ", ".join(unlisted) + ", which `group` uses. "
                "Here it sets the order the levels are drawn and read in rather than "
                "which rows are kept: every point the reduction placed is drawn, so a "
                "level left out is a point with no marker rather than a point removed."
            )

    if marker is None:
        if len(levels) > MAX_SCATTER_LEVELS:
            raise SaValueError(
                f"`group` has {len(levels)} levels and there are {MAX_SCATTER_LEVELS} "
                "markers to tell them apart with. Past that the markers are "
                "distinguishing themselves rather than the groups; name `marker` with "
                "one per level, or use `cluster_result` for the colours."
            )
        markers = list(SCATTER_MARKERS)
    else:
        held = [marker] if isinstance(marker, str) else list(marker)
        if len(held) not in (1, len(levels)):
            raise SaValueError(
                f"`marker` must hold one marker, or one per group level ({len(levels)})."
            )
        markers = [held[index % len(held)] for index in range(len(levels))]

    palette: list[Any] | None = None
    if col is not None:
        held_col = [col] if isinstance(col, str) else list(col)
        if len(held_col) not in (1, len(levels)):
            raise SaValueError(
                f"`col` must hold one colour, or one per group level ({len(levels)})."
            )
        palette = [held_col[index % len(held_col)] for index in range(len(levels))]

    at = [levels.index(value) for value in as_text]
    return ScatterGroups(
        group=pd.Categorical(as_text, categories=levels),
        marker=[markers[index] for index in at],
        levels=levels,
        marker_lv=[markers[index] for index in range(len(levels))],
        palette=palette,
        col=None if palette is None else [palette[index] for index in at],
    )


def _group_length_message(n_given: int, points: list[str], design: Any) -> str:
    """Why a ``group`` of the wrong length is usually the wrong length.

    The likeliest way to get here is to pass the grouping column of the frame that
    was reduced after the reduction had dropped rows out of it, so that case names
    itself instead of leaving two numbers to be compared.
    """
    extra = ""
    if design["n_dropped"] > 0 and n_given == design["n_samples"]:
        extra = (
            f" The reduction dropped {design['n_dropped']} of the {design['n_samples']} "
            "row(s) it was given, so it holds fewer points than `data` had rows. "
            "`reduction_result['points']` names the ones that are left."
        )
    elif design["point_type"] == "feature":
        extra = (
            " These points are features rather than samples, so `group` labels the "
            f"{len(points)} feature(s) that were embedded."
        )
    return (
        f"`group` must hold one label per point, so {len(points)} of them, but holds "
        f"{n_given}.{extra}"
    )
