"""The picture the unsupervised family ends at.

Port of ``R/draw_dim_reduction_plot.R``. A reduction gives every point a coordinate
and a clustering gives every point a label, and because both read their input
through the same helpers the two are about the same rows in the same order. That is
the whole reason this function can exist without asking either of them where their
rows came from.

The legend goes in a panel of its own beside the plot rather than inside it. A
scatter has no corner that is reliably empty - which corner is free is a property of
the data - and two readings, colour and shape, need two blocks with a gap between
them to say that they are two.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.errors import SaValueError
from ..core.validate import check_flag, check_lim, check_scalar_num
from ._scatter import (
    SCATTER_MARKERS,
    ScatterClusters,
    ScatterGroups,
    scatter_clusters,
    scatter_groups,
    scatter_space,
)
from ._theme import CHAR_WIDTH, figure, font, set_margin, theme

__all__ = ["SCATTER_VIEWS", "draw_dim_reduction_plot"]

#: Which channels ended up carrying which reading.
#:
#: Decided from which arguments arrived, so it is carried on the result rather than
#: left to be inferred from which columns are present.
SCATTER_VIEWS = ("both", "cluster", "group", "plain")

#: The marker a point gets when no group is naming the shapes.
_PLAIN_MARKER = SCATTER_MARKERS[0]

#: How much of the figure the legend panel may take, and the room left beside the
#: widest line in it.
#:
#: R measures the widest legend line in characters and gives it up to 35% of the
#: device. The same rule here, with a character's width taken from the font size.
_LEGEND_MAX_SHARE = 0.35
_LEGEND_PAD_CHARS = 5

#: How big a scatter point is at ``cex = 1``, in points squared.
#:
#: matplotlib sizes a marker by area and R by diameter, so ``cex`` is squared on the
#: way through. The base is :mod:`~statassist.plot.prediction`'s, so a point reads
#: the same size in both scatters.
_POINT_AREA = 20.0

#: The margins of the plot panel and of the legend panel, in lines of text.
_PANEL_MARGIN = (5.1, 4.6, 4.1, 1.1)


def draw_dim_reduction_plot(
    reduction_result: Any,
    group: Any = None,
    group_lv: Any = None,
    cluster_result: Any = None,
    cluster_lv: Any = None,
    dims: Any = (1, 2),
    anno_points: bool = False,
    dark: bool = False,
    asp: float | None = None,
    col: Any = None,
    marker: Any = None,
    cex: float = 1.2,
    xlim: Any = None,
    ylim: Any = None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    cex_axis: float = 1.2,
    cex_lab: float = 1.3,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    cex_anno: float | None = None,
) -> pd.DataFrame:
    """Draw a reduction as a scatter of its points.

    Plots two coordinates of a :func:`~statassist.perform_pca`,
    :func:`~statassist.perform_tsne` or :func:`~statassist.perform_umap` result
    against each other. A clustering of the same frame colours the points and a
    known grouping shapes them, so what the data was found to say and what the
    caller already knew are read off one picture.

    ``cluster_result`` takes the colours and ``group`` takes the markers when both
    are given, and giving both is the point of the arrangement rather than a
    conflict to be resolved. With one channel alone, that channel takes the colours
    when ``col`` is named; otherwise a lone clustering is coloured and a lone
    grouping is shaped. Give neither and the points are drawn in the foreground
    colour.

    The clustering has to be of the same points, which the two contracts already
    promise: both read their input through the same function, so a mismatch is
    refused rather than lined up by position.

    A density method can leave a point in no cluster at all. Those points are grey
    rather than a palette colour, since a point left out is the absence of a cluster
    and not a cluster of its own, and the legend counts them on a line of their own.

    A principal component analysis reports what share of the variance each of its
    components carries, so its axis labels carry it too: ``PC1 (30.2%)`` is read
    from ``variance`` rather than recomputed. An embedding has no such number and
    its axes are labelled with their names alone. ``asp=1`` is what makes one unit
    of the vertical axis the same length as one unit of the horizontal, which is
    worth setting when the distance between two points is what is being read.

    Args:
        reduction_result: A reduction, as returned by the three ``perform_*``
            functions.
        group: One label per point, or ``None`` for no marker channel. Points are
            the rows the reduction kept, which ``reduction_result["points"]`` names,
            and on the feature scale they are features rather than samples.
        group_lv: The levels of ``group`` in the order they are drawn and listed in,
            or ``None`` for the factor's own order and otherwise alphabetical.
            Unlike :func:`~statassist.draw_grouped_boxplot`'s, this argument selects
            no rows: a level ``group`` uses and ``group_lv`` leaves out is an error
            rather than a point dropped from the picture.
        cluster_result: A clustering of the same points, or ``None`` for no
            colouring.
        cluster_lv: One label per cluster, in the order the clustering numbers them,
            or ``None`` for ``#1``, ``#2``, and so on. Noise is still listed as
            ``noise (n)`` when present.
        dims: Which two coordinates to draw, as positions in the score table.
        anno_points: Whether to write each point's label beside it.
        dark: Whether to draw on a dark background.
        asp: Aspect ratio, or ``None`` to let the axes fill the panel
            independently. ``1`` makes distances comparable between the two axes.
        col: One colour for every point, one per cluster when ``cluster_result`` is
            given, one per group level when only ``group`` is given, or ``None`` for
            the group palette on clusters and the foreground colour elsewhere.
            Noise is grey whatever this says.
        marker: One matplotlib marker for every point, one per group level when
            ``group`` is given, or ``None`` for a filled circle and the default
            marker sequence respectively. This is R's ``pch`` under the name of the
            thing matplotlib takes.
        cex: Size of the plotted points, as a multiplier.
        xlim: Horizontal axis range, or ``None`` to take it from the coordinates.
        ylim: The same vertically.
        xlab: Horizontal axis label, or ``None`` to build it from the reduction.
        ylab: The same vertically.
        main: Figure title, or ``None`` for the engine's label.
        cex_axis: Relative size of the tick labels.
        cex_lab: Relative size of the axis labels.
        cex_main: Relative size of the title.
        cex_legend: Relative size of the legend.
        cex_anno: Relative size of the point labels. ``None`` matches
            ``cex_legend``.

    Returns:
        A DataFrame of the points as they were drawn: ``points``, ``x``, ``y``,
        ``cluster`` and ``group`` when those were given, then ``col`` and
        ``marker``. Which of the four readings this was is in
        ``frame.attrs["view"]``, one of :data:`SCATTER_VIEWS`.

    Raises:
        SaValueError: If ``reduction_result`` is not a reduction, if a clustering
            describes different points, if ``dims`` asks for a coordinate the
            reduction does not have, or if a channel argument is not of the kind it
            has to be.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import (
        ...     cluster_kmeans,
        ...     draw_dim_reduction_plot,
        ...     perform_pca,
        ...     simulate_two_groups,
        ... )
        >>> sim = simulate_two_groups(n_feats=30, n_up=5, n_down=5, seed=3)
        >>> res = perform_pca(sim.args["data"])
        >>> cl = cluster_kmeans(sim.args["data"], n_clust=2, seed=1)
        >>> drawn = draw_dim_reduction_plot(
        ...     res, group=sim.args["group"], cluster_result=cl
        ... )
        >>> drawn.attrs["view"], list(drawn.columns)
        ('both', ['points', 'x', 'y', 'cluster', 'group', 'col', 'marker'])
    """
    space = scatter_space(reduction_result, dims)
    anno_points = check_flag(anno_points, "anno_points")
    dark = check_flag(dark, "dark")
    x_limits = check_lim(xlim, "xlim")
    y_limits = check_lim(ylim, "ylim")
    cex = check_scalar_num(cex, "cex", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    if asp is not None:
        asp = check_scalar_num(asp, "asp", 0, lower_open=True)
    if cex_anno is None:
        cex_anno = cex_legend
    else:
        cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    if group is None and group_lv is not None:
        raise SaValueError("`group_lv` names the levels of `group`, which was not given.")
    if cluster_result is None and cluster_lv is not None:
        raise SaValueError("`cluster_lv` names the levels of the clustering, which was not given.")

    palette = theme(dark)
    labels = space.points

    clusters = (
        None
        if cluster_result is None
        else scatter_clusters(cluster_result, labels, col, cluster_lv, palette)
    )
    groups = (
        None
        if group is None
        else scatter_groups(
            group,
            group_lv,
            labels,
            reduction_result["design"],
            col=None if cluster_result is not None else col,
            marker=marker,
        )
    )

    point_col = _point_colors(clusters, groups, col, labels, palette.fg)
    point_marker = _point_markers(groups, marker, labels)
    view = _view(clusters, groups)

    fig = figure()
    fig.set_facecolor(palette.bg)
    keys = _legend_keys(clusters, groups, palette.fg, palette.guide)
    if keys:
        share = _legend_share(keys, cex_legend, fig.get_size_inches()[0])
        grid = fig.add_gridspec(1, 2, width_ratios=[1 - share, share], wspace=0.05)
        ax = fig.add_subplot(grid[0, 0])
        legend_ax: Any = fig.add_subplot(grid[0, 1])
    else:
        ax = fig.add_subplot()
        legend_ax = None
    set_margin(fig, _PANEL_MARGIN)

    ax.set_facecolor(palette.bg)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(palette.fg)
    ax.tick_params(colors=palette.fg, labelsize=font(cex_axis))

    # One call per distinct marker, because matplotlib takes one marker per scatter
    # and R takes one per point. The colours stay per point either way, so the
    # grouping is by shape alone and no point changes colour for being drawn in a
    # different call.
    area = _POINT_AREA * cex**2
    for shape in dict.fromkeys(point_marker):
        at = [index for index, value in enumerate(point_marker) if value == shape]
        ax.scatter(
            space.x[at],
            space.y[at],
            marker=shape,
            c=[point_col[index] for index in at],
            s=area,
        )
    if anno_points:
        for index, label in enumerate(labels):
            ax.annotate(
                label,
                (space.x[index], space.y[index]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=font(cex_anno),
                color=palette.fg,
            )

    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if asp is not None:
        ax.set_aspect(asp)
    ax.set_xlabel(space.xlab if xlab is None else xlab, fontsize=font(cex_lab), color=palette.fg)
    ax.set_ylabel(space.ylab if ylab is None else ylab, fontsize=font(cex_lab), color=palette.fg)
    ax.set_title(
        reduction_result["engine"]["label"] if main is None else main,
        fontsize=font(cex_main),
        color=palette.fg,
    )

    if legend_ax is not None:
        _draw_legend(legend_ax, keys, palette, cex_legend)

    drawn = pd.DataFrame({"points": labels, "x": space.x, "y": space.y})
    if clusters is not None:
        drawn["cluster"] = clusters.cluster
    if groups is not None:
        drawn["group"] = groups.group
    drawn["col"] = point_col
    drawn["marker"] = point_marker
    drawn.attrs["view"] = view
    return drawn


def _point_colors(
    clusters: ScatterClusters | None,
    groups: ScatterGroups | None,
    col: Any,
    labels: list[str],
    foreground: str,
) -> list[Any]:
    """One colour per point, from whichever channel is carrying them.

    With no clustering and no group colouring there is nothing for a palette to be
    one of, so ``col`` is a single colour for every point and the default is
    whatever the foreground is on this background.
    """
    if clusters is not None:
        return list(clusters.col)
    if groups is not None and groups.col is not None:
        return list(groups.col)
    if col is None:
        return [foreground] * len(labels)
    held = [col] if isinstance(col, str) else list(col)
    if len(held) not in (1, len(labels)):
        raise SaValueError(
            f"`col` must hold one colour, or one per point ({len(labels)}), when no "
            "`cluster_result` or `group` colouring applies."
        )
    return [held[index % len(held)] for index in range(len(labels))]


def _point_markers(groups: ScatterGroups | None, marker: Any, labels: list[str]) -> list[str]:
    """One marker per point, from the group channel or from ``marker`` itself."""
    if groups is not None:
        return list(groups.marker)
    if marker is None:
        return [_PLAIN_MARKER] * len(labels)
    held = [marker] if isinstance(marker, str) else list(marker)
    if len(held) not in (1, len(labels)):
        raise SaValueError(
            f"`marker` must hold one marker, or one per point ({len(labels)}), when no "
            "`group` is given."
        )
    return [str(held[index % len(held)]) for index in range(len(labels))]


def _view(clusters: ScatterClusters | None, groups: ScatterGroups | None) -> str:
    """Which of the four readings this was."""
    if clusters is not None and groups is not None:
        return SCATTER_VIEWS[0]
    if clusters is not None:
        return SCATTER_VIEWS[1]
    if groups is not None:
        return SCATTER_VIEWS[2]
    return SCATTER_VIEWS[3]


def _legend_keys(
    clusters: ScatterClusters | None,
    groups: ScatterGroups | None,
    foreground: str,
    guide: str,
) -> list[tuple[str, list[tuple[str, Any, str]]]]:
    """The legend blocks, as a title and a list of label, colour and marker.

    Built before the layout because the widest line is what decides how much of the
    figure the legend panel takes.
    """
    keys: list[tuple[str, list[tuple[str, Any, str]]]] = []
    if clusters is not None:
        block = [
            (level, colour, "s")
            for level, colour in zip(clusters.levels, clusters.palette, strict=True)
        ]
        if clusters.n_noise > 0:
            block.append((f"noise ({clusters.n_noise})", guide, "s"))
        keys.append(("cluster", block))
    if groups is not None:
        colours: list[Any] = (
            [foreground] * len(groups.levels) if groups.palette is None else list(groups.palette)
        )
        keys.append(
            (
                "group",
                [
                    (level, colour, shape)
                    for level, colour, shape in zip(
                        groups.levels, colours, groups.marker_lv, strict=True
                    )
                ],
            )
        )
    return keys


def _legend_share(
    keys: list[tuple[str, list[tuple[str, Any, str]]]],
    cex_legend: float,
    width: float,
) -> float:
    """How much of the figure's width the legend panel takes.

    Enough for the widest line it has to hold, capped at
    :data:`_LEGEND_MAX_SHARE` so that a long cluster label cannot squeeze the
    scatter down to nothing.
    """
    widest = max(
        max((len(text) for _, block in keys for text, _, _ in block), default=0),
        max(len(name) for name, _ in keys),
    )
    wanted = (widest + _LEGEND_PAD_CHARS) * font(cex_legend) / 72.0 * CHAR_WIDTH
    return min(_LEGEND_MAX_SHARE, wanted / max(width, 1.0))


def _draw_legend(
    legend_ax: Any,
    keys: list[tuple[str, list[tuple[str, Any, str]]]],
    palette: Any,
    cex_legend: float,
) -> None:
    """Put the blocks at the two ends of the legend panel.

    Anchored to the ends rather than stacked one after the other, since colour and
    marker are separate readings and a gap between them is what says so. matplotlib
    keeps one legend per axes, so a second block needs its artist added by hand.
    """
    from matplotlib.lines import Line2D

    legend_ax.set_axis_off()
    legend_ax.set_facecolor(palette.bg)
    both = len(keys) > 1
    for position, (title, block) in enumerate(keys):
        handles = [
            Line2D(
                [],
                [],
                marker=shape,
                color="none",
                markerfacecolor=colour,
                markeredgecolor=colour,
                label=label,
            )
            for label, colour, shape in block
        ]
        where = "upper center" if both and position == 0 else "lower center" if both else "center"
        legend = legend_ax.legend(
            handles=handles,
            title=title,
            loc=where,
            frameon=False,
            fontsize=font(cex_legend),
            title_fontsize=font(cex_legend),
            labelcolor=palette.fg,
        )
        if legend.get_title() is not None:
            legend.get_title().set_color(palette.fg)
        if position < len(keys) - 1:
            legend_ax.add_artist(legend)
