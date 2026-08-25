"""The clustered heatmap: one cell per feature and sample.

Port of ``R/draw_heatmap.R``. The wide input the comparison functions take is
transposed, so features run down the rows the way an expression heatmap is
usually read, with the sample groups as a coloured strip above the columns and a
dendrogram on each axis that was clustered.

R hands the drawing to :func:`stats::heatmap` and spends most of its length on
that function's panel layout and on the colour key it does not draw. matplotlib
lays panels out with a gridspec and draws a colour bar, so that arithmetic is not
carried over; what is carried over is every decision about the data - the
scaling, the distance, the trees, the colour range and its midpoint - which is
what the returned values are about.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage

from ..core.errors import SaValueError, notify
from ..core.validate import (
    check_count,
    check_flag,
    check_range,
    check_scalar_num,
    validate_wide_input,
)
from ..kernel.cluster import DIST_METHODS, cluster_dist
from ._hist import pretty
from ._theme import figure, font

__all__ = [
    "HCLUST_METHODS",
    "HEATMAP_SCALES",
    "LINKAGE_NAMES",
    "Clustering",
    "draw_heatmap",
]

#: How a feature or a sample can be put on a comparable scale before drawing.
HEATMAP_SCALES = ("feature", "sample", "none")

#: The linkages offered, R's ``hclust_method``.
HCLUST_METHODS = ("average", "complete", "ward.D2")

#: What each of them is called in :func:`scipy.cluster.hierarchy.linkage`.
#:
#: Shared with :func:`~statassist.draw_corrplot`, which clusters its own matrix
#: before handing it here and has to reach the same tree by the same name.
LINKAGE_NAMES = {"average": "average", "complete": "complete", "ward.D2": "ward"}

#: The diverging ramp, blue through white to red. Its middle is a meaningful
#: value rather than a colour choice, which is what ``zlim`` is made symmetric
#: about.
_RAMP = ("blue", "white", "red")

#: Where a cell with no value is drawn, so that a gap reads as a gap rather than
#: as background. R's ``gray92``, as hex.
_MISSING_COLOR = "#EBEBEB"

#: How many features and samples it takes to have something to cluster.
_MIN_AXIS = 2

#: How wide the colour bar and the numbers beside it are, in inches.
_KEY_BAR_INCHES = 0.9

#: How wide a dendrogram panel is, in inches.
_TREE_INCHES = 0.7

#: The air left around the whole figure, in inches.
_EDGE_INCHES = 0.12

#: How many characters a cell label is sized to fit: a sign, two digits, a point
#: and two decimals.
_ANNO_CHARS = 6


class Clustering(NamedTuple):
    """The tree behind one dendrogram, and the order it put its objects in.

    R returns an :func:`stats::hclust` object. The counterpart here is the
    linkage matrix :mod:`scipy.cluster.hierarchy` builds, beside the leaf order
    read off the drawn dendrogram and the two methods that produced it, so the
    tree the plot shows can be checked rather than eyeballed.

    ``order`` indexes from 0, as everything else in this port does, where R's
    ``hclust$order`` indexes from 1.
    """

    linkage: np.ndarray
    order: np.ndarray
    method: str
    dist_method: str
    labels: list[str]


def draw_heatmap(
    data: Any,
    group: Any = None,
    group_lv: Any = None,
    feats: Any = None,
    scale: str = "feature",
    zlim: Any = None,
    dist_method: str = "euclidean",
    hclust_method: str = "average",
    cluster_feats: bool = True,
    cluster_samples: bool = True,
    feat_labels: Any = None,
    sample_labels: Any = None,
    show_feat_names: bool = True,
    show_sample_names: bool = True,
    anno: bool = False,
    cex_anno: float = 1.0,
    n_colors: int = 101,
    main: str | None = None,
    cex_axis: float = 0.9,
    cex_main: float = 1.5,
    cex_legend: float = 1.2,
) -> dict[str, Any]:
    """Draw a clustered heatmap of features by samples.

    Args:
        data: Wide format, one row per observation and one column per feature.
            This is the layout :func:`~statassist.compare_two_groups` takes. A
            numpy array with no column names has them made up as ``V1``, ``V2``
            and so on.
        group: Grouping vector with one entry per row of ``data``, or ``None``
            with ``group_lv`` to draw without a group strip or legend.
        group_lv: Group levels, in the order they should appear in the legend, or
            ``None`` with ``group``. Rows belonging to any other level are
            dropped when both are supplied.
        feats: Feature columns to draw, or ``None`` for every column. The order
            given is the order the rows are drawn in when they are not clustered.
        scale: How to put the features on a comparable scale before drawing, one
            of :data:`HEATMAP_SCALES`. ``"feature"`` z-scores each feature across
            the samples, ``"sample"`` each sample across the features, and
            ``"none"`` draws the values as they arrived.
        zlim: Length-2 range the colours span, or ``None`` to derive it from the
            values being drawn. Values outside a supplied range are drawn at the
            end of the scale rather than left blank, and how many were is
            reported.
        dist_method: Distance behind the clustering, one of
            :data:`~statassist.kernel.DIST_METHODS`.
        hclust_method: Linkage, one of :data:`HCLUST_METHODS`.
        cluster_feats: Whether to cluster and reorder the features.
        cluster_samples: Whether to cluster and reorder the samples.
        feat_labels: Labels to draw in place of the column names, one per feature.
        sample_labels: Labels to draw in place of the row names, one per row of
            ``data``, filtered along with it.
        show_feat_names: Whether to draw the feature labels.
        show_sample_names: Whether to draw the sample labels.
        anno: If ``True``, write each cell value on the cell, rounded to two
            decimal places. The numbers are the values in the returned
            ``matrix``.
        cex_anno: Character expansion for those cell labels.
        n_colors: Number of colours in the blue-white-red ramp.
        main: Plot title.
        cex_axis: Character expansion for the axis labels.
        cex_main: Character expansion for the title.
        cex_legend: Character expansion for the group legend.

    Returns:
        A dict:

        ``matrix``
            The scaled matrix as it was drawn: features in rows, samples in
            columns, both in the order they appear on the plot, labelled as they
            were drawn. Values are not clamped to ``zlim``, which is a decision
            about colour rather than about the data.
        ``feat_order``, ``sample_order``
            The permutations the clustering chose, as indices into ``feats`` and
            into the rows of ``data`` that were kept.
        ``feat_hclust``, ``sample_hclust``
            The :class:`Clustering` behind each dendrogram, or ``None`` for an
            axis that was not clustered.
        ``zlim``
            The colour range, derived or as supplied.
        ``group_colors``
            The colour drawn for each group level, or ``None`` when no group was
            supplied.

    Raises:
        SaValueError: If an argument is out of range, if the labels do not match
            what they label, if there are fewer than two features or samples, or
            if the data hold no finite value.

    Notes:
        Features are z-scored across the samples by default. Without it a single
        high-abundance feature takes the whole colour range and everything else
        is left white, since the colour scale is shared by every cell in the plot
        and features are not measured on a common scale. A feature with no
        variance would divide by zero, so it is only centred and ends up flat at
        the middle of the scale; how many were is reported.

        A diverging palette needs a meaningful midpoint. When ``zlim`` is not
        given, zero is that midpoint if the values being drawn have both signs,
        which is always the case after z-scoring, and the range is made symmetric
        around it. Values of one sign only have no such point, so the range is
        the range of the values and white falls in the middle of it.

        Missing values are drawn as grey cells rather than dropped, so a gap is
        visible as a gap. Clustering measures a pair on the values they share; a
        pair that shares none has no distance at all, and rather than fail, that
        axis is left in its input order.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import draw_heatmap, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=8, n_up=2, n_down=2, seed=3)
        >>> out = draw_heatmap(
        ...     sim.args["data"],
        ...     sim.args["group"],
        ...     sim.args["group_lv"],
        ...     show_sample_names=False,
        ... )
        >>> out["sample_hclust"].method
        'average'
    """
    if scale not in HEATMAP_SCALES:
        raise SaValueError(
            "`scale` must be one of " + ", ".join(HEATMAP_SCALES) + f". Got {scale}."
        )
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
    cluster_feats = check_flag(cluster_feats, "cluster_feats")
    cluster_samples = check_flag(cluster_samples, "cluster_samples")
    show_feat_names = check_flag(show_feat_names, "show_feat_names")
    show_sample_names = check_flag(show_sample_names, "show_sample_names")
    anno = check_flag(anno, "anno")
    cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    n_colors = check_count(n_colors, "n_colors", 3)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    limits = None
    if zlim is not None:
        limits = check_range(zlim, "zlim")
        if limits[0] == limits[1]:
            raise SaValueError(f"`zlim` must have two different ends, but both are {limits[0]}.")
    if (group is None) != (group_lv is None):
        raise SaValueError("`group` and `group_lv` must both be supplied or both be `None`.")

    frame = _as_frame(data)
    if feats is None:
        feats = list(frame.columns)
    if sample_labels is None:
        names = [str(name) for name in frame.index]
    else:
        names = [str(name) for name in sample_labels]
        if len(names) != len(frame.index):
            raise SaValueError(
                "`sample_labels` must have one entry per row of `data`: got "
                f"{len(names)} for {len(frame.index)} rows."
            )

    validated = validate_wide_input(
        frame.reset_index(drop=True), feats, group, group_lv, id=names, min_levels=1
    )
    feats = list(validated.feats)
    kept = [str(name) for name in validated.id] if validated.id is not None else names

    if feat_labels is not None:
        rows = [str(name) for name in feat_labels]
        if len(rows) != len(feats):
            raise SaValueError(
                "`feat_labels` must have one entry per feature in `feats`: got "
                f"{len(rows)} for {len(feats)} feature(s)."
            )
    else:
        rows = list(feats)
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    # Features down the rows and samples across the columns, which is the way an
    # expression heatmap is read and the orientation the group strip needs to sit
    # above the columns it describes.
    matrix = validated.data[feats].to_numpy(dtype=float).T
    if matrix.shape[0] < _MIN_AXIS or matrix.shape[1] < _MIN_AXIS:
        raise SaValueError(
            f"`draw_heatmap()` needs at least {_MIN_AXIS} features and {_MIN_AXIS} "
            f"samples to cluster and draw, but got {matrix.shape[0]} feature(s) and "
            f"{matrix.shape[1]} sample(s)."
        )
    matrix = _scale_matrix(matrix, scale)

    feat_tree = (
        _hclust(matrix, dist_method, hclust_method, rows, "feature") if cluster_feats else None
    )
    sample_tree = (
        _hclust(matrix.T, dist_method, hclust_method, kept, "sample") if cluster_samples else None
    )
    feat_order = np.arange(matrix.shape[0]) if feat_tree is None else feat_tree.order
    sample_order = np.arange(matrix.shape[1]) if sample_tree is None else sample_tree.order

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise SaValueError("`data` holds no finite value to draw.")
    if limits is None:
        limits = _colour_range(finite)

    drawn = matrix[np.ix_(feat_order, sample_order)]
    clamped = int(np.sum((matrix < limits[0]) | (matrix > limits[1])))
    if clamped > 0:
        notify(
            f"{clamped} of {matrix.size} cell(s) lie outside `zlim` and are drawn at "
            "the end of the colour scale."
        )

    group_colors = None
    strip = None
    if validated.group is not None:
        levels = [str(level) for level in validated.group.categories]
        palette = _group_palette(len(levels))
        group_colors = dict(zip(levels, palette, strict=True))
        codes = np.asarray(validated.group.codes)[sample_order]
        strip = [palette[code] for code in codes]

    _draw(
        drawn=drawn,
        row_labels=[rows[i] for i in feat_order] if show_feat_names else None,
        col_labels=[kept[i] for i in sample_order] if show_sample_names else None,
        feat_tree=feat_tree,
        sample_tree=sample_tree,
        limits=limits,
        n_colors=n_colors,
        strip=strip,
        group_colors=group_colors,
        anno=anno,
        cex_anno=cex_anno,
        cex_axis=cex_axis,
        cex_main=cex_main,
        cex_legend=cex_legend,
        main=main,
    )

    return {
        "matrix": pd.DataFrame(
            drawn,
            index=[rows[i] for i in feat_order],
            columns=[kept[i] for i in sample_order],
        ),
        "feat_order": feat_order,
        "sample_order": sample_order,
        "feat_hclust": feat_tree,
        "sample_hclust": sample_tree,
        "zlim": limits,
        "group_colors": group_colors,
    }


def _as_frame(data: Any) -> pd.DataFrame:
    """The input as a frame with column names, making them up if it has none.

    A matrix with no column names has them made up as ``V1``, ``V2`` and so on,
    and repeated row names are kept, since a sample name is a naming choice
    rather than a key.
    """
    if isinstance(data, pd.DataFrame):
        return data
    array = np.asarray(data)
    if array.ndim != 2:
        raise SaValueError("`data` must be two-dimensional: one row per observation.")
    return pd.DataFrame(array, columns=[f"V{i + 1}" for i in range(array.shape[1])])


def _scale_matrix(x: np.ndarray, scale: str) -> np.ndarray:
    """Centre and standardise along one margin.

    Port of ``sa_scale_matrix()``. A feature with no spread has nothing to
    standardise by, so it is centred and left flat rather than turned into
    ``NaN`` by a division by zero.
    """
    if scale == "none":
        return x
    axis = 1 if scale == "feature" else 0
    with warnings.catch_warnings():
        # A margin that is entirely missing has no mean and no spread, which is
        # the flat case handled below rather than something to warn about.
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(x, axis=axis, keepdims=True)
        spread = np.nanstd(x, axis=axis, ddof=1, keepdims=True)
    flat = ~np.isfinite(spread) | (spread == 0)
    n_flat = int(np.sum(flat & np.isfinite(centre)))
    if n_flat > 0:
        notify(
            f"{n_flat} {scale}(s) have no variance to scale by and are drawn flat at "
            "the middle of the colour scale."
        )
    scaled: np.ndarray = (x - centre) / np.where(flat, 1.0, spread)
    return scaled


def _hclust(
    x: np.ndarray, dist_method: str, hclust_method: str, labels: list[str], axis: str
) -> Clustering | None:
    """Cluster the rows of ``x``.

    Port of ``sa_heatmap_hclust()``. The same distance
    :func:`~statassist.kernel.cluster_dist` measures for a clustering, so that
    the tree drawn here and a tree cut there are one tree when they are asked for
    on the same terms. Undefined distances leave the axis in its input order
    rather than failing.
    """
    distances = cluster_dist(x, dist_method)
    if not np.isfinite(distances).all():
        notify(
            f"Not clustering the {axis}s: some distances are undefined, which happens "
            "when a pair shares no observation or has no variance. The input order is "
            "kept."
        )
        return None
    tree = linkage(distances, method=LINKAGE_NAMES[hclust_method])
    order = dendrogram(tree, no_plot=True)["leaves"]
    return Clustering(
        linkage=tree,
        order=np.asarray(order, dtype=int),
        method=hclust_method,
        dist_method=dist_method,
        labels=labels,
    )


def _colour_range(finite: np.ndarray) -> tuple[float, float]:
    """The range the ramp spans, symmetric about zero when both signs are drawn."""
    if bool((finite > 0).any()) and bool((finite < 0).any()):
        reach = float(np.abs(finite).max())
        limits = (-reach, reach)
    else:
        limits = (float(finite.min()), float(finite.max()))
    if limits[0] == limits[1]:
        return limits[0] - 0.5, limits[1] + 0.5
    return limits


def _group_palette(n_levels: int) -> list[Any]:
    """One colour per group level, from a qualitative palette."""
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("Dark2")
    return [palette(index % palette.N) for index in range(n_levels)]


def _draw(
    *,
    drawn: np.ndarray,
    row_labels: list[str] | None,
    col_labels: list[str] | None,
    feat_tree: Clustering | None,
    sample_tree: Clustering | None,
    limits: tuple[float, float],
    n_colors: int,
    strip: list[Any] | None,
    group_colors: dict[str, Any] | None,
    anno: bool,
    cex_anno: float,
    cex_axis: float,
    cex_main: float,
    cex_legend: float,
    main: str | None,
) -> None:
    """Lay the panels out and draw the cells, the trees, the strip and the key.

    The panels are sized in inches and then handed to the gridspec as ratios, so
    that a label, a tree or a key gets the space it needs rather than a fixed
    fraction of the figure: the same arithmetic R does with ``margins``, and for
    the same reason - a strip reserved by guesswork either cuts the labels off or
    leaves a gap between the cells and their key.
    """
    fig = figure()
    ramp = LinearSegmentedColormap.from_list(
        "statassist_diverging", _RAMP, N=n_colors
    ).with_extremes(bad=_MISSING_COLOR)

    fig_width, fig_height = fig.get_size_inches()
    char = font(cex_axis) / 72 * 0.6
    label_in = _text_inches(row_labels, char) if row_labels else 0.05
    key_in = _KEY_BAR_INCHES + _text_inches(
        list(group_colors) if group_colors else [], font(cex_legend) / 72 * 0.6
    )
    tree_in = _TREE_INCHES if feat_tree is not None else 0.0
    cells_in = max(1.0, fig_width - _EDGE_INCHES * 2 - tree_in - label_in - key_in)

    widths = [width for width in (tree_in, cells_in, label_in, key_in) if width > 0]
    heights = [1.0]
    if strip is not None:
        heights.insert(0, 0.12)
    if sample_tree is not None:
        heights.insert(0, 0.45)

    grid = fig.add_gridspec(
        nrows=len(heights),
        ncols=len(widths),
        height_ratios=heights,
        width_ratios=widths,
        hspace=0.03,
        wspace=0.0,
    )
    cell_col = 1 if feat_tree is not None else 0
    cell_row = len(heights) - 1
    bottom_in = _text_inches(col_labels, font(cex_axis) / 72) if col_labels else _EDGE_INCHES
    fig.subplots_adjust(
        left=_EDGE_INCHES / fig_width,
        right=1 - _EDGE_INCHES / fig_width,
        bottom=min(0.5, bottom_in / fig_height),
        top=1 - (0.4 if main is not None else _EDGE_INCHES) / fig_height,
    )
    cells = fig.add_subplot(grid[cell_row, cell_col])

    image = cells.imshow(
        np.ma.masked_invalid(drawn),
        aspect="auto",
        origin="upper",
        cmap=ramp,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
    )

    cells.set_xticks(np.arange(drawn.shape[1]))
    cells.set_yticks(np.arange(drawn.shape[0]))
    cells.set_xticklabels(
        col_labels if col_labels is not None else [],
        rotation=90,
        fontsize=font(cex_axis),
    )
    cells.set_yticklabels(row_labels if row_labels is not None else [], fontsize=font(cex_axis))
    if col_labels is None:
        cells.tick_params(axis="x", length=0)
    if row_labels is None:
        cells.tick_params(axis="y", length=0)
    cells.yaxis.tick_right()
    for side in ("top", "right", "left", "bottom"):
        cells.spines[side].set_visible(False)

    if anno:
        _annotate(cells, drawn, limits, cex_anno)

    if sample_tree is not None:
        top = fig.add_subplot(grid[0, cell_col])
        _draw_tree(top, sample_tree, horizontal=False)
    if feat_tree is not None:
        side_panel = fig.add_subplot(grid[cell_row, 0])
        _draw_tree(side_panel, feat_tree, horizontal=True)
    if strip is not None:
        row = 1 if sample_tree is not None else 0
        band = fig.add_subplot(grid[row, cell_col], sharex=cells)
        # The colours are already one per column, in the order the columns are
        # drawn, so they go in as an image rather than through a colour map.
        band.imshow(
            np.asarray(strip, dtype=float).reshape(1, len(strip), -1),
            aspect="auto",
            interpolation="nearest",
        )
        band.set_xticks([])
        band.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            band.spines[side].set_visible(False)

    # The key sits in the strip reserved for it, the bar at the top so that it
    # starts level with the first row of cells and the group legend under it. The
    # bar carries no title: what the colours measure is `scale`, which is an
    # argument rather than something the plot can be asked.
    key = grid[cell_row, len(widths) - 1].subgridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[0.25, 1], hspace=0.05, wspace=0.4
    )
    ticks = pretty(limits[0], limits[1], n=3)
    bar_ax = fig.add_subplot(key[0, 0])
    bar = fig.colorbar(image, cax=bar_ax, ticks=ticks[(ticks >= limits[0]) & (ticks <= limits[1])])
    bar.ax.tick_params(labelsize=font(cex_axis))
    bar.outline.set_visible(False)

    if group_colors:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=colour, label=level) for level, colour in group_colors.items()]
        legend_ax = fig.add_subplot(key[1, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            title="group",
            loc="upper left",
            frameon=False,
            fontsize=font(cex_legend),
            title_fontsize=font(cex_legend),
        )
    if main is not None:
        fig.suptitle(main, fontsize=font(cex_main))


def _text_inches(labels: list[str] | None, per_char: float) -> float:
    """How much room a set of labels needs, in inches.

    Measured from the longest of them at the size it is drawn at, which is what
    ``strwidth()`` answers in R. A character is taken as six tenths of the font
    size, the usual approximation for a proportional face; it is only used to
    reserve space, and a panel is never made narrower than what is left over.
    """
    if not labels:
        return 0.0
    return max(len(label) for label in labels) * per_char + 0.15


def _draw_tree(ax: Any, tree: Clustering, horizontal: bool) -> None:
    """Draw one dendrogram beside the cells it ordered.

    :func:`scipy.cluster.hierarchy.dendrogram` puts leaf *i* at
    ``10 * i + 5``, so the leaf axis is set to ``0`` to ``10 * n`` to line the
    leaves up with the middle of the cells they belong to. The feature axis runs
    downwards, since the cells are drawn with the first feature at the top.
    """
    n_leaves = tree.order.size
    dendrogram(
        tree.linkage,
        ax=ax,
        orientation="left" if horizontal else "top",
        color_threshold=0,
        above_threshold_color="black",
        no_labels=True,
    )
    if horizontal:
        ax.set_ylim(10 * n_leaves, 0)
    else:
        ax.set_xlim(0, 10 * n_leaves)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _annotate(ax: Any, drawn: np.ndarray, limits: tuple[float, float], cex_anno: float) -> None:
    """Write rounded values on the cells.

    Port of ``sa_annotate_heatmap_cells()``: mid-range values sit on the pale
    part of the ramp and take dark text, the saturated ends take white. R sizes
    these labels off the axis labels; here they are sized off the cell, so a
    number stays inside the cell it belongs to however many cells there are.
    """
    span = limits[1] - limits[0]
    cell_inches = ax.get_position().width * ax.get_figure().get_size_inches()[0] / drawn.shape[1]
    size = min(font(cex_anno), cell_inches * 72 / (0.6 * _ANNO_CHARS))
    for i in range(drawn.shape[0]):
        for j in range(drawn.shape[1]):
            value = drawn[i, j]
            if not np.isfinite(value):
                continue
            position = (value - limits[0]) / span if span > 0 else 0.5
            colour = "#262626" if 0.28 < position < 0.72 else "white"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=size,
            )
