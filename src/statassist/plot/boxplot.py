"""Clusters of boxes across several features, with a legend beside them.

Port of ``R/draw_grouped_boxplot.R``. One cluster per feature for a single factor;
for a crossed design one panel per feature with the remaining factors along the x
axis, so the crossing sits inside a panel where an interaction can be seen.

The cells of a crossed design come from :func:`~statassist.core.fact_layout`, the
same helper :func:`~statassist.compare_factorial_groups` lays its observations out
with, so a box of this plot and a row of that result are the same observations and
the columns of the returned statistics are the cell labels both key on.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.factorial import fact_cell_index, fact_cell_labels, fact_grid, fact_layout
from ..core.tables import na_row
from ..core.validate import (
    check_count,
    check_flag,
    check_lim,
    check_scalar_num,
    validate_wide_input,
)
from ._theme import figure, font, group_colors, linestyle, theme, tick_rotation

__all__ = ["BOX_PANEL_AXES", "draw_grouped_boxplot"]

#: Which axis the panels of a crossed design run over.
BOX_PANEL_AXES = ("feature", "factor")

#: How far past a quartile a Tukey fence reaches, as a multiple of the IQR.
#:
#: Fixed by the definition of the box plot rather than open to a caller: the
#: returned ``lower_bound`` and ``upper_bound`` are that definition, not the
#: whisker ends the plot happens to draw.
WHISKER_REACH = 1.5

#: Half-width of the notch, as a multiple of ``IQR / sqrt(n)``.
#:
#: Fixed at the value that makes the notch an approximate 95% interval for the
#: median (McGill, Tukey and Larsen 1978), which is why it takes no level.
NOTCH_WIDTH = 1.58

#: Rows of ``box_summary_stats``, in order.
BOX_ROWS = ("min", "lower_bound", "Q1", "median", "Q3", "upper_bound", "max")

#: Rows of ``median_confidence_stats``, in order.
CONF_ROWS = ("n", "lower_conf", "upper_conf")

#: How wide a box is drawn, as a fraction of the one unit between neighbours.
_BOX_WIDTH = 0.7

#: What a box is filled with, R's ``bg_cols``: a slate that reads as unfilled on a
#: dark background, white on a light one.
_BOX_FILL = {True: "#36454F", False: "white"}

#: The horizontal grid, R's ``grid_col``: ``gray80`` on dark, ``gray40`` on light.
_GRID_COLOR = {True: "#CCCCCC", False: "#666666"}

#: How wide the panels are against the legend beside them, R's ``layout()``
#: widths: the panels share four parts and the legend takes the fifth.
_PANEL_PARTS = 4.0
_LEGEND_PARTS = 1.0

#: Space between neighbouring panels, as a fraction of a panel's width: enough
#: for a y axis and its label where every panel carries one, and only enough to
#: keep the boxes apart where the first column carries it for all of them.
_FREE_GUTTER = 0.5
_SHARED_GUTTER = 0.15


class BoxGroup(NamedTuple):
    """One group of cells: the boxes of a cluster, or of a panel.

    Attributes:
        label: What a strip would say, or ``None`` for a single factor, where
            there is nothing a strip would say that the legend does not.
        cols: Which boxes of a feature this group covers, in the order the levels
            of the primary factor are drawn in.
    """

    label: str | None
    cols: list[int]


class BoxInput(NamedTuple):
    """What the boxes are, however the caller said it.

    Port of what ``sa_box_input()`` returns. ``group`` and ``factors`` are two
    ways of saying the same thing, and both come out as this: a box per feature
    and per cell of the design, with the primary factor inside a cluster.

    Attributes:
        feats: Features in display order.
        lv: The levels inside a cluster, which is what the legend lists.
        box_labels: One label per box of a feature, which become the columns of
            the returned statistics.
        samples: Per feature, one array of values per box, aligned with
            ``box_labels``.
        groups: One entry per combination of the factors after the first.
        legend_title: The primary factor's name, or ``None`` for a single factor.
    """

    feats: list[str]
    lv: list[str]
    box_labels: list[str]
    samples: dict[str, list[np.ndarray]]
    groups: list[BoxGroup]
    legend_title: str | None


class BoxPanel(NamedTuple):
    """One panel: its strip, its x axis annotation and the boxes in it."""

    label: str | None
    cluster_labels: list[str]
    boxes: list[np.ndarray]


def draw_grouped_boxplot(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
    factors: Any = None,
    factor_lv: Any = None,
    control_label: Any = None,
    panel_by: str = "feature",
    panel_nrow: int | None = None,
    gap: float = 1.0,
    lwd: float = 1.5,
    xlab: str | None = None,
    ylab: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    ylim: Any = None,
    main: str | None = None,
    dark: bool = False,
    grid_lty: Any = 1,
    grid_lwd: float = 0.25,
    cex_legend: float = 1.1,
    out_statistics: bool = True,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """Draw a grouped boxplot across several features.

    Args:
        data: Wide format, one row per observation and one column per feature.
        feats: Numeric column names to plot, in display order along the x axis.
        group: Grouping vector with one entry per row of ``data``, for a single
            factor. Leave it ``None`` and name ``factors`` for a crossed design.
        group_lv: At least two group levels, in the order they should appear
            inside each cluster. Rows belonging to any other level are dropped.
        factors: The crossed factors, each entry either the name of a column of
            ``data`` or one value per row of it, exactly as
            :func:`~statassist.compare_factorial_groups` takes them. There have
            to be at least two: one factor is ``group``. The **first factor is
            the primary one**, whose levels are the coloured boxes inside a
            cluster and the entries of the legend.
        factor_lv: The levels of each factor, the reference first, or ``None`` to
            take them from the data in sorted order.
        control_label: The level to hold as the reference, one name per factor it
            points at. Read only under ``factors``: a single factor draws in
            ``group_lv`` order, which already says where its reference is.
        panel_by: Which axis the panels are over, read only under ``factors``.
            One of :data:`BOX_PANEL_AXES`: ``"feature"`` for one panel per
            feature with the remaining factors along the x axis, or ``"factor"``
            for one panel per combination of the remaining factors with the
            features along the x axis.
        panel_nrow: How many rows the panels are laid out in, or ``None`` to let
            the arrangement decide: one row under ``panel_by="factor"``, and a
            grid as near square as the panel count allows under ``"feature"``.
        gap: Blank box widths inserted between neighbouring clusters.
        lwd: Line width of the boxes.
        xlab, ylab, main: Axis and title labels.
        cex_lab, cex_axis, cex_main, cex_legend: Character expansion for the axis
            labels, the axis annotation, the title and the legend.
        ylim: Length-2 y range shared by every panel, or ``None`` to derive it.
            Panels over the factors hold the same features and share one range
            taken from every value drawn. Panels over the features hold different
            quantities, each with its own baseline, so each keeps its own range
            and is annotated with its own axis.
        dark: If ``True``, use a dark background with light text.
        grid_lty: Line type of the horizontal grid, R's ``lty``.
        grid_lwd: Line width of the horizontal grid.
        out_statistics: If ``True``, return the summary statistics behind the
            boxes.

    Returns:
        ``None`` when ``out_statistics`` is ``False``. Otherwise a dict of two
        entries, each holding one :class:`~pandas.DataFrame` per feature:

        ``box_summary_stats``
            Rows :data:`BOX_ROWS` and one column per box: the group levels under
            ``group``, the cell labels of the design under ``factors``. The
            bounds are the Tukey fences ``Q1 - 1.5 * IQR`` and
            ``Q3 + 1.5 * IQR``, not the drawn whisker ends.
        ``median_confidence_stats``
            Rows :data:`CONF_ROWS`, the notch interval
            ``median +/- 1.58 * IQR / sqrt(n)``. ``n`` counts non-missing values.

    Raises:
        SaValueError: If neither or both of ``group`` and ``factors`` say what
            the boxes are, if an argument is out of range, or if the features
            hold no finite value for panels that have to share an axis.

    Notes:
        A crossed design has three categorical axes to place and two dimensions
        to place them in, so one of them has to become the panels, and which one
        decides whether the picture shows an interaction. ``panel_by="feature"``
        puts the two factors side by side in one panel, so an effect that
        reverses between the levels of the other factor is a pattern of colours
        that visibly flips - which is what an interaction is.
        ``panel_by="factor"`` is the transpose: every feature of one cell
        together, which is what to ask for when the question is about the
        features rather than about the crossing.

        Either way the boxes are the same boxes and the returned statistics are
        identical. A cell holding no observation leaves its box blank rather than
        shifting the ones beside it, and is reported in a message.

    References:
        McGill, R., Tukey, J. W. and Larsen, W. A. (1978). Variations of box
        plots. *The American Statistician*, 32(1), 12-16.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import draw_grouped_boxplot, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=4, n_up=1, n_down=1, seed=3)
        >>> stats = draw_grouped_boxplot(
        ...     sim.args["data"],
        ...     sim.args["feats"],
        ...     sim.args["group"],
        ...     sim.args["group_lv"],
        ... )
        >>> list(stats["box_summary_stats"][sim.args["feats"][0]].index)[:3]
        ['min', 'lower_bound', 'Q1']
    """
    if panel_by not in BOX_PANEL_AXES:
        raise SaValueError("`panel_by` must be one of: " + ", ".join(BOX_PANEL_AXES) + ".")
    if panel_nrow is not None:
        panel_nrow = check_count(panel_nrow, "panel_nrow", 1)
    gap = check_scalar_num(gap, "gap", 0)
    lwd = check_scalar_num(lwd, "lwd", 0, lower_open=True)
    cex_lab = check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    grid_lwd = check_scalar_num(grid_lwd, "grid_lwd", 0)
    dark = check_flag(dark, "dark")
    out_statistics = check_flag(out_statistics, "out_statistics")
    limits = check_lim(ylim, "ylim")

    boxes = _box_input(data, feats, group, group_lv, factors, factor_lv, control_label)

    _box_draw(
        boxes,
        panel_by=panel_by,
        panel_nrow=panel_nrow,
        gap=gap,
        lwd=lwd,
        xlab=xlab,
        ylab=ylab,
        cex_lab=cex_lab,
        cex_axis=cex_axis,
        cex_main=cex_main,
        limits=limits,
        main=main,
        dark=dark,
        grid_lty=grid_lty,
        grid_lwd=grid_lwd,
        cex_legend=cex_legend,
    )

    if not out_statistics:
        return None

    # One pass per feature and box; the two returned tables are slices of it.
    summaries = {
        feat: pd.DataFrame(
            {
                label: _box_stats(values)
                for label, values in zip(boxes.box_labels, columns, strict=True)
            },
            columns=list(boxes.box_labels),
        )
        for feat, columns in boxes.samples.items()
    }
    return {
        "box_summary_stats": {feat: table.loc[list(BOX_ROWS)] for feat, table in summaries.items()},
        "median_confidence_stats": {
            feat: table.loc[list(CONF_ROWS)] for feat, table in summaries.items()
        },
    }


def _box_input(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    factors: Any,
    factor_lv: Any,
    control_label: Any,
) -> BoxInput:
    """Resolve what the boxes are, however the caller said it.

    Port of ``sa_box_input()``. The two paths meet here rather than in the
    drawing code, and a single factor comes out as a design with exactly one
    group of cells. What this does not decide is the layout: a group of cells is
    a fact about the design, not a panel.
    """
    said_group = group is not None or group_lv is not None
    said_factors = factors is not None or factor_lv is not None

    if said_group and said_factors:
        raise SaValueError(
            "`group` and `factors` are two ways of saying what the boxes are, so a "
            "call takes one of them: `group` with `group_lv` for a single factor, "
            "`factors` with `factor_lv` for a crossed design."
        )
    if not said_group and not said_factors:
        raise SaValueError(
            "nothing says what the boxes are. Supply `group` and `group_lv` for a "
            "single factor, or `factors` for a crossed design."
        )
    if factor_lv is not None and factors is None:
        raise SaValueError(
            "`factor_lv` gives the levels of the factors `factors` holds, which was not supplied."
        )
    # A single factor has one list of levels and `group_lv` is it, so a reference
    # named a second way would be a second place for the draw order to be decided.
    if said_group and control_label is not None:
        raise SaValueError(
            "`control_label` names a reference level per factor of a crossed design, "
            "which `factors` states. A single factor draws in `group_lv` order, so "
            "put the reference first there."
        )

    if said_group:
        return _box_one_factor(data, feats, group, group_lv)
    return _box_crossed(data, feats, factors, factor_lv, control_label)


def _box_one_factor(data: Any, feats: Any, group: Any, group_lv: Any) -> BoxInput:
    """The single-factor case, as a design of one group of cells.

    Port of ``sa_box_one_factor()``.
    """
    validated = validate_wide_input(data, feats, group, group_lv, min_levels=2)
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")
    assert validated.group is not None  # a level list was supplied, so there is a group
    levels = [str(level) for level in validated.group.categories]
    codes = np.asarray(validated.group.codes)

    samples = {
        feat: [
            validated.data[feat].to_numpy(dtype=float)[codes == position]
            for position in range(len(levels))
        ]
        for feat in validated.feats
    }
    return BoxInput(
        feats=list(validated.feats),
        lv=levels,
        box_labels=list(levels),
        samples=samples,
        # No label: with one factor there is nothing a strip would say that the
        # legend does not.
        groups=[BoxGroup(label=None, cols=list(range(len(levels))))],
        legend_title=None,
    )


def _box_crossed(
    data: Any,
    feats: Any,
    factors: Any,
    factor_lv: Any,
    control_label: Any,
) -> BoxInput:
    """The crossed case, one group of cells per combination of the later factors.

    Port of ``sa_box_crossed()``. The cells come from :func:`fact_layout`, which
    :func:`~statassist.compare_factorial_groups` calls with the same arguments,
    so the boxes are the cells the analysis fits and their labels are the ones
    the answer key uses. What is decided here is only how the cells are dealt out
    to the groups: the primary factor varies inside a cluster and the rest are
    read as a mixed-radix number, the same arithmetic that numbers the cells.
    """
    validated = validate_wide_input(data, feats, group=None, group_lv=None)
    frame = validated.data
    names = list(validated.feats)

    design = fact_layout(frame, factors, factor_lv, control_label)
    if design.n_dropped > 0:
        notify(f"Dropped {design.n_dropped} row(s) belonging to a level outside `factor_lv`.")
    if design.n_empty_cells > 0:
        empty = [
            label
            for label, count in zip(design.cell_label, design.cell_n, strict=True)
            if count == 0
        ]
        notify(
            f"{design.n_empty_cells} of {design.n_cells} cell(s) hold no observation, "
            "so their box is left blank: " + ", ".join(empty) + "."
        )

    primary = next(iter(design.factor_lv))
    levels = list(design.factor_lv[primary])
    rest = {name: design.factor_lv[name] for name in design.factor_lv if name != primary}
    group_of_cell = fact_cell_index(
        design.cells[list(rest)].to_numpy(dtype=np.int64),
        [len(rest[name]) for name in rest],
    )
    group_label = fact_cell_labels(rest, fact_grid(rest))
    own = np.asarray(design.cells[primary], dtype=np.int64)

    samples = {
        feat: [frame[feat].to_numpy(dtype=float)[rows] for rows in design.rows_of_cell]
        for feat in names
    }
    groups = []
    for index, label in enumerate(group_label):
        held = np.flatnonzero(group_of_cell == index)
        # Inside a group the primary factor varies, in its own level order.
        held = held[np.argsort(own[held], kind="stable")]
        groups.append(BoxGroup(label=label, cols=[int(cell) for cell in held]))
    return BoxInput(
        feats=names,
        lv=levels,
        box_labels=list(design.cell_label),
        samples=samples,
        groups=groups,
        legend_title=primary,
    )


def _box_arrange(boxes: BoxInput, panel_by: str) -> tuple[list[BoxPanel], bool]:
    """Deal the boxes out into panels, one layout or its transpose.

    Port of ``sa_box_arrange()``. :class:`BoxInput` leaves a feature-by-cell
    matrix of boxes; a panel is a choice of which way to read it. Both readings
    hold the same boxes and every panel of either holds the same number of
    clusters, which is what lets one drawing routine take both.

    Returns:
        The panels, and whether they hold different quantities and so cannot
        share a y axis.
    """
    # A single factor is one group of cells, so there is nothing to panel over
    # and nothing for `panel_by` to choose between: it reads as "factor" either
    # way.
    if panel_by == "feature" and len(boxes.groups) > 1:
        labels = [group.label for group in boxes.groups]
        panels = [
            BoxPanel(
                label=feat,
                cluster_labels=[str(label) for label in labels],
                boxes=[boxes.samples[feat][col] for group in boxes.groups for col in group.cols],
            )
            for feat in boxes.feats
        ]
        # Each feature has its own baseline, so a common axis would flatten all
        # of them; see the `ylim` documentation.
        return panels, True

    panels = [
        BoxPanel(
            label=group.label,
            cluster_labels=list(boxes.feats),
            boxes=[boxes.samples[feat][col] for feat in boxes.feats for col in group.cols],
        )
        for group in boxes.groups
    ]
    return panels, False


def _box_stats(values: np.ndarray) -> dict[str, float]:
    """The numbers behind one box.

    Port of ``sa_box_stats()``. A box with nothing in it gives an all-missing
    column rather than aborting the summary.
    """
    kept = values[~np.isnan(values)]
    if kept.size == 0:
        return na_row(BOX_ROWS + CONF_ROWS)

    q1, median, q3 = (float(value) for value in np.quantile(kept, (0.25, 0.5, 0.75)))
    iqr = q3 - q1
    notch = NOTCH_WIDTH * iqr / math.sqrt(kept.size)
    return {
        "min": float(kept.min()),
        "lower_bound": q1 - WHISKER_REACH * iqr,
        "Q1": q1,
        "median": median,
        "Q3": q3,
        "upper_bound": q3 + WHISKER_REACH * iqr,
        "max": float(kept.max()),
        "n": float(kept.size),
        "lower_conf": median - notch,
        "upper_conf": median + notch,
    }


def _box_positions(n_cluster: int, n_lv: int, gap: float) -> list[np.ndarray]:
    """Where each cluster's boxes sit, R's ``at``."""
    return [
        index * (n_lv + gap) + np.arange(1, n_lv + 1, dtype=float) for index in range(n_cluster)
    ]


def _box_span(boxes: BoxInput) -> tuple[float, float]:
    """The range panels of the same quantity share.

    Taken from every value drawn rather than from one panel: panels of the same
    quantity that scale to their own values cannot be read against each other.
    """
    values = np.concatenate([column for columns in boxes.samples.values() for column in columns])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SaValueError("`feats` hold no finite value in any cell, so there is nothing to draw.")
    return float(finite.min()), float(finite.max())


def _box_draw(
    boxes: BoxInput,
    *,
    panel_by: str,
    panel_nrow: int | None,
    gap: float,
    lwd: float,
    xlab: str | None,
    ylab: str | None,
    cex_lab: float,
    cex_axis: float,
    cex_main: float,
    limits: tuple[float, float] | None,
    main: str | None,
    dark: bool,
    grid_lty: Any,
    grid_lwd: float,
    cex_legend: float,
) -> None:
    """Draw the panels and the legend beside them.

    Port of ``sa_box_draw_panels()``. One panel is the whole picture for a single
    factor, so there is one drawing routine rather than one per mode, and both
    arrangements of a crossed design reach it as the same clusters and boxes. A y
    axis range the panels are meant to share has to be settled here for all of
    them at once.
    """
    from matplotlib.patches import Patch

    panels, free_panels = _box_arrange(boxes, panel_by)
    n_lv = len(boxes.lv)
    n_panel = len(panels)
    at = _box_positions(len(panels[0].cluster_labels), n_lv, gap)

    # Panels of the same quantity are separate axes, so each would otherwise
    # scale to its own values and the difference between two panels would be
    # invisible. Panels of different quantities are the opposite case: a shared
    # range flattens every one of them, so each keeps its own and says so on its
    # own axis. A lone panel is left to matplotlib, whose range carries padding
    # this one would not.
    free_scale = free_panels and limits is None
    if limits is None and not free_scale and n_panel > 1:
        limits = _box_span(boxes)

    if panel_nrow is None:
        # One panel per feature is many panels, and a single row of them is a
        # strip too thin to read, so the default shape is as near square as it
        # gets.
        panel_nrow = max(1, round(math.sqrt(n_panel))) if free_panels else 1
    n_row = min(panel_nrow, n_panel)
    n_col = math.ceil(n_panel / n_row)

    look = theme(dark)
    colours = group_colors(None, n_lv)
    strip = n_panel > 1
    # A title belongs to the figure rather than to a panel of it, so with panels
    # it moves above them and is written once.
    outer_main = strip and main is not None

    fig = figure()
    fig.patch.set_facecolor(look.bg)
    grid = fig.add_gridspec(
        n_row,
        n_col + 1,
        width_ratios=[_PANEL_PARTS / n_col] * n_col + [_LEGEND_PARTS],
        # A free scale is annotated by every panel rather than by the first
        # column alone, so the gutter has to hold an axis rather than nothing.
        wspace=_FREE_GUTTER if free_scale else _SHARED_GUTTER,
        hspace=0.35,
        top=0.86 if outer_main else 0.92,
    )

    for index, panel in enumerate(panels):
        at_col = index % n_col
        ax = fig.add_subplot(grid[index // n_col, at_col])
        # A shared scale is stated once, in the first column, so that the panels
        # can sit against each other; a free one has to be stated by every panel.
        y_annot = free_scale or at_col == 0
        _box_panel(
            ax,
            panel,
            at=at,
            colours=colours,
            n_lv=n_lv,
            lwd=lwd,
            xlab=xlab,
            ylab=ylab if y_annot else None,
            main=None if outer_main else main,
            limits=limits,
            look=look,
            dark=dark,
            grid_lty=grid_lty,
            grid_lwd=grid_lwd,
            cex_lab=cex_lab,
            cex_axis=cex_axis,
            cex_main=cex_main,
            y_annot=y_annot,
        )
        if strip and panel.label is not None:
            ax.set_title(panel.label, color=look.fg, fontsize=font(cex_axis), pad=6)

    legend_ax = fig.add_subplot(grid[:, -1])
    legend_ax.set_axis_off()
    legend_ax.set_facecolor(look.bg)
    legend = legend_ax.legend(
        handles=[
            Patch(facecolor=colours[index], edgecolor="white", label=str(level))
            for index, level in enumerate(boxes.lv)
        ],
        title=boxes.legend_title,
        loc="center",
        frameon=False,
        fontsize=font(cex_legend),
        title_fontsize=font(cex_legend),
        labelcolor=look.fg,
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_color(look.fg)

    if outer_main and main is not None:
        fig.suptitle(main, color=look.fg, fontsize=font(cex_main), fontweight="bold")


def _box_panel(
    ax: Any,
    panel: BoxPanel,
    *,
    at: list[np.ndarray],
    colours: list[Any],
    n_lv: int,
    lwd: float,
    xlab: str | None,
    ylab: str | None,
    main: str | None,
    limits: tuple[float, float] | None,
    look: Any,
    dark: bool,
    grid_lty: Any,
    grid_lwd: float,
    cex_lab: float,
    cex_axis: float,
    cex_main: float,
    y_annot: bool,
) -> None:
    """Draw one panel of clusters."""
    positions = np.concatenate(at)
    ax.set_facecolor(look.bg)
    style = linestyle(grid_lty)
    if style is not None and grid_lwd > 0:
        ax.yaxis.grid(True, linestyle=style, color=_GRID_COLOR[dark], linewidth=grid_lwd)
    ax.set_axisbelow(True)

    for index, values in enumerate(panel.boxes):
        kept = values[~np.isnan(values)]
        if kept.size == 0:
            # A cell with nothing in it leaves its place empty rather than
            # shifting the boxes beside it.
            continue
        colour = colours[index % n_lv]
        ax.boxplot(
            [kept],
            positions=[positions[index]],
            widths=_BOX_WIDTH,
            patch_artist=True,
            manage_ticks=False,
            boxprops={"facecolor": _BOX_FILL[dark], "edgecolor": colour, "linewidth": lwd},
            whiskerprops={"color": colour, "linewidth": lwd},
            capprops={"color": colour, "linewidth": lwd},
            medianprops={"color": colour, "linewidth": lwd},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": colour,
                "markeredgecolor": colour,
            },
        )

    ax.set_xlim(float(positions[0]) - 1, float(positions[-1]) + 1)
    if limits is not None:
        ax.set_ylim(limits)
    ax.set_xticks([float(np.mean(cluster)) for cluster in at])
    tilt = tick_rotation(
        panel.cluster_labels,
        ax.get_position().width * ax.get_figure().get_size_inches()[0] / len(at),
        font(cex_axis),
    )
    ax.set_xticklabels(
        panel.cluster_labels,
        fontsize=font(cex_axis),
        rotation=tilt,
        ha="right" if tilt else "center",
        rotation_mode="anchor" if tilt else None,
    )
    ax.tick_params(axis="x", colors=look.fg, length=0)
    ax.tick_params(axis="y", colors=look.fg, labelsize=font(cex_axis))
    if not y_annot:
        ax.set_yticklabels([])
    if xlab is not None:
        ax.set_xlabel(xlab, color=look.fg, fontsize=font(cex_lab))
    if ylab is not None:
        ax.set_ylabel(ylab, color=look.fg, fontsize=font(cex_lab))
    if main is not None:
        ax.set_title(main, color=look.fg, fontsize=font(cex_main))
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
