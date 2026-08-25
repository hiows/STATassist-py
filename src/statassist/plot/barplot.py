"""Clusters of bars, one per feature and group level, with a legend beside them.

Port of ``R/draw_grouped_barplot.R``. The summary counterpart of
:func:`~statassist.draw_grouped_boxplot`: a box shows the distribution a group's
observations have, a bar shows one number standing for them.

The heights are one column of :func:`~statassist.summarize_descriptive_stats`,
read from the same wide input the comparison functions take, so a bar and a row of
that table are the same number and neither has to be recomputed to check the
other.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaValueError, notify
from ..core.validate import (
    check_flag,
    check_lim,
    check_scalar_num,
    control_first,
    validate_wide_input,
)
from ..summarize.descriptive import levels_present, summarize_descriptive_stats
from ._theme import figure, font, group_colors, linestyle, theme, tick_rotation

__all__ = ["BAR_ERRORBARS", "BAR_HEIGHTS", "draw_grouped_barplot"]

#: The summary columns a bar height may be, R's ``mainbar`` choices.
BAR_HEIGHTS = (
    "mean",
    "median",
    "n",
    "n_missing",
    "sd",
    "var",
    "se",
    "cv",
    "mad",
    "skewness",
    "excess_kurtosis",
)

#: What the bars either side of a height may be.
BAR_ERRORBARS = ("none", "se", "sd", "ci")

#: Half-width of the median's interval, as a multiple of ``IQR / sqrt(n)``.
#:
#: The notch of a box plot, fixed at the value that makes it an approximate 95%
#: interval, which is why ``mainbar = "median"`` reads no ``conf_level``. Shared
#: with :func:`~statassist.draw_grouped_boxplot`, so a bar and the notch of the
#: box beside it are the same width on the same data.
NOTCH_WIDTH = 1.58

#: How much room is left past the furthest bar, as a fraction of the range drawn.
_HEADROOM = 0.04

#: How far a panel reaches either side of a set of bars that are all one height,
#: as a fraction of that height. A flat line still needs a panel to lie in.
_FLAT_PAD = 0.1

#: How wide a bar is drawn, as a fraction of the one unit between neighbours.
_BAR_WIDTH = 0.9

#: The horizontal grid, R's ``grid_col``: ``gray80`` on dark, ``gray40`` on light.
_GRID_COLOR = {True: "#CCCCCC", False: "#666666"}

#: How wide the bars are against the legend beside them, R's ``layout()`` widths.
_PANEL_PARTS = 4.0
_LEGEND_PARTS = 1.0


class BarInput(NamedTuple):
    """The bars, and the summary whose rows they are.

    Attributes:
        feats: Features in display order along the x axis.
        lv: Group levels in draw order, which is what the legend lists.
        summ: The descriptive summary, one row per bar.
    """

    feats: list[str]
    lv: list[str]
    summ: pd.DataFrame


def draw_grouped_barplot(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
    control_label: Any = None,
    mainbar: str = "mean",
    errorbar: str = "none",
    conf_level: float = 0.95,
    gap: float = 1.0,
    lwd: float = 1.5,
    col: Any = None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    ylim: Any = None,
    dark: bool = False,
    grid_lty: Any = 1,
    grid_lwd: float = 0.25,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    out_statistics: bool = True,
) -> pd.DataFrame | None:
    """Draw a grouped barplot of a descriptive summary.

    Args:
        data: Wide format, one row per observation and one column per feature.
        feats: Numeric column names to plot, in display order along the x axis.
        group: Grouping vector with one entry per row of ``data``. Required: a
            summary of every row together is what
            :func:`~statassist.summarize_descriptive_stats` returns without one,
            and it has no clusters to draw.
        group_lv: At least two group levels, in the order they should appear
            inside each cluster. Rows belonging to any other level are dropped.
            ``None`` takes the levels from ``group`` the way the summary does.
        control_label: The level to hold as the reference. It moves to the front
            of ``group_lv`` and the rest keep the order they were given, the same
            move :func:`~statassist.compare_two_groups` makes when it is passed
            the same argument.
        mainbar: Which summary column the bar heights are, one of
            :data:`BAR_HEIGHTS`.
        errorbar: What the bars either side of a height are, one of
            :data:`BAR_ERRORBARS`. Read under ``mainbar``, which decides which of
            them the height has an answer for.
        conf_level: Confidence level of ``errorbar="ci"``, read only under
            ``mainbar="mean"``. The median's interval is the notch, whose width
            is fixed, so there is no level for it to be stated at.
        gap: Blank bar widths inserted between neighbouring clusters.
        lwd: Line width of the error bars and of the baseline when the heights run
            both ways.
        col: Fill colours for the group levels, recycled if short. ``None`` takes
            the palette the boxes and the interaction traces are drawn in.
        xlab, ylab, main: Axis and title labels. ``ylab`` left ``None`` names
            ``mainbar``, since the axis of a bar chart is a particular statistic
            rather than the measurement itself; pass ``""`` for no label.
        ylim: Length-2 y range, or ``None`` to derive one that covers the bars,
            their intervals and the zero they are measured from.
        dark: If ``True``, use a dark background with light text.
        grid_lty: Line type of the horizontal grid, R's ``lty``.
        grid_lwd: Line width of the horizontal grid.
        cex_lab, cex_axis, cex_main, cex_legend: Character expansion for the axis
            labels, the axis annotation, the title and the legend.
        out_statistics: If ``True``, return the numbers behind the bars.

    Returns:
        ``None`` when ``out_statistics`` is ``False``. Otherwise one row per bar
        in the order they were drawn, which is the row order of
        :func:`~statassist.summarize_descriptive_stats`: a feature's levels stay
        together, in ``group_lv`` order.

        ``features``, ``group``
            Which bar the row is.
        ``n``
            Finite observations the bar was computed from.
        ``value``
            The bar height, the ``mainbar`` column.
        ``lower``, ``upper``
            The ends of the interval, missing under ``errorbar="none"`` and for a
            bar whose interval was not defined.

        Which height and which interval were drawn are in ``attrs["mainbar"]``
        and ``attrs["errorbar"]``.

    Raises:
        SaValueError: If ``group`` is missing, if an argument is out of range, if
            ``errorbar`` is one the height has no answer for, or if ``mainbar`` is
            missing for every bar.

    Notes:
        Only two of the summary columns are locations that an interval either side
        says something about, so ``errorbar`` is read under ``mainbar`` rather
        than independently of it. ``"mean"`` takes every bar; ``"median"`` takes
        ``"ci"`` only, the notch interval; everything else takes ``"none"``,
        being itself a spread, a count or a shape. A refused combination is an
        error rather than a silently dropped interval, because the alternative is
        a figure that answers a question other than the one it was asked.

        A bar is read against the zero it stands on, so a derived ``ylim`` always
        includes zero and puts its headroom on the side the bars run to. A bar
        whose height is missing leaves a blank rather than shifting the ones
        beside it.

    References:
        McGill, R., Tukey, J. W. and Larsen, W. A. (1978). Variations of box
        plots. *The American Statistician*, 32(1), 12-16.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import draw_grouped_barplot, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=4, n_up=1, n_down=1, seed=3)
        >>> bars = draw_grouped_barplot(
        ...     sim.args["data"],
        ...     sim.args["feats"],
        ...     sim.args["group"],
        ...     sim.args["group_lv"],
        ...     errorbar="se",
        ... )
        >>> list(bars.columns)
        ['features', 'group', 'n', 'value', 'lower', 'upper']
    """
    if mainbar not in BAR_HEIGHTS:
        raise SaValueError("`mainbar` must be one of: " + ", ".join(BAR_HEIGHTS) + ".")
    if errorbar not in BAR_ERRORBARS:
        raise SaValueError("`errorbar` must be one of: " + ", ".join(BAR_ERRORBARS) + ".")
    _bar_check_pair(mainbar, errorbar)

    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
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

    bars = _bar_input(data, feats, group, group_lv, control_label)
    drawn = _bar_values(bars, mainbar, errorbar, conf_level)

    _bar_draw(
        drawn,
        bars,
        mainbar=mainbar,
        errorbar=errorbar,
        gap=gap,
        lwd=lwd,
        col=col,
        xlab=xlab,
        ylab=ylab,
        main=main,
        limits=limits,
        dark=dark,
        grid_lty=grid_lty,
        grid_lwd=grid_lwd,
        cex_lab=cex_lab,
        cex_axis=cex_axis,
        cex_main=cex_main,
        cex_legend=cex_legend,
    )

    if not out_statistics:
        return None
    drawn.attrs["mainbar"] = mainbar
    drawn.attrs["errorbar"] = errorbar
    return drawn


def _bar_check_pair(mainbar: str, errorbar: str) -> None:
    """Refuse an interval the height has no answer for.

    Port of ``sa_bar_check_pair()``. Checked before anything is read off ``data``,
    so a call that could only produce a misleading figure fails at the boundary
    rather than after the summary has been computed.
    """
    if errorbar == "none" or mainbar == "mean":
        return

    if mainbar == "median":
        if errorbar == "ci":
            return
        raise SaValueError(
            f'`errorbar = "{errorbar}"` describes the spread of the observations '
            "about their mean, so it is not a width to draw either side of a "
            'median. `mainbar = "median"` takes errorbar = "ci", the notch '
            'interval median +/- 1.58 * IQR / sqrt(n), or "none".'
        )

    raise SaValueError(
        f'`mainbar = "{mainbar}"` is itself a spread, a count or a shape, so there '
        "is no second quantity for an interval either side of it to be about. Only "
        '"mean" and "median" take an `errorbar`; this height takes '
        'errorbar = "none".'
    )


def _bar_input(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    control_label: Any,
) -> BarInput:
    """Resolve the bars and summarise them.

    Port of ``sa_bar_input()``. The summary is asked for the levels that survived
    the validation rather than for the ones the caller named, so it has nothing
    left to drop and does not report the same rows a second time.
    """
    if group is None:
        raise SaValueError(
            "`group` says which bars there are, so it is required: one entry per row "
            "of `data`. A summary of every row together, with no clusters to draw, "
            "is what summarize_descriptive_stats() returns without one."
        )
    if group_lv is None:
        group_lv = levels_present(group)
    group_lv = control_first(group_lv, control_label)

    validated = validate_wide_input(data, feats, group, group_lv, min_levels=2)
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")
    assert validated.group is not None  # a level list was supplied, so there is a group
    levels = [str(level) for level in validated.group.categories]

    return BarInput(
        feats=list(validated.feats),
        lv=levels,
        summ=summarize_descriptive_stats(validated.data, validated.feats, validated.group, levels),
    )


def _bar_values(bars: BarInput, mainbar: str, errorbar: str, conf_level: float) -> pd.DataFrame:
    """One row per bar: its height and the ends of its interval.

    Port of ``sa_bar_values()``.
    """
    summ = bars.summ
    lower, upper = _bar_interval(summ, mainbar, errorbar, conf_level)
    return pd.DataFrame(
        {
            "features": summ["features"].astype(str),
            "group": summ["group"].astype(str),
            "n": summ["n"].to_numpy(dtype=float),
            "value": summ[mainbar].to_numpy(dtype=float),
            "lower": lower,
            "upper": upper,
        }
    )


def _bar_interval(
    summ: pd.DataFrame, mainbar: str, errorbar: str, conf_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Half-widths either side of the height, per bar.

    Port of ``sa_bar_interval()``. Every quantity involved is already a column of
    the summary, so the interval is arithmetic on the table rather than a second
    pass over the observations.
    """
    n_bars = len(summ.index)
    if errorbar == "none":
        blank = np.full(n_bars, np.nan)
        return blank, blank.copy()

    counts = summ["n"].to_numpy(dtype=float)
    if errorbar == "se":
        half = summ["se"].to_numpy(dtype=float)
    elif errorbar == "sd":
        half = summ["sd"].to_numpy(dtype=float)
    elif mainbar == "mean":
        # A single observation has no interval, and asking for one would put
        # Student's quantile on zero degrees of freedom.
        half = np.where(
            counts > 1,
            stats.t.ppf(1 - (1 - conf_level) / 2, np.maximum(counts - 1, 1))
            * summ["se"].to_numpy(dtype=float),
            np.nan,
        )
    else:
        half = NOTCH_WIDTH * summ["iqr"].to_numpy(dtype=float) / np.sqrt(counts)

    centre = summ[mainbar].to_numpy(dtype=float)
    return centre - half, centre + half


def _bar_span(
    drawn: pd.DataFrame, mainbar: str, errorbar: str, limits: tuple[float, float] | None
) -> tuple[float, float]:
    """The y range the bars, their intervals and their baseline need.

    Port of ``sa_bar_span()``.
    """
    if not np.isfinite(drawn["value"].to_numpy(dtype=float)).any():
        raise SaValueError(
            f'`mainbar = "{mainbar}"` is NA for every feature and group, so there is '
            "no bar to draw. summarize_descriptive_stats() returns the same column: a "
            "shape estimate needs three or four observations, and every statistic "
            "needs one."
        )
    if limits is not None:
        return limits

    # Zero goes in whether or not a bar reaches it, since it is what the height of
    # a bar is measured from.
    parts = [np.zeros(1), drawn["value"].to_numpy(dtype=float)]
    if errorbar != "none":
        parts += [drawn["lower"].to_numpy(dtype=float), drawn["upper"].to_numpy(dtype=float)]
    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())

    if high == low:
        # Every bar sits on the baseline, which needs a panel to be a flat line in.
        pad = max(abs(low), 1.0) * _FLAT_PAD
        return low - pad, high + pad

    # Headroom on the side the bars run to and none on the baseline, which a bar
    # has to stand on rather than float above.
    reach = high - low
    return (
        low - (_HEADROOM * reach if low < 0 else 0.0),
        high + (_HEADROOM * reach if high > 0 else 0.0),
    )


def _bar_draw(
    drawn: pd.DataFrame,
    bars: BarInput,
    *,
    mainbar: str,
    errorbar: str,
    gap: float,
    lwd: float,
    col: Any,
    xlab: str | None,
    ylab: str | None,
    main: str | None,
    limits: tuple[float, float] | None,
    dark: bool,
    grid_lty: Any,
    grid_lwd: float,
    cex_lab: float,
    cex_axis: float,
    cex_main: float,
    cex_legend: float,
) -> None:
    """Draw the clusters of bars and the legend beside them.

    Port of ``sa_bar_draw()``.
    """
    from matplotlib.patches import Patch

    n_lv = len(bars.lv)
    span = _bar_span(drawn, mainbar, errorbar, limits)
    colours = group_colors(col, n_lv)
    look = theme(dark)
    if ylab is None:
        ylab = mainbar

    # The rows of `drawn` hold a feature's levels together, so the position of a
    # bar is its cluster's start plus its level's place in it - the same order the
    # summary was built in, which is what lets the two be read off each other.
    at = np.concatenate(
        [
            index * (n_lv + gap) + np.arange(1, n_lv + 1, dtype=float)
            for index in range(len(bars.feats))
        ]
    )

    fig = figure()
    fig.patch.set_facecolor(look.bg)
    grid = fig.add_gridspec(1, 2, width_ratios=[_PANEL_PARTS, _LEGEND_PARTS], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_facecolor(look.bg)

    style = linestyle(grid_lty)
    if style is not None and grid_lwd > 0:
        ax.yaxis.grid(True, linestyle=style, color=_GRID_COLOR[dark], linewidth=grid_lwd)
    ax.set_axisbelow(True)

    heights = drawn["value"].to_numpy(dtype=float)
    # A bar is measured from zero, so a height below it is drawn downwards from
    # zero rather than up from the floor of the panel.
    ax.bar(
        at,
        heights,
        width=_BAR_WIDTH,
        color=[colours[index % n_lv] for index in range(len(at))],
        linewidth=0,
    )

    if errorbar != "none":
        lower = drawn["lower"].to_numpy(dtype=float)
        upper = drawn["upper"].to_numpy(dtype=float)
        has_bar = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        if has_bar.any():
            ax.errorbar(
                at[has_bar],
                heights[has_bar],
                yerr=[heights[has_bar] - lower[has_bar], upper[has_bar] - heights[has_bar]],
                fmt="none",
                ecolor=look.fg,
                elinewidth=lwd,
                capsize=3,
            )

    # A baseline is only worth drawing where it is not already the floor of the
    # panel: bars that run both ways need the zero they are measured from.
    if span[0] < 0:
        ax.axhline(0, color=look.fg, linewidth=lwd)

    ax.set_xlim(float(at[0]) - 1, float(at[-1]) + 1)
    ax.set_ylim(span)
    ax.set_xticks(
        [float(np.mean(at[index * n_lv : (index + 1) * n_lv])) for index in range(len(bars.feats))]
    )
    tilt = tick_rotation(
        bars.feats,
        ax.get_position().width * fig.get_size_inches()[0] / len(bars.feats),
        font(cex_axis),
    )
    ax.set_xticklabels(
        bars.feats,
        fontsize=font(cex_axis),
        rotation=tilt,
        ha="right" if tilt else "center",
        rotation_mode="anchor" if tilt else None,
    )
    ax.tick_params(axis="x", colors=look.fg, length=0)
    ax.tick_params(axis="y", colors=look.fg, labelsize=font(cex_axis))
    if xlab is not None:
        ax.set_xlabel(xlab, color=look.fg, fontsize=font(cex_lab))
    ax.set_ylabel(ylab, color=look.fg, fontsize=font(cex_lab))
    if main is not None:
        ax.set_title(main, color=look.fg, fontsize=font(cex_main))
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.set_axis_off()
    legend_ax.set_facecolor(look.bg)
    legend_ax.legend(
        handles=[
            Patch(facecolor=colours[index], label=str(level)) for index, level in enumerate(bars.lv)
        ],
        loc="center",
        frameon=False,
        fontsize=font(cex_legend),
        labelcolor=look.fg,
    )
