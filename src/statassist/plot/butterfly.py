"""The butterfly histogram: two groups of one feature, back to back.

Port of ``R/draw_butterfly_hist.R``. The first level runs left from the centre
line and the second one right, on shared breaks, so the two shapes can be
compared bin by bin instead of read off two separate panels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.validate import (
    check_flag,
    check_lim,
    check_margin,
    check_scalar_num,
    validate_wide_input,
)
from ._hist import BREAK_RULES, density, histogram, nclass, pretty
from ._theme import figure, font, set_margin

__all__ = ["BUTTERFLY_SCALES", "BUTTERFLY_TYPES", "draw_butterfly_hist"]

#: What a bar length can mean.
BUTTERFLY_SCALES = ("count", "proportion", "density")

#: Which layers can be drawn.
BUTTERFLY_TYPES = ("freq", "dens", "both")

#: What the bar axis is called, per scale.
_SCALE_LABELS = {
    "count": "Frequency",
    "proportion": "Proportion",
    "density": "Density",
}


def draw_butterfly_hist(
    data: Any,
    feat: str,
    group: Any,
    group_lv: Any,
    breaks: Any = "Sturges",
    scale: str | None = None,
    type: str = "freq",
    dens_adjust: float = 1.0,
    dens_lwd: float = 2.0,
    dens_col: Any = None,
    dens_alpha: float = 0.45,
    col: Any = ("#4575B4", "#D73027"),
    border: str = "white",
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    legend_position: Any = "upper right",
    xlim: Any = None,
    ylim: Any = None,
    margin: Any = (5, 5, 4, 3),
    out_statistics: bool = True,
) -> dict[str, Any] | None:
    """Draw the distribution of one feature for two group levels back to back.

    Args:
        data: Wide format, one row per observation and one column per feature.
        feat: Name of the numeric column to plot. One feature per call.
        group: Grouping vector with one entry per row of ``data``.
        group_lv: Exactly two group levels. The first is drawn on the left, the
            second on the right. Rows belonging to any other level are dropped.
        breaks: Break specification, shared by both groups. Accepts one of
            :data:`~statassist.plot._hist.BREAK_RULES`, a single number read as
            the approximate bin count, or a strictly increasing vector of break
            points.
        scale: What the bar length means, one of :data:`BUTTERFLY_SCALES`.
            ``None`` means ``"count"``, or ``"density"`` when a curve is drawn.
        type: Which layers to draw, one of :data:`BUTTERFLY_TYPES`: ``"freq"``
            for the bars alone, ``"dens"`` for the kernel density estimate alone,
            drawn as a filled shape, or ``"both"`` for the two overlaid.
        dens_adjust: Bandwidth multiplier. Values above 1 smooth further.
        dens_lwd: Line width of the density outline.
        dens_col: Colour of the density outline, one colour or one per level.
            ``None`` outlines each group in its own ``col``.
        dens_alpha: Opacity of the density fill, between 0 and 1. The default
            leaves the shape half transparent, so under ``type="both"`` the bars
            stay visible through it and the two layers read as one distribution
            rather than two.
        col: Two fill colours, for the left and the right group.
        border: Colour of the bar borders.
        xlab: Bar axis label, the meaning of ``scale`` by default.
        ylab: Value axis label, ``feat`` by default.
        main: Title.
        cex_lab: Character expansion for the axis labels.
        cex_axis: Character expansion for the axis annotation.
        cex_main: Character expansion for the title.
        cex_legend: Character expansion for the legend.
        legend_position: A matplotlib legend location, or ``None`` / ``False`` to
            leave the legend out.
        xlim: Length-2 range for the bar axis, or ``None`` to derive it.
        ylim: Length-2 range for the value axis, or ``None`` to derive it. A
            derived range covers the breaks, and the tails of the density
            estimate as well when one is drawn.
        margin: Plot margins in lines of text: bottom, left, top, right.
        out_statistics: If ``True``, return the numbers behind the bars.

    Returns:
        ``None`` if ``out_statistics`` is ``False``. Otherwise a dict:

        ``bin_summary_stats``
            One row per bin, with ``bin_start``, ``bin_end``, ``bin_mid`` and one
            column per group level holding the plotted bar length.
        ``group_summary_stats``
            One column per group level with rows ``n`` (finite values used),
            ``n_dropped`` (missing or non-finite values left out), ``min`` and
            ``max``.
        ``group_hists``
            One binned group per level, named by the level. Both groups are
            binned on the shared breaks, so these are not what binning one group
            alone would give.
        ``group_densities``
            Present only when ``type`` is not ``"freq"``. One kernel density
            estimate per level, named by the level.

    Raises:
        SaValueError: If an argument is out of range, if ``breaks`` is not a rule,
            a count or an increasing vector, if a level holds no finite value, or
            if there is nothing to draw.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import draw_butterfly_hist, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=4, n_up=1, n_down=1, seed=7)
        >>> out = draw_butterfly_hist(
        ...     sim.args["data"],
        ...     sim.args["feats"][0],
        ...     sim.args["group"],
        ...     sim.args["group_lv"],
        ... )
        >>> list(out["group_summary_stats"].index)
        ['n', 'n_dropped', 'min', 'max']
    """
    if type not in BUTTERFLY_TYPES:
        raise SaValueError("`type` must be one of " + ", ".join(BUTTERFLY_TYPES) + f". Got {type}.")
    if scale is not None and scale not in BUTTERFLY_SCALES:
        raise SaValueError(
            "`scale` must be one of " + ", ".join(BUTTERFLY_SCALES) + f". Got {scale}."
        )
    # A density curve and a bar can only be read against the same axis when the
    # bar is a density too: a count or a proportion per bin scales with the bin
    # width, which the curve knows nothing about.
    if type == "freq":
        scale = "count" if scale is None else scale
    elif scale is None:
        scale = "density"
    elif scale != "density":
        raise SaValueError(
            f'`type="{type}"` draws a kernel density estimate, which shares an axis '
            'with the bars only on the density scale; set `scale="density"`.'
        )

    dens_adjust = check_scalar_num(dens_adjust, "dens_adjust", 0, lower_open=True)
    dens_lwd = check_scalar_num(dens_lwd, "dens_lwd", 0, lower_open=True)
    dens_alpha = check_scalar_num(dens_alpha, "dens_alpha", 0, 1)
    cex_lab = check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    out_statistics = check_flag(out_statistics, "out_statistics")
    margins = check_margin(margin)
    x_limits = check_lim(xlim, "xlim")
    y_limits = check_lim(ylim, "ylim")

    fills = _two_colours(col, "col")
    if not isinstance(border, str):
        raise SaValueError("`border` must be a single colour.")
    outlines = fills if dens_col is None else _two_colours(dens_col, "dens_col")
    if not isinstance(feat, str):
        raise SaValueError(
            "`feat` must be a single column name; this plot shows one feature at a time."
        )

    validated = validate_wide_input(data, feat, group, group_lv, n_levels=2)
    if validated.group is None:  # pragma: no cover - two levels were required
        raise SaValueError("`group` and `group_lv` must both be supplied.")
    levels = [str(level) for level in validated.group.categories]
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    column = validated.data[feat].to_numpy(dtype=float)
    labels = np.asarray(validated.group)
    values = {level: column[labels == level] for level in levels}
    n_input = {level: values[level].size for level in levels}
    values = {level: block[np.isfinite(block)] for level, block in values.items()}
    n_used = {level: values[level].size for level in levels}

    empty = [level for level in levels if n_used[level] == 0]
    if empty:
        raise SaValueError(
            f"`{feat}` has no finite value in group level(s): " + ", ".join(empty) + "."
        )
    left_out = sum(n_input[level] - n_used[level] for level in levels)
    if left_out > 0:
        notify(f"Left out {left_out} missing or non-finite value(s) of `{feat}`.")

    pooled = np.concatenate([values[level] for level in levels])
    edges = _breaks(breaks, pooled)

    # Both groups go through the same breaks, so bin i means the same interval on
    # either side of the centre line.
    hists = {level: histogram(values[level], edges, xname=f"{feat} ({level})") for level in levels}
    densities = None
    if type != "freq":
        densities = {}
        for level in levels:
            block = values[level]
            # A single distinct value gives a zero bandwidth, which is not a
            # density; saying which group is at fault is the point of checking
            # here rather than letting the arithmetic fail.
            if np.unique(block).size < 2:
                raise SaValueError(
                    f"`{feat}` needs at least two distinct finite values in group "
                    f'level "{level}" to estimate a density.'
                )
            densities[level] = density(block, adjust=dens_adjust, data_name=f"{feat} ({level})")

    bar_length = {level: _bar_length(hists[level], scale) for level in levels}

    # Only the layers `type` actually draws set the extent, so a curve peak
    # taller than the tallest bar is not clipped and hidden bars do not pad the
    # axis.
    drawn: list[np.ndarray] = []
    if type != "dens":
        drawn.extend(bar_length.values())
    if type != "freq" and densities is not None:
        drawn.extend(curve["y"] for curve in densities.values())
    bar_max = float(np.concatenate(drawn).max())
    if not np.isfinite(bar_max) or bar_max <= 0:
        if type == "dens":
            raise SaValueError(
                "the density estimate is flat everywhere, so there is nothing to draw."
            )
        raise SaValueError("every bin is empty, so there is nothing to draw.")

    if x_limits is None:
        ticks = pretty(-bar_max, bar_max)
        x_limits = (float(ticks.min()), float(ticks.max()))
    else:
        ticks = pretty(*x_limits)
    if y_limits is None:
        # A density is evaluated past the data range by a few bandwidths, so the
        # breaks alone would cut the tails off the curve.
        reach = [edges]
        if densities is not None:
            reach.extend(curve["x"] for curve in densities.values())
        stacked = np.concatenate(reach)
        y_limits = (float(stacked.min()), float(stacked.max()))

    fig = figure()
    ax = fig.add_subplot()
    set_margin(fig, margins)

    # The left group is drawn at negative coordinates so that both distributions
    # share one axis, but the tick labels are the absolute values, so a bar is
    # read the same way on either side.
    if type != "dens":
        for index, level in enumerate(levels):
            side = -1.0 if index == 0 else 1.0
            lengths = bar_length[level]
            ax.barh(
                edges[:-1],
                side * lengths,
                height=np.diff(edges),
                align="edge",
                color=fills[index],
                edgecolor=border,
            )

    # The value is on the vertical axis here, so the shape goes in transposed:
    # the density becomes x, signed to put the first level left of the centre
    # line. The fill is translucent, so over bars this is the same shape laid on
    # top of them rather than a second thing to read.
    if densities is not None:
        for index, level in enumerate(levels):
            curve = densities[level]
            side = -1.0 if index == 0 else 1.0
            ax.fill_betweenx(
                curve["x"],
                0.0,
                side * curve["y"],
                facecolor=fills[index],
                alpha=dens_alpha,
                edgecolor=outlines[index],
                linewidth=dens_lwd,
            )

    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xticks(ticks[(ticks >= x_limits[0]) & (ticks <= x_limits[1])])
    shown = np.abs(ax.get_xticks())
    ax.set_xticklabels([f"{tick:.2f}" if scale == "proportion" else f"{tick:g}" for tick in shown])
    ax.tick_params(labelsize=font(cex_axis))
    ax.set_xlabel(_SCALE_LABELS[scale] if xlab is None else xlab, fontsize=font(cex_lab))
    ax.set_ylabel(feat if ylab is None else ylab, fontsize=font(cex_lab))
    if main is not None:
        ax.set_title(main, fontsize=font(cex_main))

    if legend_position is not None and legend_position is not False:
        # The key shows the shape that was actually drawn, so a half transparent
        # fill is half transparent in the legend too.
        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=fills[index],
                edgecolor=outlines[index] if type == "dens" else border,
                alpha=dens_alpha if type == "dens" else None,
                label=level,
            )
            for index, level in enumerate(levels)
        ]
        ax.legend(
            handles=handles,
            loc=legend_position,
            frameon=False,
            fontsize=font(cex_legend),
        )

    if not out_statistics:
        return None

    bin_stats = pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "bin_mid": (edges[:-1] + edges[1:]) / 2,
        }
    )
    for level in levels:
        bin_stats[level] = bar_length[level]

    group_stats = pd.DataFrame(
        {
            level: [
                float(n_used[level]),
                float(n_input[level] - n_used[level]),
                float(values[level].min()),
                float(values[level].max()),
            ]
            for level in levels
        },
        index=["n", "n_dropped", "min", "max"],
    )

    out: dict[str, Any] = {
        "bin_summary_stats": bin_stats,
        "group_summary_stats": group_stats,
        "group_hists": hists,
    }
    if densities is not None:
        out["group_densities"] = densities
    return out


def _two_colours(value: Any, arg: str) -> list[str]:
    """One colour per group level, from one colour or from two."""
    if isinstance(value, str):
        if arg == "col":
            raise SaValueError("`col` must contain exactly two colours, one per group level.")
        return [value, value]
    colours = list(value)
    if len(colours) != 2 or any(colour is None for colour in colours):
        if arg == "col":
            raise SaValueError("`col` must contain exactly two colours, one per group level.")
        raise SaValueError("`dens_col` must be None, one colour, or one colour per group level.")
    return [str(colour) for colour in colours]


def _breaks(breaks: Any, pooled: np.ndarray) -> np.ndarray:
    """The break points both groups are binned on.

    A rule and a bin count are both resolved against the two groups pooled, so
    the bins are shared. Port of the ``breaks`` handling in
    ``draw_butterfly_hist()``, including its use of ``pretty()``.
    """
    if isinstance(breaks, str):
        if breaks not in BREAK_RULES:
            raise SaValueError(
                "`breaks` must be one of " + ", ".join(BREAK_RULES) + f". Got {breaks}."
            )
        return pretty(
            float(pooled.min()),
            float(pooled.max()),
            n=nclass(pooled, breaks),
            min_n=1,
        )

    if np.isscalar(breaks):
        count = float(breaks)  # type: ignore[arg-type]
        if not np.isfinite(count) or count <= 0:
            raise SaValueError(
                "`breaks` must be a valid histogram rule, a positive bin count, or a "
                "strictly increasing numeric vector."
            )
        return pretty(float(pooled.min()), float(pooled.max()), n=int(count))

    edges = np.asarray(breaks, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise SaValueError(
            "`breaks` must be a valid histogram rule, a positive bin count, or a "
            "strictly increasing numeric vector."
        )
    if not bool((np.diff(edges) > 0).all()):
        raise SaValueError(
            "`breaks` must be a valid histogram rule, a positive bin count, or a "
            "strictly increasing numeric vector."
        )
    return edges


def _bar_length(hist: dict[str, Any], scale: str) -> np.ndarray:
    """How long the bar of each bin is drawn, on the scale asked for."""
    counts: np.ndarray = hist["counts"]
    if scale == "count":
        return counts
    if scale == "proportion":
        total = counts.sum()
        return counts / total if total > 0 else counts
    values: np.ndarray = hist["density"]
    return values
