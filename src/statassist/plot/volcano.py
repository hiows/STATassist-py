"""The volcano plot: effect size against significance.

Port of ``R/draw_volcano_plot.R``, for the readings this port can produce. A
term reading arrives with the factorial scenario, and with it the one-panel-per-
term figure R draws for it; a mapping of contrast tables is sent back to name
the one to draw, exactly as R does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.result import SaSignificance
from ..core.validate import (
    check_feat_names,
    check_flag,
    check_lim,
    check_margin,
    check_pvalues,
    check_scalar_num,
)
from ._theme import figure, font, set_margin

__all__ = ["draw_volcano_plot"]

#: The colours a point is drawn in: up, down, and neither. R's ``grey70`` and
#: ``green3`` are given as hex, since matplotlib does not know those names.
UP_COLOR = "#D73027"
DOWN_COLOR = "#4575B4"
NS_COLOR = "#B3B3B3"

# Labels sit on top of the points, so they use a purer and brighter shade than
# the points do rather than the same one, which would blend in.
_UP_LABEL = "red"
_DOWN_LABEL = "blue"
_GUIDE_COLOR = "#00CD00"


def draw_volcano_plot(
    significance_result: Any,
    use_adjusted: bool = True,
    log2fc_cutoff: float | None = None,
    pval_cutoff: float | None = None,
    anno_feats: bool = True,
    anno_top: int = 10,
    cex_anno: float = 1.0,
    xlim: Any = None,
    ylim: Any = None,
    xlab: str | None = None,
    main: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    margin: Any = (5, 5, 4, 3),
) -> None:
    """Plot ``log2fc`` against ``-log10(pvalue)`` and label the strongest features.

    Args:
        significance_result: The object returned by
            :func:`~statassist.estimate_significance`, whose ``significance``
            element is what is plotted. With ``by="contrast"`` that element holds
            one table per contrast, so name the one to draw:
            ``sig["significance"]["case - control"]``. A bare verdict frame is
            accepted too.
        use_adjusted: If ``True``, plot and threshold the ``adj_pvalue`` column;
            if ``False``, the unadjusted ``pvalue``. The axis label follows, so
            the y axis always describes what was actually plotted.
        log2fc_cutoff: Cutoff for calling a feature changed, drawn as a guide.
            ``None`` takes the value :func:`~statassist.estimate_significance`
            recorded, so the guides agree with the ``is_signif`` column.
        pval_cutoff: The same for significance.
        anno_feats: If ``True``, label the strongest significant features. A run
            where no feature clears both cutoffs still draws the plot, with a
            note in place of the labels.
        anno_top: How many features to label in each direction, so up to
            ``2 * anno_top`` labels in total.
        cex_anno: Character expansion for those labels.
        xlim: Length-2 x axis range, or ``None`` to derive it from the data.
        ylim: The same for the y axis.
        xlab: X axis label, or ``None`` to derive it from what ``log2fc``
            compares in the comparison behind the verdict.
        main: Plot title.
        cex_lab: Character expansion for the axis labels.
        cex_axis: Character expansion for the axis annotation.
        cex_main: Character expansion for the title.
        margin: Plot margins in lines of text: bottom, left, top, right, which
            is R's ``mar``.

    Returns:
        ``None``. The figure is drawn on the current figure, as R draws on the
        current device.

    Raises:
        SaValueError: If an argument is out of range, if the table is missing a
            column the plot needs, if the cutoffs are neither supplied nor
            recorded, or if nothing finite can be plotted.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import compare_two_groups, estimate_significance
        >>> from statassist import draw_volcano_plot, simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=20, n_up=3, n_down=3, seed=5)
        >>> sig = estimate_significance(
        ...     compare_two_groups(**sim.args, diagnose=False), log2fc_cutoff=0.1
        ... )
        >>> draw_volcano_plot(sig, main="case vs control")
    """
    use_adjusted = check_flag(use_adjusted, "use_adjusted")
    anno_feats = check_flag(anno_feats, "anno_feats")
    anno_top = int(check_scalar_num(anno_top, "anno_top", 0))
    cex_anno = check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    cex_lab = check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    cex_axis = check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    cex_main = check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    margins = check_margin(margin)
    x_limits = check_lim(xlim, "xlim")
    y_limits = check_lim(ylim, "ylim")

    # The verdict object carries the table beside the scenario name; the table on
    # its own is still accepted, since selecting rows from it produces one.
    if isinstance(significance_result, SaSignificance):
        significance_result = significance_result["significance"]
    if not isinstance(significance_result, pd.DataFrame):
        raise SaValueError(_naming_message(significance_result))

    p_col = "adj_pvalue" if use_adjusted else "pvalue"
    _check_columns(significance_result, p_col)
    cut_fc, cut_p = _cutoffs(significance_result, log2fc_cutoff, pval_cutoff)

    features = [str(name) for name in significance_result["features"]]
    check_feat_names(features)
    log2fc = significance_result["log2fc"].to_numpy(dtype=float)
    pvalue = significance_result[p_col].to_numpy(dtype=float)
    check_pvalues(pvalue, p_col)

    # A p-value of 0 gives -log10(p) = inf, which would make the axis limit
    # infinite and blank the plot. The axis follows the finite values and the
    # infinite ones are drawn at the top of it instead of being discarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        neglog_p = -np.log10(pvalue)
    x_limits, y_limits = _limits(log2fc, neglog_p, cut_fc, cut_p, x_limits, y_limits)
    label_offset = (y_limits[1] - y_limits[0]) * 0.05

    # Capped points are pulled just inside the panel rather than onto its edge,
    # so that a label still fits above them.
    plot_y = neglog_p.copy()
    plot_x = log2fc.copy()
    capped_y = np.isinf(plot_y)
    capped_x = np.isinf(plot_x)
    plot_y[capped_y] = y_limits[1] - 2 * label_offset
    span_x = x_limits[1] - x_limits[0]
    plot_x[capped_x] = np.where(
        plot_x[capped_x] > 0, x_limits[1] - span_x * 0.02, x_limits[0] + span_x * 0.02
    )
    n_capped_y = int(capped_y.sum())
    n_capped_x = int(capped_x.sum())
    if n_capped_y or n_capped_x:
        detail = ""
        if n_capped_y:
            detail += f" ({n_capped_y} with p = 0)"
        if n_capped_x:
            detail += f" ({n_capped_x} with an infinite log2 fold change)"
        notify(
            f"Drew {max(n_capped_y, n_capped_x)} point(s) at the edge of the plot"
            f"{detail}; their true position is off the axis."
        )

    # One mask per direction, shared by the points and the labels so that a point
    # can never be coloured as changed while its label is left out, or vice versa.
    with np.errstate(invalid="ignore"):
        passes_p = ~np.isnan(pvalue) & (pvalue <= cut_p)
        is_up = passes_p & ~np.isnan(log2fc) & (log2fc >= cut_fc)
        is_down = passes_p & ~np.isnan(log2fc) & (log2fc <= -cut_fc)

    fig = figure()
    ax = fig.add_subplot()
    set_margin(fig, margins)

    ax.scatter(plot_x, plot_y, color=NS_COLOR, s=20)
    ax.scatter(plot_x[is_up], plot_y[is_up], color=UP_COLOR, s=20)
    ax.scatter(plot_x[is_down], plot_y[is_down], color=DOWN_COLOR, s=20)
    ax.axhline(-np.log10(cut_p), color=_GUIDE_COLOR, linewidth=2, linestyle=":")
    for edge in (-cut_fc, cut_fc):
        ax.axvline(edge, color=_GUIDE_COLOR, linewidth=2, linestyle=":")

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel(_xlab(significance_result) if xlab is None else xlab, fontsize=font(cex_lab))
    y_lab = r"$-\log_{10}$ adjusted $P$" if use_adjusted else r"$-\log_{10}\,P$"
    ax.set_ylabel(y_lab, fontsize=font(cex_lab))
    ax.tick_params(labelsize=font(cex_axis))
    if main is not None:
        ax.set_title(main, fontsize=font(cex_main))

    if anno_feats and anno_top >= 1:
        picked = _strongest(is_up, pvalue, log2fc, anno_top, up=True)
        picked_down = _strongest(is_down, pvalue, log2fc, anno_top, up=False)
        if picked.size == 0 and picked_down.size == 0:
            notify("No feature clears both cutoffs, so nothing was labelled.")
        for index, colour in ((picked, _UP_LABEL), (picked_down, _DOWN_LABEL)):
            for i in index:
                ax.text(
                    plot_x[i],
                    plot_y[i] + label_offset,
                    features[i],
                    color=colour,
                    fontsize=font(cex_anno),
                    ha="center",
                )


def _naming_message(held: Any) -> str:
    """What to say to a caller who handed over a whole contrast reading.

    Port of ``sa_volcano_term_tables()``'s refusal. A list of contrasts is as
    long as the level counts make it, so the plot asks which one to draw. The
    term reading R draws whole arrives with the factorial scenario.
    """
    if isinstance(held, Mapping) and held:
        first = next(iter(held))
        return (
            "`significance_result` holds one verdict table per contrast, and a volcano "
            f'plot draws one of them. Name it: `sig["significance"]["{first}"]`.'
        )
    return "`significance_result` must be the object returned by estimate_significance()."


def _check_columns(table: pd.DataFrame, p_col: str) -> None:
    """The columns a verdict table has to carry to be plotted."""
    absent = [name for name in ("features", "log2fc", p_col) if name not in table.columns]
    if absent:
        raise SaValueError(
            "`significance_result` is missing the column(s) "
            + ", ".join(absent)
            + ". Pass the table returned by estimate_significance()."
        )


def _cutoffs(
    table: pd.DataFrame, log2fc_cutoff: float | None, pval_cutoff: float | None
) -> tuple[float, float]:
    """The rule a plot draws its guides for.

    Falling back to the recorded cutoffs is what keeps the guides on the plot and
    the verdict in the table describing the same rule.
    """
    if log2fc_cutoff is None:
        log2fc_cutoff = table.attrs.get("log2fc_cutoff")
    if pval_cutoff is None:
        pval_cutoff = table.attrs.get("pval_cutoff")
    if log2fc_cutoff is None or pval_cutoff is None:
        raise SaValueError(
            "`significance_result` does not carry the cutoffs estimate_significance() "
            "records, so `log2fc_cutoff` and `pval_cutoff` must be supplied. Selecting "
            "columns from the table drops them."
        )
    return (
        check_scalar_num(log2fc_cutoff, "log2fc_cutoff", 0),
        check_scalar_num(pval_cutoff, "pval_cutoff", 0, 1, lower_open=True),
    )


def _limits(
    log2fc: np.ndarray,
    neglog_p: np.ndarray,
    cut_fc: float,
    cut_p: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis ranges wide enough for the points and the guides both.

    Port of ``sa_volcano_lims()`` for a single table. A plot whose ranges are
    both supplied is never asked whether it has anything finite to derive them
    from.
    """
    if xlim is not None and ylim is not None:
        return xlim, ylim

    y_finite = neglog_p[np.isfinite(neglog_p)]
    x_finite = log2fc[np.isfinite(log2fc)]
    if y_finite.size == 0 or x_finite.size == 0:
        raise SaValueError(
            "nothing can be plotted: no feature has both a finite `log2fc` and a finite -log10(p)."
        )

    # A run where every p-value is 1 leaves the top at 0, which is not a usable
    # axis, so the guide line height sets the floor.
    y_top = max(float(y_finite.max()), float(-np.log10(cut_p)), 1.0)
    x_max = max(float(np.abs(x_finite).max()), cut_fc)
    return (
        xlim if xlim is not None else (-x_max * 1.05, x_max * 1.05),
        ylim if ylim is not None else (0.0, y_top * 1.1),
    )


def _strongest(
    mask: np.ndarray, pvalue: np.ndarray, log2fc: np.ndarray, anno_top: int, up: bool
) -> np.ndarray:
    """The features to label on one side.

    Strongest first means smallest p-value, then largest fold change away from
    zero in the direction concerned.
    """
    index = np.flatnonzero(mask)
    if index.size == 0:
        return index
    second = -log2fc[index] if up else log2fc[index]
    order = np.lexsort((second, pvalue[index]))
    return index[order][:anno_top]


def _xlab(table: pd.DataFrame) -> str:
    """The x axis label a verdict table earns.

    Port of ``sa_volcano_xlab()``. One reading does not compare two centres the
    caller named, and saying so on the axis is what keeps it from being read as a
    two-group plot: a multi-group omnibus verdict holds the level furthest from
    the reference, and which level that is differs per feature. A contrast table
    of the same comparison does compare a named pair, so it is left alone.
    """
    plain = r"$\log_2$ FC"
    if table.attrs.get("contrast") is not None:
        return plain
    if table.attrs.get("analysis") != "multi_group_comparison":
        return plain
    levels = table.attrs.get("group_lv") or []
    reference = str(levels[0]) if len(levels) > 0 else "reference"
    return f"{plain} (most extreme level vs {reference})"
