"""The forest plot: one estimate and its interval per row.

Port of ``R/draw_forest_plot.R``. Only the columns the result contract
guarantees are read, which is why one function covers every scenario: it never
asks whether the object came from two groups, three groups or a single sample,
only whether the table it was handed has intervals or p-values, and every table
has one or the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.result import pick_test
from ..core.validate import check_feat_names, check_flag, check_lim, check_scalar_num
from ._theme import estimate_column, figure, font, line_inches, theme

__all__ = ["FOREST_VIEWS", "draw_forest_plot"]

#: The views a forest plot can be drawn as, ``"auto"`` first.
FOREST_VIEWS = ("auto", "estimate", "posthoc", "pvalue")

#: Where the relative effect's null sits. It is the one quantity in the package
#: whose null is not zero, so a guide drawn at zero would put every point of a
#: Brunner-Munzel table on one side of it.
RELATIVE_EFFECT_NULL = 0.5

#: How much of the data range is left as air either side of it, when the caller
#: names no ``xlim``.
_PAD = 0.08


def draw_forest_plot(
    comparison_result: Any,
    test: str | None = None,
    type: str = "auto",
    feats: Any = None,
    use_adjusted: bool = True,
    alpha: float = 0.05,
    sort_by: str = "none",
    dark: bool = False,
    xlim: Any = None,
    xlab: str | None = None,
    main: str | None = None,
    col_signif: str = "#D1495B",
    col_plain: str = "#7F8C8D",
    cex_axis: float = 0.9,
    cex_lab: float = 1.1,
    cex_main: float = 1.2,
    cex_legend: float = 1.0,
) -> pd.DataFrame:
    """Draw the estimate of each feature beside its confidence interval.

    Falls back to a bar of ``-log10(pval_adj)`` for a table that has no interval
    to draw, which is every omnibus test. Three views are available:

    ``"estimate"``
        A forest plot of the effect estimate and its confidence interval, one
        row per feature. Available whenever the chosen table has finite
        intervals, which the two-group and one-sample scenarios always do.
    ``"posthoc"``
        The same forest plot for the pairwise contrasts of a multi-group
        comparison, one row per contrast. Only the first feature of the post-hoc
        table is drawn unless ``feats`` names the ones to draw. An estimate reads
        as ``group1 - group2``, the direction the row label spells out, so a
        point to the right of the guide agrees in sign with the ``log2fc`` a
        volcano plot of the same comparison draws.
    ``"pvalue"``
        ``-log10()`` of the p-value per feature with the ``alpha`` threshold
        marked. The fallback when a table has no interval to draw.

    ``type = "auto"``, the default, picks the first of those three that the
    chosen table can actually support.

    Args:
        comparison_result: A comparison result, as returned by
            :func:`~statassist.compare_two_groups`,
            :func:`~statassist.compare_multiple_groups` or
            :func:`~statassist.compare_one_sample`.
        test: Which test to draw. One of ``comparison_result["tests"]``, the
            first of them by default.
        type: One of :data:`FOREST_VIEWS`.
        feats: Features to draw, in display order from the top of the plot down.
            ``None`` draws every feature of the chosen table, except in the
            post-hoc view, where it draws the first feature of the post-hoc
            table. A feature that has no contrasts because its omnibus test did
            not qualify is reported and left out.
        use_adjusted: If ``True``, read the ``pval_adj`` column; if ``False``,
            the unadjusted ``pval``. The colouring, the sorting, the p-value view
            and the labels all follow, so the plot always describes the p-value
            it actually used.
        alpha: Threshold marked on the p-value view and used to colour the points
            of the estimate view.
        sort_by: ``"none"`` to keep the feature order of the result, or
            ``"pvalue"`` to draw the most significant rows at the top.
        dark: If ``True``, use a dark background with light text.
        xlim: Length-2 x axis range, or ``None`` to derive it from the values
            being drawn. A supplied range is used as given, and the interval
            bounds are clamped to it, so an interval running past the range still
            reaches the edge of the panel instead of vanishing.
        xlab: Axis label, derived from the result when ``None``.
        main: Title, derived from the result when ``None``.
        col_signif: Colour for rows at or below ``alpha``.
        col_plain: Colour for the rest.
        cex_axis: Character expansion for the axis annotation.
        cex_lab: Character expansion for the axis label.
        cex_main: Character expansion for the title.
        cex_legend: Character expansion for the legend.

    Returns:
        The plotted table, in the row order it was drawn, with the view that
        ``type = "auto"`` resolved to in ``attrs["view"]``.

    Raises:
        SaValueError: If an argument is not one of the values named above, if
            ``feats`` names something the comparison does not hold, or if the
            view asked for is one the chosen table cannot support.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from statassist import compare_two_groups, draw_forest_plot
        >>> from statassist import simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=6, n_up=2, n_down=2, seed=3)
        >>> drawn = compare_two_groups(**sim.args, diagnose=False)
        >>> plotted = draw_forest_plot(drawn)
        >>> plotted.attrs["view"]
        'estimate'
    """
    if type not in FOREST_VIEWS:
        raise SaValueError("`type` must be one of " + ", ".join(FOREST_VIEWS) + f". Got {type}.")
    if sort_by not in ("none", "pvalue"):
        raise SaValueError(f"`sort_by` must be none or pvalue. Got {sort_by}.")
    dark = check_flag(dark, "dark")
    use_adjusted = check_flag(use_adjusted, "use_adjusted")
    alpha = check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    limits_given = check_lim(xlim, "xlim")

    if test is None:
        test = next(iter(comparison_result["tests"]))
    table = pick_test(comparison_result, test, arg="comparison_result")
    posthoc = comparison_result.get("posthoc", {}).get(test)
    p_col = "pval_adj" if use_adjusted else "pval"
    had_posthoc = posthoc is not None and len(posthoc.index) > 0
    not_qualified: list[str] = []

    if feats is not None:
        feats = check_feat_names(feats)
        known = list(table["features"])
        unknown = [name for name in feats if name not in known]
        if unknown:
            raise SaValueError(
                "`feats` must name features present in the comparison: "
                + ", ".join(known)
                + ". Not found: "
                + ", ".join(unknown)
                + "."
            )
        # Selected before `type = "auto"` resolves, so the view is chosen from
        # what will actually be drawn rather than from the whole table. Picking
        # three features whose intervals are all missing falls through to the
        # p-value view instead of drawing an empty panel.
        table = table.set_index("features").loc[feats].reset_index()
        if had_posthoc:
            assert posthoc is not None
            not_qualified = [name for name in feats if name not in set(posthoc["features"])]
            keep = posthoc["features"].isin(feats)
            posthoc = posthoc.loc[keep].copy()
            order = pd.Categorical(posthoc["features"], categories=feats, ordered=True)
            posthoc = posthoc.iloc[np.argsort(order.codes, kind="stable")]

    has_estimate = estimate_column(table) is not None and bool(
        (np.isfinite(table["lower_conf"]) & np.isfinite(table["upper_conf"])).any()
    )
    has_posthoc = posthoc is not None and len(posthoc.index) > 0

    if type == "auto":
        if has_estimate:
            type = "estimate"
        elif has_posthoc:
            type = "posthoc"
        else:
            type = "pvalue"

    info = comparison_result["test_info"][test]
    estimate: np.ndarray | None
    null_value: float | None

    if type == "posthoc":
        if not has_posthoc:
            if had_posthoc:
                raise SaValueError(
                    "none of the features named in `feats` has contrasts to draw: "
                    + ", ".join(not_qualified)
                    + ". The pairwise stage runs only for the features whose omnibus "
                    "test qualified."
                )
            raise SaValueError(
                f'`comparison_result["posthoc"]["{test}"]` holds no contrasts to draw. '
                "A post-hoc stage belongs to a comparison of three or more levels or "
                "to a factorial one, and even there it runs only for the features that "
                "qualified."
            )
        assert posthoc is not None
        if not_qualified:
            notify(
                "No contrasts for "
                + ", ".join(not_qualified)
                + "; the omnibus test did not qualify them for the post-hoc stage."
            )
        # Every contrast of every feature at once is a wall of rows that says
        # nothing, so an unselected post-hoc view stays on the first feature.
        if feats is None:
            first = posthoc["features"].iloc[0]
            drawn = posthoc.loc[posthoc["features"] == first].copy()
        else:
            drawn = posthoc.copy()
        many_feats = drawn["features"].nunique() > 1
        # A contrast label reads the same for every feature, so it only
        # identifies a row once the plot is down to one feature.
        if many_feats:
            labels = [
                f"{feature}: {contrast}"
                for feature, contrast in zip(drawn["features"], drawn["contrast"], strict=True)
            ]
        else:
            labels = list(drawn["contrast"])
        estimate = drawn["estimate"].to_numpy(dtype=float)
        null_value = 0.0
        default_xlab = "estimate (group1 - group2)"
        posthoc_label = info["posthoc_label"]
        default_main = (
            posthoc_label if many_feats else f"{drawn['features'].iloc[0]}: {posthoc_label}"
        )
    elif type == "estimate":
        column = estimate_column(table)
        if column is None:
            raise SaValueError(
                f'`comparison_result["tests"]["{test}"]` holds no estimate to draw. '
                "An omnibus test reports that the levels differ, not by how much; "
                'use type="posthoc" for the contrasts or type="pvalue".'
            )
        drawn = table.copy()
        labels = list(drawn["features"])
        estimate = drawn[column].to_numpy(dtype=float)
        null_value = RELATIVE_EFFECT_NULL if column == "relative_effect" else 0.0
        default_xlab = column
        default_main = info["label"]
    else:
        drawn = table.copy()
        labels = list(drawn["features"])
        estimate = None
        null_value = None
        default_xlab = r"$-\log_{10}$ adjusted $P$" if use_adjusted else r"$-\log_{10}\,P$"
        default_main = info["label"]

    if sort_by == "pvalue":
        # numpy sorts missing p-values to the end, which is R's `na.last = TRUE`.
        order_by = np.argsort(drawn[p_col].to_numpy(dtype=float), kind="stable")
        drawn = drawn.iloc[order_by]
        labels = [labels[i] for i in order_by]
        if estimate is not None:
            estimate = estimate[order_by]

    colours = theme(dark)
    n_rows = len(drawn.index)
    # Row 1 at the top, so a sorted plot reads downwards like the table it came
    # from rather than upwards like a default plot.
    at = np.arange(n_rows, 0, -1, dtype=float)
    p_values = drawn[p_col].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        signif = ~np.isnan(p_values) & (p_values <= alpha)
    row_colours = [col_signif if flag else col_plain for flag in signif]

    fig = figure()
    fig.patch.set_facecolor(colours.bg)
    ax = fig.add_subplot()
    ax.set_facecolor(colours.bg)

    # A long label earns a wider left margin, but only up to half of the panel
    # it is labelling. Past that the labels are shrunk to fit the margin they
    # get, which keeps every name readable in full instead of squeezing the plot
    # into a strip too narrow to carry an axis.
    label_width = max((len(name) for name in labels), default=0)
    wanted = max(4.1, 0.6 * label_width)
    panel_lines = fig.get_size_inches()[0] * 0.8 / line_inches()
    left = min(wanted, 0.5 * panel_lines)
    cex_labels = cex_axis * min(1.0, left / wanted)

    if type == "pvalue":
        with np.errstate(divide="ignore", invalid="ignore"):
            height = -np.log10(p_values)
        height = np.where(np.isfinite(height), height, 0.0)
        if limits_given is None:
            tallest = float(height.max()) if height.size else 0.0
            limits = (0.0, max(tallest, float(-np.log10(alpha))) * 1.1)
        else:
            limits = limits_given
        ax.axvline(-np.log10(alpha), linestyle="--", color=colours.guide, linewidth=1)
        ax.hlines(at, 0.0, height, colors=row_colours, linewidth=6)
    else:
        bounds = np.concatenate(
            [
                drawn["lower_conf"].to_numpy(dtype=float),
                drawn["upper_conf"].to_numpy(dtype=float),
                np.asarray([] if estimate is None else estimate, dtype=float),
                np.asarray([null_value], dtype=float),
            ]
        )
        finite = bounds[np.isfinite(bounds)]
        if finite.size == 0:
            raise SaValueError(
                "nothing can be drawn: the chosen table holds no finite estimate or interval bound."
            )
        span = (float(finite.min()), float(finite.max()))
        if limits_given is None:
            air = (span[1] - span[0]) * _PAD
            limits = (span[0] - air, span[1] + air)
        else:
            limits = limits_given
        assert null_value is not None and estimate is not None
        ax.axvline(null_value, linestyle="--", color=colours.guide, linewidth=1)
        # Bounds are clamped to the panel so that a one-sided interval, which
        # runs to infinity on the side it does not test, still draws as a line
        # reaching the edge instead of vanishing. A narrowed `xlim` cuts the same
        # way.
        lower = np.maximum(drawn["lower_conf"].to_numpy(dtype=float), min(limits))
        upper = np.minimum(drawn["upper_conf"].to_numpy(dtype=float), max(limits))
        ax.hlines(at, lower, upper, colors=row_colours, linewidth=2)
        ax.scatter(estimate, at, marker="D", s=60, c=row_colours, zorder=3)

    ax.set_xlim(limits)
    ax.set_ylim(0.5, n_rows + 0.5)
    ax.set_yticks(at)
    ax.set_yticklabels(labels, fontsize=font(cex_labels), color=colours.fg)
    ax.tick_params(axis="y", length=0, colors=colours.fg)
    ax.tick_params(axis="x", labelsize=font(cex_axis), colors=colours.fg)
    ax.set_xlabel(default_xlab if xlab is None else xlab, fontsize=font(cex_lab), color=colours.fg)
    ax.set_title(default_main if main is None else main, fontsize=font(cex_main), color=colours.fg)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colours.fg)

    # The threshold moves to the legend title so that the two entries can say
    # what they mean rather than repeat the comparison that produced them. The
    # legend sits outside the panel, where it cannot cover the rows it describes.
    subscript = "_{adj}" if use_adjusted else ""
    entries = ("Significant", "Not significant")
    handles = [
        _swatch(col_signif, entries[0]),
        _swatch(col_plain, entries[1]),
    ]
    legend = ax.legend(
        handles=handles,
        title=rf"$P{subscript} \leq {alpha:g}$",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=font(cex_legend),
        labelcolor=colours.fg,
    )
    legend.get_title().set_color(colours.fg)
    legend.get_title().set_fontsize(font(cex_legend))

    # The panel gives up as much of its right side as the legend's longest entry
    # needs, which is what keeps a long entry from being drawn off the figure.
    width = float(fig.get_size_inches()[0])
    entry_inches = (max(len(text) for text in entries) * 0.6 + 3) * font(cex_legend) / 72
    fig.subplots_adjust(
        left=min(0.6, left * line_inches() / width),
        right=max(0.5, 1 - (entry_inches + 0.1) / width),
    )

    # The view is carried on the result because `type = "auto"` resolves it here
    # and the caller would otherwise have no way to find out which of the three
    # it got, short of reading the axis label off the figure.
    out: pd.DataFrame = drawn.reset_index(drop=True)
    out.attrs = dict(drawn.attrs)
    out.attrs["view"] = type
    return out


def _swatch(colour: str, label: str) -> Any:
    """A filled square for the legend, which is what R's ``pch = 15`` draws."""
    from matplotlib.lines import Line2D

    return Line2D([], [], marker="s", linestyle="none", color=colour, markersize=8, label=label)
