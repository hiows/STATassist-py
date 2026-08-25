"""Draw an interaction plot of a factorial comparison.

Port of ``R/draw_interaction_plot.R``. Joins the cell means of one factor across
the levels of another, one line per level of the tracing factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaValueError, notify
from ..core.result import SaFactorial
from ..core.validate import check_count, check_feat_names, check_flag, check_lim, check_scalar_num
from ._theme import figure, font, group_colors, theme

__all__ = ["INTERACTION_VIEWS", "draw_interaction_plot"]

#: The views an interaction plot can be drawn as, ``"auto"`` first.
INTERACTION_VIEWS = ("auto", "pairwise", "matrix", "facet")

#: Error-bar styles.
ERRORBARS = ("none", "se", "ci")

#: Matplotlib markers matching R's ``pch`` cycle for tracing levels.
_MARKERS = ("o", "s", "^", "D", "*", "x")

#: Matplotlib line styles matching R's ``lty`` cycle.
_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":", "--", "-.")


def draw_interaction_plot(
    comparison_result: Any,
    x: str | None = None,
    trace: str | None = None,
    facet: str | None = None,
    type: str = "auto",
    feats: Any = None,
    errorbar: str = "none",
    panel_nrow: int | None = None,
    dark: bool = False,
    ylim: Any = None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    col: Any = None,
    lwd: float = 2.0,
    cex_axis: float = 1.2,
    cex_lab: float = 1.3,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
) -> pd.DataFrame:
    """Draw an interaction plot of a factorial comparison.

    The means come from ``comparison_result["cells"]``. Remaining factors are
    averaged away unweighted, matching the marginal means the post-hoc stage
    contrasts.

    Args:
        comparison_result: A factorial comparison from
            :func:`~statassist.compare_factorial_groups`.
        x: Factor on the x axis.
        trace: Factor with one line per level. Unnamed factors are taken in
            declaration order, ``trace`` first.
        facet: Factor whose levels stay in separate panels (facet view only).
        type: One of :data:`INTERACTION_VIEWS`.
        feats: Features to draw. ``None`` draws every feature in the pairwise
            view and the first feature otherwise.
        errorbar: ``"none"``, ``"se"`` or ``"ci"``.
        panel_nrow: Panel rows, or ``None`` for a view-dependent default.
        dark: Dark background with light text.
        ylim: Length-2 y range, or ``None`` to derive it.
        xlab, ylab, main: Labels; derived when ``None``.
        col: Colours for tracing levels, recycled if short.
        lwd: Line width.
        cex_axis, cex_lab, cex_main, cex_legend: Character expansion multipliers.

    Returns:
        The plotted means, one row per point, with ``attrs["view"]`` holding the
        resolved view.
    """
    if type not in INTERACTION_VIEWS:
        raise SaValueError("`type` must be one of: " + ", ".join(INTERACTION_VIEWS) + ".")
    if errorbar not in ERRORBARS:
        raise SaValueError("`errorbar` must be one of: " + ", ".join(ERRORBARS) + ".")
    dark = check_flag(dark, "dark")
    lwd = check_scalar_num(lwd, "lwd", 0, lower_open=True)
    cex_legend = check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    limits_given = check_lim(ylim, "ylim")
    if panel_nrow is not None:
        panel_nrow = check_count(panel_nrow, "panel_nrow", 1)

    cells = _inter_cells(comparison_result)
    factor_lv = {
        str(name): [str(level) for level in levels]
        for name, levels in comparison_result["design"]["factor_lv"].items()
    }
    roles = _inter_roles(list(factor_lv), x, trace, facet, type)
    type = str(roles["type"])
    drawn_feats = _inter_feats(comparison_result, feats, type)
    mult = _inter_multiplier(comparison_result, drawn_feats, errorbar)
    panels = _inter_panels(cells, factor_lv, roles, drawn_feats, mult, type)
    drawn = pd.concat([panel["tbl"] for panel in panels], ignore_index=True)

    if not np.isfinite(drawn["mean"].to_numpy(dtype=float)).any():
        raise SaValueError(
            "none of the cells of "
            + ", ".join(drawn_feats)
            + " holds a mean to draw, so the model was not fitted for any of them. "
            "`comparison_result['tests']['anova_test']` says why."
        )

    if ylab is None:
        ylab = _inter_ylab(drawn, comparison_result["parameters"]["input_scale"])
    _inter_draw(
        panels,
        drawn,
        factor_lv,
        roles,
        type,
        errorbar,
        panel_nrow,
        dark,
        limits_given,
        xlab,
        ylab,
        main,
        col,
        lwd,
        cex_axis,
        cex_lab,
        cex_main,
        cex_legend,
    )
    drawn.attrs["view"] = type
    return cast(pd.DataFrame, drawn)


def _inter_cells(res: Any) -> pd.DataFrame:
    """The cell table of a factorial comparison, or the reason there is none."""
    if not isinstance(res, SaFactorial):
        raise SaValueError(
            "`comparison_result` must be a factorial comparison result, as "
            "returned by compare_factorial_groups(). An interaction is a "
            "statement about two crossed factors, and a result over a single "
            "factor holds no second one to trace against."
        )
    if "cells" not in res or res["cells"] is None:
        raise SaValueError(
            "`comparison_result` carries no `cells` table, so it was produced "
            "by a version of the package that did not record the cell means. "
            "Re-run compare_factorial_groups() on the same data."
        )
    return cast(pd.DataFrame, res["cells"])


def _inter_roles(
    fac: Sequence[str],
    x: str | None,
    trace: str | None,
    facet: str | None,
    type: str,
) -> dict[str, Any]:
    """Settle which view is being drawn and which factor plays which part."""

    def check_one(name: str | None, arg: str) -> None:
        if name is None:
            return
        if not isinstance(name, str) or name == "":
            raise SaValueError(f"`{arg}` must be a single factor name, or NULL.")
        if name not in fac:
            raise SaValueError(
                f"`{arg}` must name one of the factors of the comparison: "
                + ", ".join(fac)
                + f". Got {name}."
            )

    check_one(trace, "trace")
    check_one(x, "x")
    check_one(facet, "facet")

    named: dict[str, str] = {}
    if trace is not None:
        named["trace"] = trace
    if x is not None:
        named["x"] = x
    if facet is not None:
        named["facet"] = facet
    if len(named) != len(set(named.values())):
        dup = [value for value in named.values() if list(named.values()).count(value) > 1]
        raise SaValueError(
            "`x`, `trace` and `facet` must name different factors; "
            + ", ".join(dict.fromkeys(dup))
            + " is named twice. One factor cannot be two axes of the same plot."
        )

    if type == "auto":
        if facet is not None:
            type = "facet"
        elif named:
            type = "pairwise"
        elif len(fac) > 2:
            type = "matrix"
        else:
            type = "pairwise"

    if type == "matrix":
        if named:
            raise SaValueError(
                '`type = "matrix"` draws every pair of factors, so there is '
                "nothing for "
                + ", ".join(named)
                + ' to choose. Drop it, or use type = "pairwise" to draw one pair.'
            )
        return {"type": type, "x": None, "trace": None, "facet": None}

    if type == "facet":
        if facet is None:
            raise SaValueError(
                '`type = "facet"` needs `facet` to name the factor whose levels '
                "go in panels of their own. Without it the third factor is "
                'averaged away, which is type = "pairwise".'
            )
        if len(fac) < 3:
            raise SaValueError(
                '`type = "facet"` needs at least three factors, one for each of '
                "`x`, `trace` and `facet`, and this comparison holds "
                f"{len(fac)}: " + ", ".join(fac) + "."
            )
    elif facet is not None:
        raise SaValueError(
            '`facet` belongs to type = "facet". In the pairwise view every '
            "factor other than `x` and `trace` is averaged away, so there is no "
            f"panel for {facet} to be kept in."
        )

    free = [name for name in fac if name not in named.values()]
    for part in ("trace", "x"):
        if part not in named:
            named[part] = free[0]
            free = free[1:]

    return {
        "type": type,
        "x": named["x"],
        "trace": named["trace"],
        "facet": named.get("facet") if type == "facet" else None,
    }


def _inter_feats(res: Any, feats: Any, type: str) -> list[str]:
    """Which features the view has room for."""
    one_panel_each = type == "pairwise"
    if feats is None:
        if one_panel_each:
            return list(res["features"])
        first = str(res["features"][0])
        notify(
            f"Drawing {first}. The {type} view spends its panels on the factors, "
            "so it draws one feature at a time; name another in `feats`."
        )
        return [first]

    names = check_feat_names(feats)
    unknown = [name for name in names if name not in res["features"]]
    if unknown:
        raise SaValueError(
            "`feats` must name features present in the comparison: "
            + ", ".join(res["features"])
            + ". Not found: "
            + ", ".join(unknown)
            + "."
        )
    if not one_panel_each and len(names) > 1:
        raise SaValueError(
            f'`type = "{type}"` draws one feature at a time, and `feats` names '
            f"{len(names)}: " + ", ".join(names) + ". Its panels are spent on "
            'the factors; use type = "pairwise" for a panel per feature.'
        )
    return names


def _inter_multiplier(res: Any, feats: Sequence[str], errorbar: str) -> dict[str, float]:
    """How many standard errors wide a bar is, per feature."""
    if errorbar != "ci":
        value = 1.0 if errorbar == "se" else float("nan")
        return {name: value for name in feats}
    table = res["tests"]["anova_test"].set_index("features")
    conf = float(res["parameters"]["conf_level"])
    out: dict[str, float] = {}
    for name in feats:
        df2 = float(table.loc[name, "df2"])
        out[name] = float(stats.t.ppf(1 - (1 - conf) / 2, df2))
    return out


def _inter_marginal(
    cells: pd.DataFrame,
    factor_lv: Mapping[str, Sequence[str]],
    x: str,
    trace: str,
) -> pd.DataFrame:
    """The marginal means of one pair of factors."""
    rows: list[dict[str, Any]] = []
    for trace_level in factor_lv[trace]:
        for x_level in factor_lv[x]:
            at = (cells[x].astype(str) == str(x_level)) & (
                cells[trace].astype(str) == str(trace_level)
            )
            k = int(at.sum())
            if k == 0:
                mean = float("nan")
                se = float("nan")
            else:
                mean = float(np.mean(cells.loc[at, "mean"].to_numpy(dtype=float)))
                se_vals = cells.loc[at, "se"].to_numpy(dtype=float)
                se = float(np.sqrt(np.sum(se_vals**2) / k**2))
            rows.append(
                {
                    "x_level": str(x_level),
                    "trace_level": str(trace_level),
                    "n_cells": k,
                    "mean": mean,
                    "se": se,
                }
            )
    return pd.DataFrame(rows)


def _inter_panels(
    cells: pd.DataFrame,
    factor_lv: Mapping[str, Sequence[str]],
    roles: Mapping[str, Any],
    feats: Sequence[str],
    mult: Mapping[str, float],
    type: str,
) -> list[dict[str, Any]]:
    """Assemble one table per panel, in the order the panels are drawn."""
    fac = list(factor_lv)

    def one(
        label: str,
        feature: str,
        x_name: str,
        trace_name: str,
        keep: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        rows = cells.loc[cells["features"].astype(str) == feature]
        if keep is not None:
            name, level = next(iter(keep.items()))
            rows = rows.loc[rows[name].astype(str) == str(level)]
        tbl = _inter_marginal(rows, factor_lv, x_name, trace_name)
        half = float(mult[feature]) * tbl["se"].to_numpy(dtype=float)
        out = pd.DataFrame(
            {
                "panel": label,
                "features": feature,
                "x_factor": x_name,
                "x_level": tbl["x_level"],
                "trace_factor": trace_name,
                "trace_level": tbl["trace_level"],
                "n_cells": tbl["n_cells"],
                "mean": tbl["mean"],
                "se": tbl["se"],
                "lower_conf": tbl["mean"].to_numpy(dtype=float) - half,
                "upper_conf": tbl["mean"].to_numpy(dtype=float) + half,
            }
        )
        return {"label": label, "x": x_name, "trace": trace_name, "tbl": out}

    if type == "pairwise":
        return [one(feature, feature, str(roles["x"]), str(roles["trace"])) for feature in feats]

    if type == "facet":
        facet_name = str(roles["facet"])
        return [
            one(
                f"{facet_name}: {level}",
                feats[0],
                str(roles["x"]),
                str(roles["trace"]),
                keep={facet_name: str(level)},
            )
            for level in factor_lv[facet_name]
        ]

    out: list[dict[str, Any]] = []
    for i in range(len(fac) - 1):
        for j in range(i, len(fac) - 1):
            panel = one(
                f"{fac[i]} x {fac[j + 1]}",
                feats[0],
                fac[j + 1],
                fac[i],
            )
            panel["row"] = i
            panel["col"] = j
            out.append(panel)
    return out


def _inter_span(tbl: pd.DataFrame, errorbar: str) -> tuple[float, float]:
    """The y range the means and their bars need."""
    if errorbar == "none":
        vals = tbl["mean"].to_numpy(dtype=float)
    else:
        vals = np.concatenate(
            [
                tbl["mean"].to_numpy(dtype=float),
                tbl["lower_conf"].to_numpy(dtype=float),
                tbl["upper_conf"].to_numpy(dtype=float),
            ]
        )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    low, high = float(vals.min()), float(vals.max())
    if high == low:
        pad = max(abs(low), 1.0) * 0.1
        return (low - pad, high + pad)
    pad = (high - low) * 0.08
    return (low - pad, high + pad)


def _inter_ylab(drawn: pd.DataFrame, input_scale: str) -> str:
    """What the y axis is measuring."""
    what = "marginal mean" if (drawn["n_cells"] > 1).any() else "cell mean"
    return f"{what} (log2)" if input_scale == "log2" else what


def _inter_main(
    panels: Sequence[Mapping[str, Any]], roles: Mapping[str, Any], type: str, strip: bool
) -> str:
    """The title the view describes itself with."""
    if type == "matrix":
        return f"Interactions of {panels[0]['tbl']['features'].iloc[0]}"
    head = f"Interaction of {roles['trace']} and {roles['x']}"
    if type == "facet":
        return f"{panels[0]['tbl']['features'].iloc[0]}: {head} by {roles['facet']}"
    if strip:
        return head
    return f"{panels[0]['tbl']['features'].iloc[0]}: {head}"


def _inter_draw(
    panels: Sequence[dict[str, Any]],
    drawn: pd.DataFrame,
    factor_lv: Mapping[str, Sequence[str]],
    roles: Mapping[str, Any],
    type: str,
    errorbar: str,
    panel_nrow: int | None,
    dark: bool,
    ylim: tuple[float, float] | None,
    xlab: str | None,
    ylab: str,
    main: str | None,
    col: Any,
    lwd: float,
    cex_axis: float,
    cex_lab: float,
    cex_main: float,
    cex_legend: float,
) -> None:
    """Lay the panels out and draw them."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    look = theme(dark)
    n_panel = len(panels)
    shared_legend = type != "matrix"
    free_scale = type == "pairwise" and n_panel > 1
    span = None if ylim is not None else _inter_span(drawn, errorbar)

    fig = figure()
    fig.patch.set_facecolor(look.bg)

    if type == "matrix":
        if panel_nrow is not None:
            notify(
                "`panel_nrow` is not used by the matrix view, whose grid is fixed by the factors."
            )
        side = len(factor_lv) - 1
        n_row = side
        n_col = side
        grid_map = np.full((side, side), -1, dtype=int)
        for index, panel in enumerate(panels):
            grid_map[int(panel["row"]), int(panel["col"])] = index
    else:
        if panel_nrow is None:
            panel_nrow = max(1, int(round(np.sqrt(n_panel)))) if free_scale else 1
        n_row = min(panel_nrow, n_panel)
        n_col = int(np.ceil(n_panel / n_row))
        grid_map = np.full((n_row, n_col), -1, dtype=int)
        for index in range(n_panel):
            grid_map[index // n_col, index % n_col] = index

    strip = n_panel > 1
    default_main = _inter_main(panels, roles, type, strip)
    figure_main = default_main if main is None else main
    outer_main = strip and figure_main is not None

    width_ratios = [4] * n_col + ([1] if shared_legend else [])
    gs = GridSpec(
        n_row,
        n_col + (1 if shared_legend else 0),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.25,
        hspace=0.35,
        top=0.88 if outer_main else 0.92,
    )

    axes_by_panel: list[Any] = [None] * n_panel
    for row in range(n_row):
        for column in range(n_col):
            index = int(grid_map[row, column])
            if index < 0:
                continue
            axes_by_panel[index] = fig.add_subplot(gs[row, column])

    for index, panel in enumerate(panels):
        ax = axes_by_panel[index]
        at_col = int(panel["col"]) if type == "matrix" else index % n_col
        y_annot = free_scale or at_col == 0 or (type == "matrix" and panel["row"] == panel["col"])
        if ylim is not None:
            panel_ylim = ylim
        elif free_scale:
            panel_ylim = _inter_span(panel["tbl"], errorbar)
        else:
            assert span is not None
            panel_ylim = span
        _inter_panel(
            ax,
            panel["tbl"],
            list(factor_lv[panel["x"]]),
            list(factor_lv[panel["trace"]]),
            panel_ylim,
            panel["x"] if xlab is None else xlab,
            ylab if y_annot else "",
            None if outer_main else figure_main,
            group_colors(col, len(factor_lv[panel["trace"]])),
            lwd,
            errorbar,
            look,
            y_annot,
            None if shared_legend else panel["trace"],
            cex_axis,
            cex_lab,
            cex_main,
            cex_legend,
        )
        if strip:
            ax.set_title(panel["label"], color=look.fg, fontsize=font(cex_axis), pad=6)

    if shared_legend:
        legend_ax = fig.add_subplot(gs[:, -1])
        legend_ax.set_axis_off()
        legend_ax.set_facecolor(look.bg)
        trace_lv = list(factor_lv[str(roles["trace"])])
        colours = group_colors(col, len(trace_lv))
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=colours[index],
                marker=_MARKERS[index % len(_MARKERS)],
                linestyle=cast(Any, _LINESTYLES[index % len(_LINESTYLES)]),
                linewidth=lwd,
                label=str(level),
            )
            for index, level in enumerate(trace_lv)
        ]
        legend = legend_ax.legend(
            handles=handles,
            title=str(roles["trace"]),
            loc="center",
            frameon=False,
            fontsize=font(cex_legend),
            title_fontsize=font(cex_legend),
            labelcolor=look.fg,
        )
        if legend is not None:
            legend.get_title().set_color(look.fg)

    if outer_main:
        fig.suptitle(str(figure_main), color=look.fg, fontsize=font(cex_main), fontweight="bold")


def _inter_panel(
    ax: Any,
    tbl: pd.DataFrame,
    x_lv: Sequence[str],
    trace_lv: Sequence[str],
    ylim: tuple[float, float],
    xlab: str,
    ylab: str,
    main: str | None,
    colours: Sequence[Any],
    lwd: float,
    errorbar: str,
    look: Any,
    y_annot: bool,
    key: str | None,
    cex_axis: float,
    cex_lab: float,
    cex_main: float,
    cex_legend: float,
) -> None:
    """Draw one panel of traces."""
    n_x = len(x_lv)
    ax.set_facecolor(look.bg)
    ax.set_xlim(1 - 0.25, n_x + 0.25)
    ax.set_ylim(ylim)
    ax.set_xticks(list(range(1, n_x + 1)), list(x_lv))
    ax.tick_params(colors=look.fg, labelsize=font(cex_axis))
    ax.set_xlabel(xlab, color=look.fg, fontsize=font(cex_lab))
    ax.set_ylabel(ylab, color=look.fg, fontsize=font(cex_lab))
    if main:
        ax.set_title(main, color=look.fg, fontsize=font(cex_main))
    if not y_annot:
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_color(look.fg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", color=look.guide, linewidth=1)
    ax.set_axisbelow(True)

    for index, level in enumerate(trace_lv):
        point = tbl.loc[tbl["trace_level"].astype(str) == str(level)].copy()
        if len(point.index) == 0:
            continue
        order = pd.Categorical(point["x_level"].astype(str), categories=list(x_lv), ordered=True)
        point = point.iloc[np.argsort(order.codes, kind="stable")]
        xs = np.arange(1, n_x + 1, dtype=float)
        colour = colours[index]
        marker = _MARKERS[index % len(_MARKERS)]
        linestyle = _LINESTYLES[index % len(_LINESTYLES)]
        if errorbar != "none":
            lower = point["lower_conf"].to_numpy(dtype=float)
            upper = point["upper_conf"].to_numpy(dtype=float)
            has_bar = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
            if has_bar.any():
                ax.errorbar(
                    xs[has_bar],
                    point["mean"].to_numpy(dtype=float)[has_bar],
                    yerr=[
                        point["mean"].to_numpy(dtype=float)[has_bar] - lower[has_bar],
                        upper[has_bar] - point["mean"].to_numpy(dtype=float)[has_bar],
                    ],
                    fmt="none",
                    ecolor=colour,
                    elinewidth=1,
                    capsize=2,
                )
        ax.plot(
            xs,
            point["mean"].to_numpy(dtype=float),
            color=colour,
            linestyle=linestyle,
            linewidth=lwd,
            marker=marker,
            markersize=7,
        )

    if key is not None:
        handles = [
            ax.plot(
                [],
                [],
                color=colours[index],
                marker=_MARKERS[index % len(_MARKERS)],
                linestyle=_LINESTYLES[index % len(_LINESTYLES)],
                linewidth=lwd,
                label=str(level),
            )[0]
            for index, level in enumerate(trace_lv)
        ]
        legend = ax.legend(
            handles=handles,
            title=key,
            loc="upper left",
            frameon=False,
            fontsize=font(cex_legend) * 0.85,
            title_fontsize=font(cex_legend) * 0.85,
            labelcolor=look.fg,
        )
        legend.get_title().set_color(look.fg)
