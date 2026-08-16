"""Grouped barplot of a descriptive summary (matplotlib port)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from statassist.describe.summarize_descriptive_stats import summarize_descriptive_stats
from statassist.plot._theme import dark2_colors, sa_plot_theme
from statassist.utils.validate import (
    sa_check_flag,
    sa_check_lim,
    sa_check_scalar_num,
    sa_control_first,
    sa_validate_wide_input,
)

MAINBARS = (
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
ERRORBARS = ("none", "se", "sd", "ci")


def _sa_bar_check_pair(mainbar: str, errorbar: str) -> None:
    if errorbar == "none" or mainbar == "mean":
        return
    if mainbar == "median":
        if errorbar == "ci":
            return
        raise ValueError(
            f'`errorbar = "{errorbar}"` describes the spread of the '
            "observations about their mean, so it is not a width to draw either "
            "side of a median. `mainbar = \"median\"` takes "
            'errorbar = "ci", the notch interval '
            "median +/- 1.58 * IQR / sqrt(n), or \"none\"."
        )
    raise ValueError(
        f'`mainbar = "{mainbar}"` is itself a spread, a count or a shape, '
        "so there is no second quantity for an interval either side of it to be "
        'about. Only "mean" and "median" take an `errorbar`; this height '
        'takes errorbar = "none".'
    )


def _sa_bar_input(
    data: pd.DataFrame,
    feats: list[str],
    group: Any,
    group_lv: list[str] | None,
    control_label: str | None,
) -> dict[str, Any]:
    if group is None:
        raise ValueError(
            "`group` says which bars there are, so it is required: one entry per "
            "row of `data`. A summary of every row together, with no clusters to "
            "draw, is what summarize_descriptive_stats() returns without one."
        )
    if group_lv is None:
        if isinstance(group, pd.Categorical):
            group_lv = list(group.categories)
        else:
            group_lv = sorted(set(str(g) for g in group))
    group_lv = sa_control_first(group_lv, control_label)

    input_data = sa_validate_wide_input(
        data, feats, group, group_lv, min_levels=2
    )
    if input_data["n_dropped"] > 0:
        print(
            f"Dropped {input_data['n_dropped']} row(s) belonging to a level "
            "outside `group_lv`."
        )
    lv = list(input_data["group"].categories)
    summ = summarize_descriptive_stats(
        input_data["data"],
        input_data["feats"],
        input_data["group"],
        lv,
    )
    return {"feats": input_data["feats"], "lv": lv, "summ": summ}


def _sa_bar_interval(
    summ: pd.DataFrame,
    mainbar: str,
    errorbar: str,
    conf_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(summ)
    if errorbar == "none":
        return np.full(n, np.nan), np.full(n, np.nan)

    if errorbar == "se":
        half = summ["se"].to_numpy(dtype=float)
    elif errorbar == "sd":
        half = summ["sd"].to_numpy(dtype=float)
    elif mainbar == "mean":
        half = np.where(
            summ["n"].to_numpy() > 1,
            stats.t.ppf(1 - (1 - conf_level) / 2, np.maximum(summ["n"] - 1, 1))
            * summ["se"].to_numpy(),
            np.nan,
        )
    else:
        half = 1.58 * summ["iqr"].to_numpy() / np.sqrt(summ["n"].to_numpy())

    centre = summ[mainbar].to_numpy(dtype=float)
    return centre - half, centre + half


def _sa_bar_span(
    drawn: pd.DataFrame,
    mainbar: str,
    errorbar: str,
    ylim: tuple[float, float] | None,
) -> tuple[float, float]:
    if not np.isfinite(drawn["value"]).any():
        raise ValueError(
            f'`mainbar = "{mainbar}"` is NA for every feature and group, so '
            "there is no bar to draw. summarize_descriptive_stats() returns the "
            "same column: a shape estimate needs three or four observations, and "
            "every statistic needs one."
        )
    if ylim is not None:
        return ylim

    vals = [0.0, *drawn["value"].to_numpy(dtype=float)]
    if errorbar != "none":
        vals.extend(drawn["lower"].to_numpy(dtype=float))
        vals.extend(drawn["upper"].to_numpy(dtype=float))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    span = (float(np.min(vals)), float(np.max(vals)))

    if span[0] == span[1]:
        pad = max(abs(span[0]), 1.0) * 0.1
        return span[0] - pad, span[1] + pad

    pad_lo = -0.04 * (span[1] - span[0]) if span[0] < 0 else 0.0
    pad_hi = 0.04 * (span[1] - span[0]) if span[1] > 0 else 0.0
    return span[0] + pad_lo, span[1] + pad_hi


def _sa_bar_values(
    input_data: dict[str, Any],
    mainbar: str,
    errorbar: str,
    conf_level: float,
) -> pd.DataFrame:
    summ = input_data["summ"]
    lower, upper = _sa_bar_interval(summ, mainbar, errorbar, conf_level)
    return pd.DataFrame(
        {
            "features": summ["features"],
            "group": summ["group"].astype(str),
            "n": summ["n"],
            "value": summ[mainbar],
            "lower": lower,
            "upper": upper,
        }
    )


def draw_grouped_barplot(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Any | None = None,
    group_lv: list[str] | None = None,
    control_label: str | None = None,
    mainbar: str = "mean",
    errorbar: str = "none",
    conf_level: float = 0.95,
    gap: float = 1.0,
    lwd: float = 1.5,
    col: list[str] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    ylim: tuple[float, float] | None = None,
    dark: bool = False,
    grid_lty: str = "-",
    grid_lwd: float = 0.25,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    cex_legend: float = 1.1,
    out_statistics: bool = True,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
) -> pd.DataFrame | None:
    if mainbar not in MAINBARS:
        raise ValueError(f"`mainbar` must be one of: {', '.join(MAINBARS)}.")
    if errorbar not in ERRORBARS:
        raise ValueError(f"`errorbar` must be one of: {', '.join(ERRORBARS)}.")
    _sa_bar_check_pair(mainbar, errorbar)

    sa_check_scalar_num(
        conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True
    )
    sa_check_scalar_num(gap, "gap", 0)
    sa_check_scalar_num(lwd, "lwd", 0, lower_open=True)
    sa_check_scalar_num(cex_lab, "cex_lab", 0, lower_open=True)
    sa_check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    sa_check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    sa_check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    sa_check_scalar_num(grid_lwd, "grid_lwd", 0)
    sa_check_flag(dark, "dark")
    sa_check_flag(out_statistics, "out_statistics")
    ylim = sa_check_lim(ylim, "ylim")

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    input_data = _sa_bar_input(data, feats, group, group_lv, control_label)
    drawn = _sa_bar_values(input_data, mainbar, errorbar, conf_level)

    feats = input_data["feats"]
    lv = input_data["lv"]
    n_lv = len(lv)
    n_feat = len(feats)

    m = drawn["value"].to_numpy(dtype=float).reshape(n_feat, n_lv).T
    span = _sa_bar_span(drawn, mainbar, errorbar, ylim)
    cols = col if col is not None else dark2_colors(n_lv)
    if len(cols) < n_lv:
        cols = (cols * ((n_lv // len(cols)) + 1))[:n_lv]
    theme = sa_plot_theme(dark)
    if ylab is None:
        ylab = mainbar

    created = ax is None
    if created:
        fig = plt.figure(figsize=(10, 6), facecolor=theme["bg"])
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        leg_ax = fig.add_subplot(gs[0, 1])
    else:
        leg_ax = None

    ax.set_facecolor(theme["bg"])
    ax.yaxis.grid(True, color=theme["guide"], linestyle=grid_lty, linewidth=grid_lwd)
    ax.set_axisbelow(True)

    x = np.arange(n_feat) * (n_lv + gap)
    bar_w = 0.8
    bar_positions = []
    for j in range(n_lv):
        pos = x + j * bar_w
        bar_positions.append(pos)
        ax.bar(
            pos,
            m[j, :],
            width=bar_w,
            color=cols[j],
            edgecolor="none",
            label=lv[j],
        )

    if errorbar != "none":
        at = np.array(bar_positions).T.reshape(-1)
        lower = drawn["lower"].to_numpy(dtype=float)
        upper = drawn["upper"].to_numpy(dtype=float)
        values = drawn["value"].to_numpy(dtype=float)
        has_bar = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        if has_bar.any():
            yerr = np.vstack([values[has_bar] - lower[has_bar], upper[has_bar] - values[has_bar]])
            ax.errorbar(
                at[has_bar],
                values[has_bar],
                yerr=yerr,
                fmt="none",
                ecolor=theme["fg"],
                elinewidth=lwd,
                capsize=3,
            )

    if span[0] < 0:
        ax.axhline(0, color=theme["fg"], linewidth=lwd)

    ax.set_xticks(x + (n_lv - 1) * bar_w / 2)
    ax.set_xticklabels(feats, fontsize=10 * cex_axis)
    ax.set_ylim(span)
    ax.set_xlabel(xlab or "", fontsize=11 * cex_lab)
    ax.set_ylabel(ylab, fontsize=11 * cex_lab)
    if main:
        ax.set_title(main, fontsize=12 * cex_main, color=theme["fg"])
    ax.tick_params(colors=theme["fg"], labelsize=10 * cex_axis)

    if created and leg_ax is not None:
        leg_ax.set_facecolor(theme["bg"])
        leg_ax.axis("off")
        leg_ax.legend(
            lv,
            loc="center",
            frameon=False,
            fontsize=10 * cex_legend,
            labelcolor=theme["fg"],
        )

    if not out_statistics:
        return None

    drawn.attrs = {"mainbar": mainbar, "errorbar": errorbar}
    return drawn
