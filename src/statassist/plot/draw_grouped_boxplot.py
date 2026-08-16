"""Grouped boxplot across several features (matplotlib port)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from statassist.plot._theme import dark2_colors, sa_plot_theme
from statassist.utils.validate import (
    sa_check_count,
    sa_check_flag,
    sa_check_lim,
    sa_check_scalar_num,
    sa_na_row,
    sa_validate_wide_input,
)

BOX_ROWS = (
    "min",
    "lower_bound",
    "Q1",
    "median",
    "Q3",
    "upper_bound",
    "max",
    "n",
    "lower_conf",
    "upper_conf",
)
SUMMARY_ROWS = (
    "min",
    "lower_bound",
    "Q1",
    "median",
    "Q3",
    "upper_bound",
    "max",
)
CONF_ROWS = ("n", "lower_conf", "upper_conf")


def sa_box_stats(values: dict[str, np.ndarray]) -> pd.DataFrame:
    cols = {}
    for name, raw in values.items():
        v = np.asarray(raw, dtype=float)
        v = v[np.isfinite(v)]
        n = v.size
        if n == 0:
            cols[name] = sa_na_row(BOX_ROWS).to_dict()
            continue
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        notch = 1.58 * iqr / np.sqrt(n)
        cols[name] = {
            "min": float(np.min(v)),
            "lower_bound": float(q1 - 1.5 * iqr),
            "Q1": float(q1),
            "median": float(med),
            "Q3": float(q3),
            "upper_bound": float(q3 + 1.5 * iqr),
            "max": float(np.max(v)),
            "n": float(n),
            "lower_conf": float(med - notch),
            "upper_conf": float(med + notch),
        }
    return pd.DataFrame(cols)


def _sa_box_one_factor(
    data: pd.DataFrame,
    feats: list[str],
    group: Any,
    group_lv: list[str] | None,
) -> dict[str, Any]:
    if group_lv is None:
        if isinstance(group, pd.Categorical):
            group_lv = list(group.categories)
        else:
            group_lv = sorted(set(str(g) for g in group))
    input_data = sa_validate_wide_input(
        data, feats, group, group_lv, min_levels=2
    )
    if input_data["n_dropped"] > 0:
        print(
            f"Dropped {input_data['n_dropped']} row(s) belonging to a level "
            "outside `group_lv`."
        )
    lv = list(input_data["group"].categories)
    samples = {}
    for f in input_data["feats"]:
        samples[f] = {
            level: input_data["data"].loc[input_data["group"] == level, f].to_numpy()
            for level in lv
        }
    return {
        "feats": input_data["feats"],
        "lv": lv,
        "samples": samples,
        "groups": [{"label": None, "cols": list(range(len(lv)))}],
        "legend_title": None,
    }


def _cluster_positions(n_cluster: int, n_lv: int, gap: float) -> list[list[float]]:
    positions = []
    for i in range(n_cluster):
        start = i * (n_lv + gap)
        positions.append([start + j + 1 for j in range(n_lv)])
    return positions


def draw_grouped_boxplot(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Any | None = None,
    group_lv: list[str] | None = None,
    factors: dict[str, Any] | None = None,
    factor_lv: dict[str, list[str]] | None = None,
    control_label: str | dict[str, str] | None = None,
    panel_by: str = "feature",
    panel_nrow: int | None = None,
    gap: float = 1.0,
    lwd: float = 1.5,
    xlab: str | None = None,
    ylab: str | None = None,
    cex_lab: float = 1.3,
    cex_axis: float = 1.2,
    cex_main: float = 1.3,
    ylim: tuple[float, float] | None = None,
    main: str | None = None,
    dark: bool = False,
    grid_lty: str = "-",
    grid_lwd: float = 0.25,
    cex_legend: float = 1.1,
    out_statistics: bool = True,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
) -> dict[str, pd.DataFrame] | None:
    if panel_by not in ("feature", "factor"):
        raise ValueError("`panel_by` must be 'feature' or 'factor'.")
    if panel_nrow is not None:
        sa_check_count(panel_nrow, "panel_nrow", 1)
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

    said_group = group is not None or group_lv is not None
    said_factors = factors is not None or factor_lv is not None
    if said_group and said_factors:
        raise ValueError(
            "`group` and `factors` are two ways of saying what the boxes are, so "
            "a call takes one of them: `group` with `group_lv` for a single "
            "factor, `factors` with `factor_lv` for a crossed design."
        )
    if not said_group and not said_factors:
        raise ValueError(
            "nothing says what the boxes are. Supply `group` and `group_lv` for "
            "a single factor, or `factors` for a crossed design."
        )
    if factor_lv is not None and factors is None:
        raise ValueError(
            "`factor_lv` gives the levels of the factors `factors` holds, which "
            "was not supplied."
        )
    if said_group and control_label is not None:
        raise ValueError(
            "`control_label` names a reference level per factor of a crossed "
            "design, which `factors` states. A single factor draws in `group_lv` "
            "order, so put the reference first there."
        )
    if factors is not None:
        raise NotImplementedError(
            "Crossed factorial designs for `draw_grouped_boxplot()` require "
            "`utils.factorial` (not yet ported). Use `group`/`group_lv` for a "
            "single factor."
        )

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    box_input = _sa_box_one_factor(data, feats, group, group_lv)

    feats = box_input["feats"]
    lv = box_input["lv"]
    samples = box_input["samples"]
    n_lv = len(lv)
    n_cluster = len(feats)
    positions = _cluster_positions(n_cluster, n_lv, gap)
    all_positions = [p for cluster in positions for p in cluster]
    all_boxes = []
    for f in feats:
        all_boxes.extend([samples[f][level] for level in lv])

    if ylim is None:
        vals = np.concatenate(
            [v[np.isfinite(v)] for s in samples.values() for v in s.values()]
        )
        if vals.size == 0:
            raise ValueError(
                "`feats` hold no finite value in any cell, so there is nothing to "
                "draw."
            )
        ylim = (float(np.min(vals)), float(np.max(vals)))

    theme = sa_plot_theme(dark)
    cols = dark2_colors(n_lv)
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

    bp = ax.boxplot(
        all_boxes,
        positions=all_positions,
        widths=0.6,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], cols * n_cluster):
        patch.set_facecolor(theme["bg"])
        patch.set_edgecolor(color)
        patch.set_linewidth(lwd)
    for element in ("whiskers", "caps", "medians"):
        for i, artist in enumerate(bp[element]):
            artist.set_color(cols[i % n_lv])
            artist.set_linewidth(lwd)

    cluster_centers = [float(np.mean(cluster)) for cluster in positions]
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(feats, fontsize=10 * cex_axis)
    ax.set_xlabel(xlab or "", fontsize=11 * cex_lab)
    ax.set_ylabel(ylab or "", fontsize=11 * cex_lab)
    if main:
        ax.set_title(main, fontsize=12 * cex_main, color=theme["fg"])
    ax.set_ylim(ylim)
    ax.tick_params(colors=theme["fg"], labelsize=10 * cex_axis)

    if created and leg_ax is not None:
        from matplotlib.patches import Patch

        leg_ax.set_facecolor(theme["bg"])
        leg_ax.axis("off")
        leg_ax.legend(
            handles=[Patch(facecolor=c, edgecolor="white") for c in cols],
            labels=lv,
            loc="center",
            frameon=False,
            fontsize=10 * cex_legend,
            labelcolor=theme["fg"],
        )

    if not out_statistics:
        return None

    summaries = {f: sa_box_stats(samples[f]) for f in feats}
    return {
        "box_summary_stats": {
            f: summaries[f].loc[list(SUMMARY_ROWS)] for f in feats
        },
        "median_confidence_stats": {
            f: summaries[f].loc[list(CONF_ROWS)] for f in feats
        },
    }
