"""Draw an interaction plot from a factorial comparison result."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from statassist.contracts.comparison import ComparisonResult
from statassist.plot._theme import dark2_colors, sa_plot_theme
from statassist.utils.validate import sa_check_flag


def draw_interaction_plot(
    comparison_result: ComparisonResult,
    feat: str,
    factor: str,
    *,
    line_factor: str | None = None,
    dark: bool = False,
    main: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    sa_check_flag(dark, "dark")
    if comparison_result.cells is None:
        raise ValueError(
            "`comparison_result$cells` is absent. compare_factorial_groups() is "
            "the scenario that builds it."
        )
    cells = comparison_result.cells
    sub = cells[(cells["features"] == feat)].copy()
    if factor not in sub.columns:
        raise ValueError(f"`factor` must name a factor column; got {factor!r}.")

    if line_factor is None:
        others = [c for c in sub.columns if c in comparison_result.design["factor_lv"] and c != factor]
        line_factor = others[0] if others else factor

    theme = sa_plot_theme(dark)
    colors = dark2_colors(sub[line_factor].nunique())

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor(theme["bg"])

    for i, lv in enumerate(sorted(sub[line_factor].unique())):
        part = sub[sub[line_factor] == lv].groupby(factor, as_index=False).agg(
            mean=("mean", "mean"), se=("se", "mean")
        )
        ax.errorbar(
            part[factor],
            part["mean"],
            yerr=part["se"],
            marker="o",
            color=colors[i % len(colors)],
            label=str(lv),
            capsize=3,
        )

    ax.set_xlabel(factor)
    ax.set_ylabel(f"{feat} mean")
    ax.legend(title=line_factor)
    ax.set_title(main or f"{feat}: {factor} x {line_factor}")
    return ax
