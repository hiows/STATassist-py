"""Draw a butterfly histogram comparing two groups."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statassist.plot._theme import dark2_colors, sa_plot_theme
from statassist.utils.validate import sa_validate_wide_input


def draw_butterfly_hist(
    data: pd.DataFrame | np.ndarray,
    feat: str,
    group: pd.Series | np.ndarray | list[Any],
    group_lv: list[str],
    *,
    bins: int = 30,
    dark: bool = False,
    main: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    inp = sa_validate_wide_input(data, [feat], group, group_lv, n_levels=2)
    data = inp["data"]
    group_lv = list(group_lv)[:2]
    theme = sa_plot_theme(dark)
    colors = dark2_colors(2)

    x = data[feat].to_numpy()
    g = np.asarray(inp["group"])
    v0 = x[g == group_lv[0]]
    v1 = x[g == group_lv[1]]
    v0 = v0[np.isfinite(v0)]
    v1 = v1[np.isfinite(v1)]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor(theme["bg"])

    h0, edges = np.histogram(v0, bins=bins)
    h1, _ = np.histogram(v1, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.9

    ax.barh(centers, -h0, height=width, color=colors[0], alpha=0.7, label=group_lv[0])
    ax.barh(centers, h1, height=width, color=colors[1], alpha=0.7, label=group_lv[1])
    ax.axvline(0, color=theme["guide"], lw=0.8)
    ax.set_xlabel("count")
    ax.set_ylabel(feat)
    ax.legend()
    if main:
        ax.set_title(main)
    return ax
