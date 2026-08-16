"""Draw a mosaic plot from a categorical comparison result."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statassist.contracts.categorical import CategoricalResult, sa_categorical, sa_null_label
from statassist.plot._theme import sa_plot_theme


def draw_mosaic_plot(
    categorical_result: CategoricalResult,
    *,
    dark: bool = False,
    main: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    if not isinstance(categorical_result, (CategoricalResult, sa_categorical)):
        raise ValueError("`categorical_result` must be a categorical comparison result.")

    cells = categorical_result.cells
    row_lv = cells["row_level"].unique().tolist()
    col_lv = cells["col_level"].unique().tolist()
    mat = np.zeros((len(row_lv), len(col_lv)))
    resid = np.zeros_like(mat)
    for _, r in cells.iterrows():
        i = row_lv.index(r["row_level"])
        j = col_lv.index(r["col_level"])
        mat[i, j] = r["observed"]
        resid[i, j] = r["residual"] if pd.notna(r["residual"]) else 0

    theme = sa_plot_theme(dark)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor(theme["bg"])

    total = mat.sum()
    x0 = 0.0
    for j, cl in enumerate(col_lv):
        col_sum = mat[:, j].sum()
        w = col_sum / total if total else 0
        y0 = 0.0
        for i, rl in enumerate(row_lv):
            val = mat[i, j]
            h = val / total if total else 0
            color = plt.cm.RdBu_r(0.5 + 0.25 * np.tanh(resid[i, j]))
            ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="white"))
            if h > 0.02:
                ax.text(x0 + w / 2, y0 + h / 2, int(val), ha="center", va="center", fontsize=8)
            y0 += h
        ax.text(x0 + w / 2, -0.02, cl, ha="center", va="top", transform=ax.transAxes, fontsize=9)
        x0 += w

    for i, rl in enumerate(row_lv):
        ax.text(-0.02, (i + 0.5) / len(row_lv), rl, ha="right", va="center", transform=ax.transAxes)

    null = categorical_result.design.get("null", "independence")
    ax.set_title(main or sa_null_label(null))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax
