"""Scatter plot for dimensionality reduction results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_dim_reduction_plot(
    reduction_result: dict[str, Any],
    *,
    dims: tuple[int, int] = (1, 2),
    cluster_result: dict[str, Any] | None = None,
    group: pd.Series | np.ndarray | None = None,
    dark: bool = False,
) -> None:
    if reduction_result.get("__class__", ("",))[0] != "sa_reduction":
        raise ValueError(
            "`reduction_result` must be a reduction from perform_pca(), perform_tsne() or perform_umap()."
        )

    scores = reduction_result["scores"]
    coord_cols = [c for c in scores.columns if c != "points"]
    d1, d2 = dims[0] - 1, dims[1] - 1
    if d1 < 0 or d2 < 0 or max(d1, d2) >= len(coord_cols):
        raise ValueError("`dims` asks for coordinates outside the reduction result.")

    x = scores[coord_cols[d1]].to_numpy()
    y = scores[coord_cols[d2]].to_numpy()

    bg = "#1a1a1a" if dark else "white"
    fg = "white" if dark else "black"
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=bg)
    ax.set_facecolor(bg)

    colours = None
    if cluster_result is not None:
        assign = cluster_result["assignments"].set_index("points")
        pts = scores["points"]
        clusters = assign.reindex(pts)["cluster"].fillna(0).astype(int).to_numpy()
        unique = sorted(set(clusters))
        cmap = plt.cm.tab10
        colours = [cmap((i % 10) / 10) if c > 0 else "#888888" for i, c in enumerate(clusters)]

    _PCH = {16: "o", 17: "^", 15: "s", 18: "D", 8: "*", 1: ".", 2: "+", 0: "x", 5: "v", 6: "<"}
    pch = 16
    if group is not None:
        group = np.asarray(group)
        unique_g = pd.unique(group)
        markers = [16, 17, 15, 18, 8, 1, 2, 0, 5, 6]
        for gi, gval in enumerate(unique_g):
            mask = group == gval
            cvals = colours[mask] if colours is not None else None
            ax.scatter(
                x[mask],
                y[mask],
                c=cvals if cvals is not None else None,
                marker=_PCH.get(markers[gi % len(markers)], "o"),
                s=40,
                label=str(gval),
            )
        ax.legend(facecolor=bg, labelcolor=fg)
    else:
        ax.scatter(x, y, c=colours, s=40)

    xlab = coord_cols[d1]
    ylab = coord_cols[d2]
    if reduction_result.get("variance") is not None:
        var = reduction_result["variance"]
        if d1 < len(var):
            xlab = f"{xlab} ({var.iloc[d1]['prop_var']:.2f}%)"
        if d2 < len(var):
            ylab = f"{ylab} ({var.iloc[d2]['prop_var']:.2f}%)"

    ax.set_xlabel(xlab, color=fg)
    ax.set_ylabel(ylab, color=fg)
    ax.set_title(reduction_result["analysis"], color=fg)
    ax.tick_params(colors=fg)
    fig.tight_layout()
    if plt.get_backend().lower() != "agg":
        plt.show()
    return fig
