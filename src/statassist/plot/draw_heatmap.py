"""Clustered heatmap of features by samples (matplotlib port)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from statassist.plot._theme import dark2_colors
from statassist.utils.validate import (
    sa_check_count,
    sa_check_flag,
    sa_check_range,
    sa_check_scalar_num,
    sa_validate_wide_input,
)


def _scale_matrix(x: np.ndarray, scale: str) -> np.ndarray:
    if scale == "none":
        return x.copy()
    axis = 0 if scale == "feature" else 1
    centre = np.nanmean(x, axis=axis, keepdims=True)
    spread = np.nanstd(x, axis=axis, ddof=1, keepdims=True)
    flat = ~np.isfinite(spread) | (spread == 0)
    spread = np.where(flat, 1.0, spread)
    out = (x - centre) / spread
    n_flat = int(np.sum(flat & np.isfinite(centre.squeeze())))
    if n_flat > 0:
        print(
            f"{n_flat} {scale}(s) have no variance to scale by and are "
            "drawn flat at the middle of the colour scale."
        )
    return out


def _cluster_dist(x: np.ndarray, dist_method: str) -> np.ndarray:
    if dist_method == "correlation":
        corr = np.corrcoef(x)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        d = 1.0 - corr
        return squareform(np.clip(d, 0.0, None), checks=False)
    return pdist(x, metric=dist_method)


def _heatmap_hclust(
    x: np.ndarray,
    dist_method: str,
    hclust_method: str,
    axis: str,
) -> np.ndarray | None:
    method = "ward" if hclust_method == "ward.D2" else hclust_method
    d = _cluster_dist(x, dist_method)
    if not np.all(np.isfinite(d)):
        print(
            f"Not clustering the {axis}s: some distances are undefined, "
            "which happens when a pair shares no observation or has no "
            "variance. The input order is kept."
        )
        return None
    return hierarchy.linkage(d, method=method)


def _draw_color_key(
    ax: plt.Axes,
    cmap: LinearSegmentedColormap,
    zlim: tuple[float, float],
    group_colors: dict[str, str] | None,
    cex_axis: float,
    cex_legend: float,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    grad = np.linspace(zlim[0], zlim[1], 101).reshape(101, 1)
    ax.imshow(
        grad,
        aspect="auto",
        origin="lower",
        extent=(0.35, 0.55, 0.15, 0.85),
        cmap=cmap,
        vmin=zlim[0],
        vmax=zlim[1],
    )
    for t in np.linspace(zlim[0], zlim[1], 5):
        y = 0.15 + (t - zlim[0]) / (zlim[1] - zlim[0]) * 0.7
        ax.text(0.58, y, f"{t:g}", va="center", ha="left", fontsize=9 * cex_axis)
    if group_colors:
        y = 0.85
        ax.text(0.72, y, "group", fontweight="bold", fontsize=10 * cex_legend)
        for i, (name, col) in enumerate(group_colors.items()):
            yy = y - 0.08 * (i + 1)
            ax.add_patch(plt.Rectangle((0.72, yy - 0.02), 0.04, 0.04, color=col))
            ax.text(0.78, yy, name, va="center", fontsize=9 * cex_legend)


def _prepare_wide_matrix(
    data: pd.DataFrame | np.ndarray,
    group: Any | None,
    group_lv: list[str] | None,
    feats: list[str] | None,
    feat_labels: list[str] | None,
    sample_labels: list[str] | None,
) -> tuple[np.ndarray, list[str], list[str], Any, dict[str, str] | None]:
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if feats is None:
        feats = list(data.columns)
    if sample_labels is None:
        sample_labels = (
            list(data.index.astype(str))
            if data.index is not None
            else [str(i) for i in range(len(data))]
        )
    input_data = sa_validate_wide_input(
        data, list(feats), group, group_lv, min_levels=1
    )
    feats = input_data["feats"]
    group = input_data["group"]
    sample_labels = input_data["id"] or sample_labels
    if input_data["n_dropped"] > 0:
        print(
            f"Dropped {input_data['n_dropped']} row(s) belonging to a level "
            "outside `group_lv`."
        )
    if feat_labels is not None and len(feat_labels) != len(feats):
        raise ValueError(
            f"`feat_labels` must have one entry per feature in `feats`: got "
            f"{len(feat_labels)} for {len(feats)} feature(s)."
        )
    m = input_data["data"][feats].to_numpy(dtype=float).T
    rownames = [str(x) for x in (feat_labels or feats)]
    colnames = [str(x) for x in sample_labels]
    if group is not None:
        lv = list(group.categories)
        group_colors = dict(zip(lv, dark2_colors(len(lv))))
    else:
        group_colors = None
    return m, rownames, colnames, group, group_colors


def draw_heatmap(
    data: pd.DataFrame | np.ndarray,
    group: Any | None = None,
    group_lv: list[str] | None = None,
    feats: list[str] | None = None,
    scale: str = "feature",
    zlim: tuple[float, float] | None = None,
    dist_method: str = "euclidean",
    hclust_method: str = "average",
    cluster_feats: bool = True,
    cluster_samples: bool = True,
    feat_labels: list[str] | None = None,
    sample_labels: list[str] | None = None,
    show_feat_names: bool = True,
    show_sample_names: bool = True,
    anno: bool = False,
    cex_anno: float = 1.0,
    n_colors: int = 101,
    main: str | None = None,
    cex_axis: float = 0.9,
    cex_main: float = 1.5,
    cex_legend: float = 1.2,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
) -> dict[str, Any]:
    """Draw a clustered heatmap; returns layout metadata."""
    if scale not in ("feature", "sample", "none"):
        raise ValueError("`scale` must be 'feature', 'sample', or 'none'.")
    if dist_method not in ("euclidean", "correlation", "manhattan"):
        raise ValueError(
            "`dist_method` must be 'euclidean', 'correlation', or 'manhattan'."
        )
    if hclust_method not in ("average", "complete", "ward.D2"):
        raise ValueError(
            "`hclust_method` must be 'average', 'complete', or 'ward.D2'."
        )
    sa_check_flag(cluster_feats, "cluster_feats")
    sa_check_flag(cluster_samples, "cluster_samples")
    sa_check_flag(show_feat_names, "show_feat_names")
    sa_check_flag(show_sample_names, "show_sample_names")
    sa_check_flag(anno, "anno")
    sa_check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    sa_check_count(n_colors, "n_colors", 3)
    sa_check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    sa_check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    sa_check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)
    if zlim is not None:
        zlim = sa_check_range(zlim, "zlim")
        if zlim[0] == zlim[1]:
            raise ValueError(
                f"`zlim` must have two different ends, but both are {zlim[0]}."
            )
    if (group is None) ^ (group_lv is None):
        raise ValueError(
            "`group` and `group_lv` must both be supplied or both be `NULL`."
        )

    if isinstance(data, pd.DataFrame) and feats is None and group is None:
        arr = data.to_numpy(dtype=float)
        if arr.shape[0] == arr.shape[1]:
            m_raw = arr
            names = [str(c) for c in data.columns]
            rownames = feat_labels or names
            colnames = rownames
            group_colors = None
        else:
            m_raw, rownames, colnames, group, group_colors = _prepare_wide_matrix(
                data, group, group_lv, feats, feat_labels, sample_labels
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == data.shape[1]:
        m_raw = np.asarray(data, dtype=float)
        names = [f"V{i + 1}" for i in range(m_raw.shape[0])]
        rownames = feat_labels or names
        colnames = rownames
        group_colors = None
    else:
        m_raw, rownames, colnames, group, group_colors = _prepare_wide_matrix(
            data, group, group_lv, feats, feat_labels, sample_labels
        )

    n_feats, n_samples = m_raw.shape
    if n_feats < 2 or n_samples < 2:
        raise ValueError(
            f"`draw_heatmap()` needs at least 2 features and 2 samples to "
            f"cluster and draw, but got {n_feats} feature(s) and "
            f"{n_samples} sample(s)."
        )

    m_scaled = _scale_matrix(m_raw, scale)
    feat_hc = (
        _heatmap_hclust(m_scaled, dist_method, hclust_method, "feature")
        if cluster_feats
        else None
    )
    sample_hc = (
        _heatmap_hclust(m_scaled.T, dist_method, hclust_method, "sample")
        if cluster_samples
        else None
    )
    feat_order = (
        list(hierarchy.leaves_list(feat_hc))
        if feat_hc is not None
        else list(range(n_feats))
    )
    sample_order = (
        list(hierarchy.leaves_list(sample_hc))
        if sample_hc is not None
        else list(range(n_samples))
    )
    m_plot = m_scaled[np.ix_(feat_order, sample_order)]
    feat_names = [rownames[i] for i in feat_order]
    sample_names = [colnames[i] for i in sample_order]

    finite = m_plot[np.isfinite(m_plot)]
    if finite.size == 0:
        raise ValueError("`data` holds no finite value to draw.")
    if zlim is None:
        if (finite > 0).any() and (finite < 0).any():
            bound = float(np.max(np.abs(finite)))
            zlim = (-bound, bound)
        else:
            zlim = (float(np.min(finite)), float(np.max(finite)))
        if zlim[0] == zlim[1]:
            zlim = (zlim[0] - 0.5, zlim[1] + 0.5)

    z_draw = m_plot.copy()
    z_draw[np.isfinite(z_draw) & (z_draw < zlim[0])] = zlim[0]
    z_draw[np.isfinite(z_draw) & (z_draw > zlim[1])] = zlim[1]

    cmap = LinearSegmentedColormap.from_list("bwr", ["blue", "white", "red"], N=n_colors)
    bounds = np.linspace(zlim[0], zlim[1], n_colors + 1)
    norm = BoundaryNorm(bounds, n_colors)

    created = ax is None
    if created:
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        key_ax = fig.add_subplot(gs[0, 1])
    else:
        key_ax = None

    masked = np.ma.masked_invalid(z_draw)
    im = ax.imshow(
        masked,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(sample_names)))
    ax.set_yticks(range(len(feat_names)))
    ax.set_xticklabels(
        sample_names if show_sample_names else [""] * len(sample_names),
        rotation=90,
        fontsize=10 * cex_axis,
    )
    ax.set_yticklabels(
        feat_names if show_feat_names else [""] * len(feat_names),
        fontsize=10 * cex_axis,
    )
    if main:
        ax.set_title(main, fontsize=12 * cex_main)

    if anno:
        for i in range(m_plot.shape[0]):
            for j in range(m_plot.shape[1]):
                v = m_plot[i, j]
                if not np.isfinite(v):
                    continue
                t = (v - zlim[0]) / (zlim[1] - zlim[0]) if zlim[1] != zlim[0] else 0.5
                color = "#262626" if 0.28 < t < 0.72 else "white"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8 * cex_anno,
                )

    for i in range(m_plot.shape[0]):
        for j in range(m_plot.shape[1]):
            if np.isnan(m_plot[i, j]):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#EBEBEB",
                        edgecolor="none",
                        zorder=10,
                    )
                )

    if created and key_ax is not None:
        _draw_color_key(key_ax, cmap, zlim, group_colors, cex_axis, cex_legend)

    matrix = pd.DataFrame(m_plot, index=feat_names, columns=sample_names)
    return {
        "matrix": matrix,
        "feat_order": feat_order,
        "sample_order": sample_order,
        "feat_hclust": feat_hc,
        "sample_hclust": sample_hc,
        "zlim": zlim,
        "group_colors": group_colors,
        "ax": ax,
        "fig": fig,
        "im": im,
    }
