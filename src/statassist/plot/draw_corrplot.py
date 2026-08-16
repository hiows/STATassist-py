"""Correlation matrix heatmap with optional significance masking."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from statassist.plot.draw_heatmap import draw_heatmap
from statassist.utils.validate import sa_check_flag, sa_check_scalar_num


def _sa_corrplot_input(
    cor_matrix: Any,
    method: str | None,
    pvalue: np.ndarray | pd.DataFrame | None,
    use_adjusted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    is_result = (
        isinstance(cor_matrix, dict)
        and "design" in cor_matrix
        and isinstance(cor_matrix["design"], dict)
        and "methods" in cor_matrix["design"]
    )

    if is_result:
        if pvalue is not None:
            raise ValueError(
                "`pvalue` cannot be given for a `summarize_association_stats()` "
                "result: the p-values come from the same slot as the coefficients. "
                "Use `use_adjusted` to choose between them."
            )
        methods = cor_matrix["design"]["methods"]
        if method is None:
            method = methods[0]
        elif not isinstance(method, str) or method not in methods:
            raise ValueError(
                f"`method` must name one of the methods `cor_matrix` holds: "
                f"{', '.join(methods)}."
            )
        slot = cor_matrix[method]
        corr = slot["corr"]
        pvalue = slot["adj_pvalue"] if use_adjusted else slot["pvalue"]
    else:
        if method is not None:
            raise ValueError(
                "`method` names a slot of a `summarize_association_stats()` result, "
                "and `cor_matrix` is a matrix. Leave it NULL."
            )
        corr = cor_matrix

    if isinstance(corr, np.ndarray):
        corr = pd.DataFrame(corr)
    if not isinstance(corr, pd.DataFrame):
        raise ValueError(
            "`cor_matrix` must be a numeric correlation matrix or the result of "
            "`summarize_association_stats()`."
        )
    corr = corr.astype(float)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError(
            f"`cor_matrix` must be square, but is {corr.shape[0]} by "
            f"{corr.shape[1]}."
        )
    if corr.shape[1] < 2:
        raise ValueError(
            f"`draw_corrplot()` needs at least 2 features to draw, but got "
            f"{corr.shape[1]}."
        )
    if not np.allclose(corr.to_numpy(), corr.to_numpy().T, equal_nan=True):
        raise ValueError(
            "`cor_matrix` must be symmetric: a correlation between two features "
            "is one number, so the two cells that hold it must agree."
        )
    finite = corr.to_numpy()[np.isfinite(corr.to_numpy())]
    if finite.size > 0 and (finite.min() < -1 or finite.max() > 1):
        raise ValueError(
            "`cor_matrix` holds value(s) outside [-1, 1], so it is not a matrix "
            "of correlations."
        )

    if corr.columns is None or corr.columns.tolist() == list(range(corr.shape[1])):
        names = [f"V{i + 1}" for i in range(corr.shape[1])]
        corr.index = names
        corr.columns = names
    else:
        corr.index = corr.columns

    if pvalue is not None:
        if isinstance(pvalue, np.ndarray):
            pvalue = pd.DataFrame(pvalue, index=corr.index, columns=corr.columns)
        if pvalue.shape != corr.shape:
            raise ValueError(
                f"`pvalue` must be a numeric matrix laid out like `cor_matrix`: "
                f"{corr.shape[0]} by {corr.shape[1]}."
            )
        if list(pvalue.columns) != list(corr.columns):
            raise ValueError(
                "`pvalue` must name the same features as `cor_matrix`, in the same "
                "order."
            )
        pvalue = pvalue.astype(float)
        pvalue.index = corr.index
        pvalue.columns = corr.columns

    return corr, pvalue


def draw_corrplot(
    cor_matrix: Any,
    method: str | None = None,
    pvalue: np.ndarray | pd.DataFrame | None = None,
    use_adjusted: bool = True,
    sig_level: float = 0.05,
    cluster: bool = True,
    hclust_method: str = "average",
    zlim: tuple[float, float] = (-1.0, 1.0),
    anno: bool = True,
    main: str | None = None,
    cex_anno: float = 1.0,
    cex_axis: float = 0.9,
    cex_main: float = 1.5,
    cex_legend: float = 1.2,
    ax: plt.Axes | None = None,
    fig: plt.Figure | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if hclust_method not in ("average", "complete", "ward.D2"):
        raise ValueError(
            "`hclust_method` must be 'average', 'complete', or 'ward.D2'."
        )
    sa_check_flag(use_adjusted, "use_adjusted")
    sa_check_flag(cluster, "cluster")
    sa_check_flag(anno, "anno")
    sa_check_scalar_num(sig_level, "sig_level", 0, 1, lower_open=True)
    sa_check_scalar_num(cex_anno, "cex_anno", 0, lower_open=True)
    sa_check_scalar_num(cex_axis, "cex_axis", 0, lower_open=True)
    sa_check_scalar_num(cex_main, "cex_main", 0, lower_open=True)
    sa_check_scalar_num(cex_legend, "cex_legend", 0, lower_open=True)

    corr, pvalue = _sa_corrplot_input(cor_matrix, method, pvalue, use_adjusted)
    feats = list(corr.columns)
    p = len(feats)

    ord_idx = list(range(p))
    hc = None
    if cluster:
        dist = 1.0 - corr.to_numpy()
        dist = np.clip(dist, 0.0, None)
        d = squareform(dist, checks=False)
        if np.any(~np.isfinite(d)):
            print(
                "Some pair of features has no correlation to measure a distance "
                "between, so the features are drawn in the order they arrived."
            )
        else:
            link_method = "ward" if hclust_method == "ward.D2" else hclust_method
            hc = hierarchy.linkage(d, method=link_method)
            ord_idx = list(hierarchy.leaves_list(hc))

    drawn = corr.iloc[ord_idx, ord_idx].copy()
    pv = pvalue.iloc[ord_idx, ord_idx].copy() if pvalue is not None else None

    n_masked = 0
    if pv is not None:
        blank = pv.to_numpy() > sig_level
        np.fill_diagonal(blank, False)
        blank = blank & np.isfinite(pv.to_numpy())
        n_masked = int(blank.sum())
        masked_vals = drawn.to_numpy().copy()
        masked_vals[blank] = np.nan
        drawn = pd.DataFrame(masked_vals, index=drawn.index, columns=drawn.columns)

    heatmap_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "data",
            "group",
            "group_lv",
            "scale",
            "cluster_feats",
            "cluster_samples",
            "anno",
            "cex_anno",
            "cex_axis",
            "cex_main",
            "cex_legend",
        )
    }

    out = draw_heatmap(
        data=drawn,
        group=None,
        group_lv=None,
        scale="none",
        zlim=zlim,
        cluster_feats=False,
        cluster_samples=False,
        anno=anno,
        cex_anno=cex_anno,
        main=main,
        cex_axis=cex_axis,
        cex_main=cex_main,
        cex_legend=cex_legend,
        ax=ax,
        fig=fig,
        **heatmap_kwargs,
    )

    out["corr"] = drawn
    out["pvalue"] = pv
    out["order"] = ord_idx
    out["hclust"] = hc
    out["n_masked"] = n_masked
    out["feats"] = feats
    return out
