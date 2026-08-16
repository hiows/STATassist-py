"""Draw a volcano plot from a significance verdict."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statassist.contracts.significance import SignificanceResult, sa_significance
from statassist.utils.validate import sa_check_flag, sa_check_lim, sa_check_scalar_num


def draw_volcano_plot(
    significance_result: SignificanceResult | pd.DataFrame | dict[str, pd.DataFrame],
    *,
    terms: list[str] | None = None,
    panel_nrow: int | None = None,
    use_adjusted: bool = True,
    log2fc_cutoff: float | None = None,
    pval_cutoff: float | None = None,
    anno_feats: bool = True,
    anno_top: int = 10,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xlab: str | None = None,
    main: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes | list[plt.Axes]:
    sa_check_flag(use_adjusted, "use_adjusted")
    sa_check_flag(anno_feats, "anno_feats")
    sa_check_scalar_num(anno_top, "anno_top", 0)
    sa_check_lim(xlim, "xlim")
    sa_check_lim(ylim, "ylim")

    if isinstance(significance_result, SignificanceResult) or isinstance(
        significance_result, sa_significance
    ):
        sig = significance_result.significance
    else:
        sig = significance_result

    if isinstance(sig, dict):
        keys = list(sig.keys()) if terms is None else terms
        axes = []
        n = len(keys)
        n_row = panel_nrow or 1
        n_col = int(np.ceil(n / n_row))
        fig, axes_arr = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
        axes_flat = np.atleast_1d(axes_arr).ravel()
        for i, k in enumerate(keys):
            draw_volcano_plot(
                sig[k],
                use_adjusted=use_adjusted,
                log2fc_cutoff=log2fc_cutoff,
                pval_cutoff=pval_cutoff,
                anno_feats=anno_feats,
                anno_top=anno_top,
                xlim=xlim,
                ylim=ylim,
                main=k if main is None else main,
                ax=axes_flat[i],
            )
        if main:
            fig.suptitle(main)
        return list(axes_flat[:n])

    tbl = sig.copy()
    p_col = "adj_pvalue" if use_adjusted else "pvalue"
    lfc = tbl["log2fc"].to_numpy(dtype=float)
    pvals = tbl[p_col].to_numpy(dtype=float)
    y = -np.log10(np.clip(pvals, np.finfo(float).tiny, 1))

    attrs = getattr(tbl, "attrs", {})
    lfc_cut = log2fc_cutoff if log2fc_cutoff is not None else attrs.get("log2fc_cutoff", 1.0)
    p_cut = pval_cutoff if pval_cutoff is not None else attrs.get("pval_cutoff", 0.05)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lfc, y, c="gray", alpha=0.7, s=20, **kwargs)
    sig_mask = (np.abs(lfc) >= lfc_cut) & (pvals <= p_cut)
    ax.scatter(lfc[sig_mask], y[sig_mask], c="#D1495B", s=30)

    ax.axvline(lfc_cut, color="#666", ls="--", lw=0.8)
    ax.axvline(-lfc_cut, color="#666", ls="--", lw=0.8)
    ax.axhline(-np.log10(p_cut), color="#666", ls="--", lw=0.8)

    if anno_feats and sig_mask.any():
        idx = np.where(sig_mask)[0]
        order = idx[np.argsort(-y[idx])[:anno_top]]
        for j in order:
            ax.annotate(tbl["features"].iloc[j], (lfc[j], y[j]), fontsize=8)

    ax.set_xlabel(xlab or "log2 FC")
    ax.set_ylabel("-log10(p)" + (" (adj)" if use_adjusted else ""))
    if main:
        ax.set_title(main)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    return ax
