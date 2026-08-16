"""Draw a forest plot of a comparison result."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statassist.contracts.comparison import ComparisonResult, sa_pick_test
from statassist.plot._theme import sa_plot_theme
from statassist.utils.validate import sa_check_feat_names, sa_check_flag, sa_check_lim, sa_check_scalar_num


def _estimate_column(tbl: pd.DataFrame) -> str | None:
    for col in (
        "mean_diff",
        "hl_shift",
        "trim_diff",
        "relative_effect",
        "diff",
        "estimate",
    ):
        if col in tbl.columns:
            return col
    return None


def draw_forest_plot(
    comparison_result: ComparisonResult,
    test: str | None = None,
    *,
    type: str = "auto",
    feats: list[str] | None = None,
    use_adjusted: bool = True,
    alpha: float = 0.05,
    sort_by: str = "none",
    dark: bool = False,
    xlim: tuple[float, float] | None = None,
    xlab: str | None = None,
    main: str | None = None,
    col_signif: str = "#D1495B",
    col_plain: str = "#7F8C8D",
    **kwargs: Any,
) -> pd.DataFrame:
    if type not in ("auto", "estimate", "posthoc", "pvalue"):
        raise ValueError('`type` must be one of "auto", "estimate", "posthoc", or "pvalue".')
    if sort_by not in ("none", "pvalue"):
        raise ValueError('`sort_by` must be one of "none" or "pvalue".')
    sa_check_flag(dark, "dark")
    sa_check_flag(use_adjusted, "use_adjusted")
    sa_check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)
    sa_check_lim(xlim, "xlim")

    if test is None:
        test = next(iter(comparison_result.tests))
    tbl = sa_pick_test(comparison_result, test)
    posthoc = (
        comparison_result.posthoc.get(test)
        if comparison_result.posthoc
        else None
    )
    p_col = "pval_adj" if use_adjusted else "pval"

    if feats is not None:
        sa_check_feat_names(feats)
        tbl = tbl.loc[tbl["features"].isin(feats)].copy()
        tbl = tbl.set_index("features").loc[feats].reset_index()
        if posthoc is not None and len(posthoc):
            posthoc = posthoc[posthoc["features"].isin(feats)]

    est_col = _estimate_column(tbl)
    has_estimate = est_col is not None and tbl[["lower_conf", "upper_conf"]].notna().any().any()
    has_posthoc = posthoc is not None and len(posthoc) > 0

    view = type
    if view == "auto":
        view = "estimate" if has_estimate else ("posthoc" if has_posthoc else "pvalue")

    theme = sa_plot_theme(dark)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(tbl))))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    if view == "posthoc":
        drawn = posthoc if feats else posthoc[posthoc["features"] == posthoc["features"].iloc[0]]
        labels = (
            drawn["contrast"].tolist()
            if drawn["features"].nunique() == 1
            else (drawn["features"] + ": " + drawn["contrast"]).tolist()
        )
        estimate = drawn["estimate"].to_numpy()
        lower = drawn["lower_conf"].to_numpy()
        upper = drawn["upper_conf"].to_numpy()
        pvals = drawn[p_col].to_numpy()
        null_value = 0.0
        xlabel = xlab or "estimate (group1 - group2)"
        title = main or comparison_result.test_info[test].get("posthoc_label", test)
    elif view == "estimate":
        drawn = tbl
        labels = drawn["features"].tolist()
        estimate = drawn[est_col].to_numpy()
        lower = drawn["lower_conf"].to_numpy()
        upper = drawn["upper_conf"].to_numpy()
        pvals = drawn[p_col].to_numpy()
        null_value = 0.5 if est_col == "relative_effect" else 0.0
        xlabel = xlab or est_col
        title = main or comparison_result.test_info[test]["label"]
    else:
        drawn = tbl
        labels = drawn["features"].tolist()
        estimate = -np.log10(np.clip(drawn[p_col].to_numpy(), 1e-300, 1))
        lower = upper = np.full_like(estimate, np.nan)
        pvals = drawn[p_col].to_numpy()
        null_value = -np.log10(alpha)
        xlabel = xlab or f"-log10({p_col})"
        title = main or comparison_result.test_info[test]["label"]

    if sort_by == "pvalue":
        order = np.argsort(pvals)
        labels = [labels[i] for i in order]
        estimate = estimate[order]
        lower = lower[order]
        upper = upper[order]
        pvals = pvals[order]

    y = np.arange(len(labels))[::-1] + 1
    colors = np.where(pvals <= alpha, col_signif, col_plain)

    if view == "pvalue":
        ax.barh(y, estimate, color=colors, height=0.6)
        ax.axvline(null_value, color=theme["guide"], ls="--")
    else:
        for yi, est, lo, hi, col in zip(y, estimate, lower, upper, colors):
            ax.plot([lo, hi], [yi, yi], color=col, lw=2)
            ax.plot(est, yi, "o", color=col)
        ax.axvline(null_value, color=theme["guide"], ls="--")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=theme["fg"])
    ax.set_xlabel(xlabel, color=theme["fg"])
    ax.set_title(title, color=theme["fg"])
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(colors=theme["fg"])

    out = drawn.copy() if isinstance(drawn, pd.DataFrame) else pd.DataFrame(drawn)
    out.attrs["view"] = view
    plt.tight_layout()
    return out
