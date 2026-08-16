"""Correlation matrices with p-values and multiplicity adjustment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.utils.validate import p_adjust, sa_check_p_adjust


def sa_cor_test_pvalue(
    u: np.ndarray,
    v: np.ndarray,
    method: str,
) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(u) & np.isfinite(v)
    u, v = u[mask], v[mask]
    if u.size < 3:
        return np.nan
    if np.std(u, ddof=1) == 0 or np.std(v, ddof=1) == 0:
        return np.nan
    try:
        if method == "pearson":
            _, p = stats.pearsonr(u, v)
        elif method == "spearman":
            _, p = stats.spearmanr(u, v)
        elif method == "kendall":
            _, p = stats.kendalltau(u, v)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    except Exception:
        return np.nan
    return float(p) if np.isfinite(p) else np.nan


def sa_pairwise_n(x: np.ndarray) -> np.ndarray:
    present = np.isfinite(x)
    n = present.astype(float).T @ present.astype(float)
    return n.astype(int)


def sa_association_matrices(
    x: np.ndarray,
    method: str,
    adj_type: str,
    feats: list[str] | None = None,
) -> dict[str, np.ndarray]:
    sa_check_p_adjust(adj_type, "adj_type")
    arr = np.asarray(x, dtype=float)
    p = arr.shape[1]
    if feats is None:
        feats = [f"V{i + 1}" for i in range(p)]
    elif len(feats) != p:
        raise ValueError(
            f"`feats` length {len(feats)} does not match matrix columns {p}."
        )

    empty = np.full((p, p), np.nan)
    df = pd.DataFrame(arr, columns=feats)
    corr = df.corr(method=method, min_periods=1).to_numpy(dtype=float).copy()
    np.fill_diagonal(corr, 1.0)

    pvalue = empty.copy()
    for j in range(p - 1):
        for k in range(j + 1, p):
            pv = sa_cor_test_pvalue(arr[:, j], arr[:, k], method)
            pvalue[j, k] = pv
            pvalue[k, j] = pv

    upper_idx = np.triu_indices(p, k=1)
    pair_pvals = pvalue[upper_idx]
    valid = np.isfinite(pair_pvals)
    adj_pairs = np.full(pair_pvals.shape, np.nan)
    if valid.any():
        adj_pairs[valid] = p_adjust(pair_pvals[valid], adj_type)
    adj_pvalue = empty.copy()
    adj_pvalue[upper_idx] = adj_pairs
    adj_pvalue[upper_idx[1], upper_idx[0]] = adj_pairs
    np.fill_diagonal(adj_pvalue, np.nan)

    n = sa_pairwise_n(arr)
    idx = pd.Index(feats, name=None)
    return {
        "corr": pd.DataFrame(corr, index=idx, columns=idx),
        "pvalue": pd.DataFrame(pvalue, index=idx, columns=idx),
        "adj_pvalue": pd.DataFrame(adj_pvalue, index=idx, columns=idx),
        "n": pd.DataFrame(n, index=idx, columns=idx),
    }
