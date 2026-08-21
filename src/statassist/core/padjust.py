"""Multiplicity adjustment, transcribed from R's ``stats::p.adjust``.

This is not a convenience wrapper. Every test table in the package carries a
``pval_adj`` column produced here, and the tables routinely hold a missing
p-value for a feature the test could not be run on, so two details of R's
implementation decide the numbers:

1. ``n`` defaults to the number of **non-missing** p-values, not to the length of
   the vector. In R the default ``n = length(p)`` is a promise that is forced
   only after ``p <- p[!is.na(p)]`` has run, so a table with ten features of
   which two failed is adjusted against eight, not ten.
2. The missing entries come back missing, in place. The adjustment is computed
   on the present values and written back into their own positions.

``statsmodels.stats.multitest.multipletests`` does neither, which is why this
exists rather than a call to it. Everything here is deterministic, so it matches
R exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import SaInternalError, SaValueError

__all__ = ["P_ADJUST_METHODS", "p_adjust"]

#: The contents of R's ``stats::p.adjust.methods``, in its order. ``"fdr"`` is an
#: alias of ``"BH"`` kept for backwards compatibility, as in R.
P_ADJUST_METHODS: tuple[str, ...] = (
    "holm",
    "hochberg",
    "hommel",
    "bonferroni",
    "BH",
    "BY",
    "fdr",
    "none",
)


def _inverse_order(order: np.ndarray) -> np.ndarray:
    """R's ``order(o)`` for a permutation ``o``: original position -> rank."""
    inverse = np.empty(order.size, dtype=np.intp)
    inverse[order] = np.arange(order.size, dtype=np.intp)
    return inverse


def _ascending(values: np.ndarray) -> np.ndarray:
    """R's ``order(p)``: ascending, ties left in their original order."""
    return np.argsort(values, kind="stable")


def _descending(values: np.ndarray) -> np.ndarray:
    """R's ``order(p, decreasing = TRUE)``: ties left in their original order."""
    return np.argsort(-values, kind="stable")


def _hommel(present: np.ndarray, n: int, lp: int) -> np.ndarray:
    """Hommel's adjustment, transcribed from R statement by statement.

    Requires ``n >= 3``; the caller sends ``n == 2`` to Hochberg and returns
    early for ``n <= 1``, exactly as R does.
    """
    padded = present if n == lp else np.concatenate([present, np.ones(n - lp)])
    order = _ascending(padded)
    sorted_p = padded[order]
    ranks = np.arange(1, n + 1, dtype=float)

    start = float(np.min(n * sorted_p / ranks))
    q = np.full(n, start)
    pa = np.full(n, start)

    for m in range(n - 1, 1, -1):
        head = slice(0, n - m + 1)  # R seq_len(n - m + 1)
        tail = slice(n - m + 1, n)  # R (n - m + 2):n
        q1 = float(np.min(m * sorted_p[tail] / np.arange(2, m + 1, dtype=float)))
        q[head] = np.minimum(m * sorted_p[head], q1)
        q[tail] = q[n - m]  # R q[n - m + 1L]
        pa = np.maximum(pa, q)

    adjusted = np.maximum(pa, sorted_p)
    back = _inverse_order(order)
    return np.asarray(adjusted[back[:lp] if lp < n else back], dtype=float)


def p_adjust(p: Any, method: str = "none", n: int | None = None) -> np.ndarray:
    """Adjust a vector of p-values for multiple comparisons.

    Args:
        p: The p-values. Missing entries are carried through untouched.
        method: One of :data:`P_ADJUST_METHODS`.
        n: Number of comparisons the family holds. Defaults to the number of
            non-missing entries in ``p``, which is what R's own default works
            out to, and may not be smaller than that.

    Returns:
        A new ``float64`` array the same length as ``p``.
    """
    if method not in P_ADJUST_METHODS:
        raise SaValueError("`method` must be one of: " + ", ".join(P_ADJUST_METHODS) + ".")
    if method == "fdr":
        method = "BH"

    out = np.array(np.asarray(p, dtype=float), dtype=float, copy=True).reshape(-1)
    present_mask = ~np.isnan(out)
    present = out[present_mask]
    lp = int(present.size)

    if n is None:
        n = lp
    if n < lp:
        raise SaInternalError(
            f"internal error: `n` is {n}, fewer than the {lp} p-value(s) being adjusted."
        )

    if n <= 1 or method == "none":
        return out

    if method == "bonferroni":
        adjusted = np.minimum(1.0, n * present)

    elif method == "holm":
        order = _ascending(present)
        ranks = np.arange(1, lp + 1, dtype=float)
        running = np.maximum.accumulate((n + 1 - ranks) * present[order])
        adjusted = np.minimum(1.0, running)[_inverse_order(order)]

    elif method == "hochberg":
        order = _descending(present)
        ranks = np.arange(lp, 0, -1, dtype=float)
        running = np.minimum.accumulate((n + 1 - ranks) * present[order])
        adjusted = np.minimum(1.0, running)[_inverse_order(order)]

    elif method == "BH":
        order = _descending(present)
        ranks = np.arange(lp, 0, -1, dtype=float)
        running = np.minimum.accumulate(n / ranks * present[order])
        adjusted = np.minimum(1.0, running)[_inverse_order(order)]

    elif method == "BY":
        order = _descending(present)
        ranks = np.arange(lp, 0, -1, dtype=float)
        harmonic = float(np.sum(1.0 / np.arange(1, n + 1, dtype=float)))
        running = np.minimum.accumulate(harmonic * n / ranks * present[order])
        adjusted = np.minimum(1.0, running)[_inverse_order(order)]

    else:  # hommel
        if n == 2:
            return p_adjust(out, "hochberg", n)
        adjusted = _hommel(present, n, lp)

    out[present_mask] = adjusted
    return out
