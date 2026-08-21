"""Correlation between every pair of features, with all three coefficients.

Port of ``R/summarize_association_stats.R`` and the three helpers of
``R/utils_associate.R``. The tests themselves are in :mod:`._correlation`, which
is where R's ``cor.test`` branches live.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.padjust import p_adjust
from ..core.validate import check_p_adjust, validate_wide_input
from ._correlation import METHODS, cor_test_pvalue, kendall_tau, spearman_rho

__all__ = [
    "MISSING_POLICIES",
    "association_matrices",
    "pairwise_n",
    "summarize_association_stats",
]

#: How a missing value may be handled, in R's spelling. The first is the default.
MISSING_POLICIES: tuple[str, ...] = ("pairwise.complete.obs", "complete.obs")

#: Features a screen needs before there is a pair to report.
MIN_FEATS = 2


def pairwise_n(x: np.ndarray) -> np.ndarray:
    """How many observations each pair of features shares.

    Port of ``sa_pairwise_n()``. The cross product of the indicator of what is
    present counts, for every pair of columns, the rows where both are. The
    diagonal is then the count for one column on its own, which is what the pair
    of a feature with itself would have.

    Args:
        x: Features in columns, non-finite values already missing.

    Returns:
        A features-by-features integer array.
    """
    present = np.isfinite(x).astype(float)
    counts: np.ndarray = (present.T @ present).astype(np.int64)
    return counts


def association_matrices(
    x: np.ndarray,
    method: str,
    adj_type: str,
    feats: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """The four matrices one correlation method produces.

    Port of ``sa_association_matrices()``.

    The diagonal is set rather than estimated. A feature's correlation with
    itself is a property of the matrix and not a test, so ``corr`` carries 1
    there even for a feature with no variance to correlate, while ``pvalue`` and
    ``adj_pvalue`` carry a missing value. That convention is what lets a
    correlation plot mask a cell by its p-value without having to make an
    exception of the diagonal.

    The family the adjustment runs over is the pairs that produced a p-value, not
    every cell of the triangle. A pair the test refused is not a test that was
    performed, and counting it would shrink the others for a comparison that
    never happened.

    Args:
        x: Features in columns, non-finite values already missing and any
            listwise deletion already done.
        method: One of :data:`~statassist.summarize._correlation.METHODS`.
        adj_type: Multiplicity adjustment, already checked.
        feats: Names for the rows and columns. Numbered when omitted.

    Returns:
        The four features-by-features frames ``corr``, ``pvalue``,
        ``adj_pvalue`` and ``n``.
    """
    p = x.shape[1]
    names = feats if feats is not None else [str(index) for index in range(p)]

    correlation = np.full((p, p), np.nan)
    pvalue = np.full((p, p), np.nan)
    for j in range(p):
        correlation[j, j] = 1.0
    for j in range(p - 1):
        for k in range(j + 1, p):
            correlation[j, k] = correlation[k, j] = _pairwise_corr(x[:, j], x[:, k], method)
            pvalue[j, k] = pvalue[k, j] = cor_test_pvalue(x[:, j], x[:, k], method)

    adjusted = np.full((p, p), np.nan)
    # Column-major over the upper triangle, which is the order R extracts them
    # in. No adjustment this package offers depends on the order, and following
    # it costs nothing.
    tested = [
        (row, column)
        for column in range(p)
        for row in range(column)
        if math.isfinite(pvalue[row, column])
    ]
    if tested:
        family = p_adjust([pvalue[row, column] for row, column in tested], adj_type)
        for (row, column), value in zip(tested, family, strict=False):
            adjusted[row, column] = adjusted[column, row] = value

    return {
        "corr": _frame(correlation, names),
        "pvalue": _frame(pvalue, names),
        "adj_pvalue": _frame(adjusted, names),
        "n": _frame(pairwise_n(x), names),
    }


def summarize_association_stats(
    data: Any,
    feats: Any = None,
    methods: Any = METHODS,
    adj_type: str = "BH",
    use: str = MISSING_POLICIES[0],
) -> dict[str, Any]:
    """Correlation between every pair of features, with all three coefficients.

    Reduces a set of features to the association between each pair of them, as a
    square matrix per quantity: the coefficient, its p-value, the p-value
    adjusted across the pairs, and how many observations the pair shared.
    Pearson, Spearman and Kendall are reported side by side on the same pairs, so
    that a linear coefficient and a monotonic one disagreeing is visible rather
    than a matter of which call was made.

    This is a screen rather than a test of one hypothesis. It is the companion of
    :func:`~statassist.summarize_descriptive_stats`, which reduces one feature at
    a time to a row of its own; this reduces a pair at a time.

    Every matrix is symmetric: the upper triangle is computed and mirrored, so a
    pair is tested once rather than twice. The diagonal is a property of the
    matrix rather than an estimate - ``corr`` is 1 and the two p-values are
    missing - while ``n`` on the diagonal is how many observations that feature
    has.

    Missing and non-finite values are treated alike, an infinity being as much
    "no value to correlate" as a blank, and ``n`` counts what was left.

    The cost is one test per pair per method: thirty features are 435 pairs, and
    1305 tests with all three methods asked for. Kendall's is the slowest of the
    three, so naming ``methods`` is worth doing on a wide frame.

    Args:
        data: Wide frame (or 2-D array), one row per observation and one column
            per feature.
        feats: Names of the numeric columns to correlate, in output order, or
            ``None`` for every numeric column of ``data``. At least
            :data:`MIN_FEATS` are needed.
        methods: Which coefficients to compute, drawn from
            :data:`~statassist.summarize._correlation.METHODS`. Each one named
            gets a key of the result, in the order given; the ones not named have
            no key at all.
        adj_type: Multiplicity adjustment applied across the pairs.
        use: One of :data:`MISSING_POLICIES`. ``"pairwise.complete.obs"`` reads
            each pair on the observations that pair shares, and
            ``"complete.obs"`` drops any row with a missing value in any feature
            first, so that every pair is read on one set of rows.

    Returns:
        One key per entry of ``methods``, named after it and holding the four
        frames of :func:`association_matrices`. Beside those, ``design`` records
        ``feats``, ``n_obs``, ``methods``, ``adj_type`` and ``use``.

    Raises:
        SaValueError: If ``data`` is neither a frame nor a 2-D array, if
            ``methods`` holds something unknown or repeated, if fewer than
            :data:`MIN_FEATS` features are left, or if ``complete.obs`` leaves no
            row at all.
    """
    if use not in MISSING_POLICIES:
        raise SaValueError("`use` must be one of: " + ", ".join(MISSING_POLICIES) + ".")
    check_p_adjust(adj_type, "adj_type")
    wanted = _check_methods(methods)

    frame = _as_frame(data)
    if feats is None:
        feats = [
            str(name)
            for name in frame.columns
            if pd.api.types.is_numeric_dtype(frame[name]) and frame[name].dtype != bool
        ]
        if not feats:
            raise SaValueError("`data` holds no numeric column to correlate.")

    validated = validate_wide_input(frame, feats, group=None, group_lv=None)
    names = validated.feats
    if len(names) < MIN_FEATS:
        raise SaValueError(
            "`summarize_association_stats()` needs at least 2 features to correlate, "
            f"but got {len(names)}."
        )

    # An infinity is as much "no value to correlate" as a blank, and letting one
    # through would turn every coefficient that feature has into a nan.
    x = validated.data[names].to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan

    if use == "complete.obs":
        keep = np.isfinite(x).all(axis=1)
        if not keep.any():
            raise SaValueError(
                '`use = "complete.obs"` leaves no row: every row has a missing value '
                "in at least one of `feats`."
            )
        if not keep.all():
            notify(
                f"Dropped {int((~keep).sum())} row(s) with a missing value, as "
                '`use = "complete.obs"` asks.'
            )
        x = x[keep]

    flat = [name for name, column in zip(names, x.T, strict=False) if not _has_spread(column)]
    if flat:
        notify(
            f"{len(flat)} feature(s) have no variance to correlate and come back as "
            "NA: " + ", ".join(flat) + "."
        )

    out: dict[str, Any] = {
        method: association_matrices(x, method, adj_type, names) for method in wanted
    }
    out["design"] = {
        "feats": names,
        "n_obs": int(x.shape[0]),
        "methods": wanted,
        "adj_type": adj_type,
        "use": use,
    }
    return out


def _check_methods(methods: Any) -> list[str]:
    """Read ``methods`` the way R's three checks in a row read it."""
    if isinstance(methods, str):
        methods = [methods]
    try:
        items = [] if methods is None else list(methods)
    except TypeError:
        items = []
    if not items or any(not isinstance(item, str) for item in items):
        raise SaValueError(
            "`methods` must be a non-empty character vector drawn from: " + ", ".join(METHODS) + "."
        )
    unknown = [item for item in items if item not in METHODS]
    if unknown:
        raise SaValueError(
            "`methods` must be drawn from: "
            + ", ".join(METHODS)
            + ". Not recognised: "
            + ", ".join(unknown)
            + "."
        )
    repeated = [method for method in dict.fromkeys(items) if items.count(method) > 1]
    if repeated:
        raise SaValueError("`methods` contains duplicated names: " + ", ".join(repeated))
    return items


def _as_frame(data: Any) -> pd.DataFrame:
    """Read ``data`` as a wide frame.

    R drops a matrix's row names here before converting, because repeated ones
    would fail the conversion and nothing downstream reads them. A NumPy array
    has none to drop.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return pd.DataFrame(data)
    raise SaValueError("`data` must be a data.frame or a matrix.")


def _has_spread(column: np.ndarray) -> bool:
    """Whether a column has a finite, non-zero standard deviation."""
    finite = column[np.isfinite(column)]
    if finite.size < 2:
        return False
    spread = float(np.std(finite, ddof=1))
    return math.isfinite(spread) and spread != 0


def _pairwise_corr(u: np.ndarray, v: np.ndarray, method: str) -> float:
    """One cell of ``cor(x, use = "pairwise.complete.obs")``.

    Each pair is read on the observations it shares, so a Spearman coefficient is
    ranked within that subset rather than within the whole column.
    """
    complete = np.isfinite(u) & np.isfinite(v)
    x = u[complete]
    y = v[complete]
    if x.size < 2:
        return float("nan")
    if method == "spearman":
        return spearman_rho(x, y)
    if method == "kendall":
        return kendall_tau(x, y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _frame(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """A square matrix with the feature names on both axes."""
    return pd.DataFrame(values, index=names, columns=names)
