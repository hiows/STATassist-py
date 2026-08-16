"""Internal summary kernels shared by descriptive functions."""

from __future__ import annotations

import numpy as np

from statassist.utils.validate import sa_na_row

DESCRIBE_COLUMNS = (
    "n",
    "n_missing",
    "mean",
    "sd",
    "var",
    "se",
    "cv",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "iqr",
    "out_lower_bound",
    "out_upper_bound",
    "mad",
    "skewness",
    "excess_kurtosis",
)


def sa_skewness(v: np.ndarray) -> float:
    """Sample skewness G1 (bias-corrected, e1071 type 2)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    m = np.mean(v)
    m2 = np.sum((v - m) ** 2) / n
    if n < 3 or m2 <= 0:
        return np.nan
    g1 = (np.sum((v - m) ** 3) / n) / (m2**1.5)
    return float(g1 * np.sqrt(n * (n - 1)) / (n - 2))


def sa_kurtosis(v: np.ndarray) -> float:
    """Sample excess kurtosis G2 (bias-corrected, e1071 type 2)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    m = np.mean(v)
    m2 = np.sum((v - m) ** 2) / n
    if n < 4 or m2 <= 0:
        return np.nan
    g2 = (np.sum((v - m) ** 4) / n) / (m2**2) - 3
    return float(((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)))


def sa_describe_vector(x: np.ndarray | list[float]) -> dict[str, float]:
    """Describe one numeric vector."""
    x = np.asarray(x, dtype=float)
    v = x[np.isfinite(x)]
    n = v.size

    if n == 0:
        out = sa_na_row(DESCRIBE_COLUMNS).to_dict()
        out["n"] = 0.0
        out["n_missing"] = float(x.size)
        return out

    m = float(np.mean(v))
    s = float(np.std(v, ddof=1))
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    mad = float(np.median(np.abs(v - np.median(v))) * 1.4826)

    return {
        "n": float(n),
        "n_missing": float(x.size - n),
        "mean": m,
        "sd": s,
        "var": float(np.var(v, ddof=1)),
        "se": s / np.sqrt(n),
        "cv": s / m if m != 0 else np.nan,
        "min": float(np.min(v)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(np.max(v)),
        "iqr": float(iqr),
        "out_lower_bound": float(q1 - 1.5 * iqr),
        "out_upper_bound": float(q3 + 1.5 * iqr),
        "mad": mad,
        "skewness": sa_skewness(v),
        "excess_kurtosis": sa_kurtosis(v),
    }
