"""Robust test kernels (Brunner-Munzel, Yuen paired)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def sa_t_pval(stat: float, df: float, alternative: str) -> float:
    if alternative == "two.sided":
        return float(2 * stats.t.cdf(-abs(stat), df=df))
    if alternative == "greater":
        return float(stats.t.sf(stat, df=df))
    if alternative == "less":
        return float(stats.t.cdf(stat, df=df))
    raise ValueError(f"unknown alternative: {alternative!r}")


def sa_t_ci(
    est: float,
    se: float,
    df: float,
    alternative: str,
    conf_level: float,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    if alternative == "two.sided":
        q = stats.t.ppf(1 - alpha / 2, df=df)
        return (est - q * se, est + q * se)
    if alternative == "greater":
        q = stats.t.ppf(1 - alpha, df=df)
        return (est - q * se, bounds[1])
    if alternative == "less":
        q = stats.t.ppf(1 - alpha, df=df)
        return (bounds[0], est + q * se)
    raise ValueError(f"unknown alternative: {alternative!r}")


def sa_winsorize(v: np.ndarray, tr: float) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    n = v.size
    g = int(np.floor(tr * n))
    if g <= 0:
        return v
    sorted_v = np.sort(v)
    lower = sorted_v[g]
    upper = sorted_v[n - g - 1]
    v[v < lower] = lower
    v[v > upper] = upper
    return v


def sa_winsorized_normal_var(tr: float) -> float:
    if tr <= 0:
        return 1.0
    z = stats.norm.ppf(tr)
    return (1 - 2 * tr) + 2 * z * stats.norm.pdf(z) + 2 * z**2 * tr


def sa_brunner_munzel(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Brunner-Munzel test for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = x.size
    n_y = y.size

    r_x = stats.rankdata(x)
    r_y = stats.rankdata(y)
    pooled = stats.rankdata(np.concatenate([x, y]))
    r_pooled_x = pooled[:n_x]
    r_pooled_y = pooled[n_x:]

    m_x = float(np.mean(r_pooled_x))
    m_y = float(np.mean(r_pooled_y))

    v_x = float(
        np.sum((r_pooled_x - r_x - m_x + (n_x + 1) / 2) ** 2) / (n_x - 1)
    )
    v_y = float(
        np.sum((r_pooled_y - r_y - m_y + (n_y + 1) / 2) ** 2) / (n_y - 1)
    )

    pooled_var = n_x * v_x + n_y * v_y
    if pooled_var <= 0:
        raise ValueError(
            "the groups do not overlap, leaving the Brunner-Munzel variance "
            "estimate at zero and the statistic undefined."
        )

    bm_stat = n_x * n_y * (m_x - m_y) / (n_x + n_y) / np.sqrt(pooled_var)
    df = pooled_var**2 / (
        (n_x * v_x) ** 2 / (n_x - 1) + (n_y * v_y) ** 2 / (n_y - 1)
    )

    relative_effect = 1 - (m_y - (n_y + 1) / 2) / n_x
    se = np.sqrt(v_x / (n_x * n_y**2) + v_y / (n_y * n_x**2))
    ci = sa_t_ci(relative_effect, se, df, alternative, conf_level, bounds=(0, 1))

    return {
        "relative_effect": float(relative_effect),
        "bm_stat": float(bm_stat),
        "df": float(df),
        "pval": sa_t_pval(bm_stat, df, alternative),
        "lower_conf": float(ci[0]),
        "upper_conf": float(ci[1]),
    }


def brunner_munzel_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Public wrapper around the manual Brunner-Munzel implementation."""
    return sa_brunner_munzel(x, y, alternative=alternative, conf_level=conf_level)


def sa_yuen_paired(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    tr: float = 0.2,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Yuen's trimmed mean test for two dependent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pairs = x.size
    h = n_pairs - 2 * int(np.floor(tr * n_pairs))

    win_x = sa_winsorize(x, tr)
    win_y = sa_winsorize(y, tr)

    ss_x = (n_pairs - 1) * float(np.var(win_x, ddof=1))
    ss_y = (n_pairs - 1) * float(np.var(win_y, ddof=1))
    ss_xy = (n_pairs - 1) * float(np.cov(win_x, win_y, ddof=1)[0, 1])

    stderr = np.sqrt((ss_x + ss_y - 2 * ss_xy) / (h * (h - 1)))
    if not np.isfinite(stderr) or stderr <= 0:
        raise ValueError(
            "the winsorised paired differences have zero variance, leaving the "
            "standard error at zero and the statistic undefined."
        )

    df = h - 1
    x_trim_mean = float(stats.trim_mean(x, tr))
    y_trim_mean = float(stats.trim_mean(y, tr))
    trim_diff = x_trim_mean - y_trim_mean
    yuen_stat = trim_diff / stderr
    ci = sa_t_ci(trim_diff, stderr, df, alternative, conf_level)

    win_diff_var = float(np.var(sa_winsorize(x - y, tr), ddof=1))
    if win_diff_var > 0:
        robust_dz = trim_diff / np.sqrt(
            win_diff_var / sa_winsorized_normal_var(tr)
        )
    else:
        robust_dz = np.nan

    return {
        "x_trim_mean": x_trim_mean,
        "y_trim_mean": y_trim_mean,
        "trim_diff": float(trim_diff),
        "stderr": float(stderr),
        "yuen_stat": float(yuen_stat),
        "df": float(df),
        "pval": sa_t_pval(yuen_stat, df, alternative),
        "lower_conf": float(ci[0]),
        "upper_conf": float(ci[1]),
        "robust_dz": float(robust_dz),
    }
