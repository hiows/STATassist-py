"""Assumption-check kernels (Shapiro, KS, Levene, Bartlett)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from statassist.kernels.anova import sa_oneway_anova


def sa_shapiro(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    n = v.size
    if n < 3 or n > 5000:
        raise ValueError(f"Shapiro-Wilk needs between 3 and 5000 observations, got {n}.")
    res = stats.shapiro(v)
    return {"shapiro_stat": float(res.statistic), "shapiro_pval": float(res.pvalue)}


def sa_ks_normal(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    if v.size < 2:
        raise ValueError(
            f"the Kolmogorov-Smirnov test needs at least 2 observations, got {v.size}."
        )
    spread = float(np.std(v, ddof=1))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError(
            "the sample is constant, so no normal reference distribution can be fitted."
        )
    res = stats.kstest(v, stats.norm.cdf, args=(float(np.mean(v)), spread))
    return {"ks_stat": float(res.statistic), "ks_pval": float(res.pvalue)}


def sa_levene(
    samples: dict[str, np.ndarray] | list[np.ndarray],
    center: str = "median",
    trim: float = 0.1,
) -> dict[str, float]:
    if isinstance(samples, dict):
        samples = list(samples.values())

    def _centre(v: np.ndarray) -> float:
        if center == "median":
            return float(np.median(v))
        if center == "mean":
            return float(np.mean(v))
        if center == "trimmed":
            return float(stats.trim_mean(v, trim))
        raise ValueError("`center` must be one of: median, mean, trimmed.")

    deviations = [np.abs(v - _centre(v)) for v in samples]
    res = sa_oneway_anova(deviations)
    return {
        "levene_stat": res["f_stat"],
        "levene_df1": res["df1"],
        "levene_df2": res["df2"],
        "levene_pval": res["pval"],
    }


def sa_bartlett(samples: dict[str, np.ndarray] | list[np.ndarray]) -> dict[str, float]:
    if isinstance(samples, dict):
        samples = list(samples.values())
    res = stats.bartlett(*samples)
    return {
        "bartlett_stat": float(res.statistic),
        "bartlett_df": float(res.df),
        "bartlett_pval": float(res.pvalue),
    }
