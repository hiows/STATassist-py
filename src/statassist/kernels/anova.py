"""Omnibus ANOVA kernels (one-way, Welch, Yuen, Kruskal-Wallis, RM-ANOVA, Friedman)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from statassist.kernels.robust import sa_winsorize


def sa_split_groups(
    values: np.ndarray | Sequence[float],
    group: Sequence[Any],
    n_min: int = 2,
) -> dict[str, np.ndarray]:
    """Split a numeric vector into one sample per group level."""
    values = np.asarray(values, dtype=float)
    group = np.asarray(group)
    levels = list(dict.fromkeys(group))
    samples: dict[str, np.ndarray] = {}
    for lv in levels:
        v = values[group == lv]
        v = v[np.isfinite(v)]
        samples[str(lv)] = v
    sizes = {k: v.size for k, v in samples.items()}
    short = [k for k, n in sizes.items() if n < n_min]
    if short:
        detail = ", ".join(f"{k} = {sizes[k]}" for k in short)
        raise ValueError(
            f"needs at least {n_min} usable observation(s) per group; {detail}."
        )
    return samples


def sa_oneway_anova(samples: dict[str, np.ndarray] | list[np.ndarray]) -> dict[str, float]:
    if isinstance(samples, dict):
        samples = list(samples.values())
    k = len(samples)
    n = np.array([s.size for s in samples])
    total = int(n.sum())
    df1 = k - 1
    df2 = total - k
    if df2 < 1:
        raise ValueError(
            "needs more observations than groups to leave any residual degrees of freedom."
        )

    means = np.array([float(np.mean(s)) for s in samples])
    grand = float(np.sum(n * means) / total)

    ss_between = float(np.sum(n * (means - grand) ** 2))
    ss_within = float(
        sum(float(np.sum((s - m) ** 2)) for s, m in zip(samples, means))
    )
    ss_total = ss_between + ss_within

    if ss_within <= 0:
        raise ValueError(
            "every group has zero variance, leaving the mean square error at "
            "zero and the F statistic undefined."
        )

    ms_within = ss_within / df2
    f_stat = (ss_between / df1) / ms_within

    return {
        "n_used": float(total),
        "n_groups": float(k),
        "f_stat": float(f_stat),
        "df1": float(df1),
        "df2": float(df2),
        "eta_sq": ss_between / ss_total,
        "omega_sq": (ss_between - df1 * ms_within) / (ss_total + ms_within),
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }


def sa_welch_anova(samples: dict[str, np.ndarray] | list[np.ndarray]) -> dict[str, float]:
    if isinstance(samples, dict):
        names = list(samples.keys())
        samples = list(samples.values())
    else:
        names = None
    k = len(samples)
    n = np.array([s.size for s in samples])
    if np.any(n < 2):
        raise ValueError(
            "Welch's ANOVA needs at least 2 observations per group to estimate "
            "a within-group variance."
        )

    means = np.array([float(np.mean(s)) for s in samples])
    vars_ = np.array([float(np.var(s, ddof=1)) for s in samples])
    if np.any(vars_ <= 0):
        zero = [names[i] if names else str(i) for i, v in enumerate(vars_) if v <= 0]
        raise ValueError(
            f"group(s) with zero variance leave the Welch weight infinite: "
            f"{', '.join(zero)}."
        )

    w = n / vars_
    sum_w = float(np.sum(w))
    weighted_mean = float(np.sum(w * means) / sum_w)

    lambda_ = float(np.sum((1 - w / sum_w) ** 2 / (n - 1)))
    numerator = float(np.sum(w * (means - weighted_mean) ** 2) / (k - 1))
    denominator = 1 + 2 * (k - 2) / (k**2 - 1) * lambda_
    f_stat = numerator / denominator

    df1 = float(k - 1)
    df2 = float(1 / (3 / (k**2 - 1) * lambda_))

    ref = sa_oneway_anova(samples)

    return {
        "n_used": float(n.sum()),
        "n_groups": float(k),
        "f_stat": float(f_stat),
        "df1": df1,
        "df2": df2,
        "eta_sq": ref["eta_sq"],
        "omega_sq": ref["omega_sq"],
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }


def sa_yuen_anova(
    samples: dict[str, np.ndarray] | list[np.ndarray],
    tr: float = 0.2,
) -> dict[str, float]:
    if isinstance(samples, dict):
        names = list(samples.keys())
        samples = list(samples.values())
    else:
        names = None
    k = len(samples)
    n = np.array([s.size for s in samples])

    h = n - 2 * np.floor(tr * n).astype(int)
    if np.any(h < 2):
        short = [names[i] if names else str(i) for i in range(k) if h[i] < 2]
        raise ValueError(
            f"fewer than 2 observations survive trimming {tr} from each tail in "
            f"group(s): {', '.join(short)}."
        )

    trim_means = np.array([float(stats.trim_mean(s, tr)) for s in samples])
    win_vars = np.array(
        [float(np.var(sa_winsorize(s, tr), ddof=1)) for s in samples]
    )
    if np.any(win_vars <= 0):
        flat = [names[i] if names else str(i) for i, v in enumerate(win_vars) if v <= 0]
        raise ValueError(
            f"group(s) whose winsorised values are constant leave the trimmed "
            f"weight infinite: {', '.join(flat)}."
        )

    d = (n - 1) * win_vars / (h * (h - 1))
    w = 1 / d
    sum_w = float(np.sum(w))
    weighted_mean = float(np.sum(w * trim_means) / sum_w)

    lambda_ = float(np.sum((1 - w / sum_w) ** 2 / (h - 1)))
    numerator = float(np.sum(w * (trim_means - weighted_mean) ** 2) / (k - 1))
    f_stat = numerator / (1 + 2 * (k - 2) / (k**2 - 1) * lambda_)

    df1 = float(k - 1)
    df2 = float(1 / (3 / (k**2 - 1) * lambda_))

    between = float(np.sum((trim_means - np.mean(trim_means)) ** 2) / k)
    within = float(np.mean(win_vars / (1 - 2 * tr) ** 2))

    return {
        "n_used": float(n.sum()),
        "n_groups": float(k),
        "f_stat": float(f_stat),
        "df1": df1,
        "df2": df2,
        "robust_eta_sq": between / (between + within) if (between + within) > 0 else np.nan,
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }


def sa_kruskal(samples: dict[str, np.ndarray] | list[np.ndarray]) -> dict[str, float]:
    if isinstance(samples, dict):
        samples = list(samples.values())
    k = len(samples)
    total = sum(s.size for s in samples)
    pooled = np.concatenate(samples)
    if np.unique(pooled).size < 2:
        raise ValueError(
            "every observation takes the same value, so the ranks carry no "
            "information and the tie correction is undefined."
        )
    res = stats.kruskal(*samples)
    h = float(res.statistic)
    df = float(len(samples) - 1)

    return {
        "n_used": float(total),
        "n_groups": float(k),
        "h_stat": h,
        "df": df,
        "epsilon_sq": h * (total + 1) / (total**2 - 1),
        "eta_sq_rank": (h - k + 1) / (total - k),
        "pval": float(res.pvalue),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }


def sa_sphericity(mat: np.ndarray) -> dict[str, float]:
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    p = k - 1
    f = n - 1
    lower_bound = 1 / p
    na_out = {
        "mauchly_w": np.nan,
        "mauchly_pval": np.nan,
        "gg_eps": lower_bound,
        "hf_eps": lower_bound,
    }
    if n <= k:
        return na_out

    cov = np.cov(mat, rowvar=False)
    dev = np.eye(k) - np.ones((k, k)) / k
    q, _ = np.linalg.qr(dev)
    contrasts = q[:, :p]
    transformed = contrasts.T @ cov @ contrasts
    eigenvalues = np.linalg.eigvalsh(transformed)
    if np.any(eigenvalues <= 0):
        return na_out

    w = float(np.prod(eigenvalues) / (np.sum(eigenvalues) / p) ** p)
    rho = 1 - (2 * p**2 + p + 2) / (6 * p * f)
    chi_sq = -f * rho * np.log(w)
    chi_df = p * (p + 1) / 2 - 1
    weight = (
        (p + 2)
        * (p - 1)
        * (p - 2)
        * (2 * p**3 + 6 * p**2 + 3 * p + 2)
        / (288 * (f * p * rho) ** 2)
    )
    lead = float(stats.chi2.sf(chi_sq, chi_df))
    correction = float(stats.chi2.sf(chi_sq, chi_df + 4))
    mauchly_pval = lead + weight * (correction - lead)

    gg = float(np.sum(eigenvalues) ** 2 / (p * np.sum(eigenvalues**2)))
    gg = min(max(gg, lower_bound), 1)
    hf = (n * p * gg - 2) / (p * (f - p * gg))
    hf = min(max(hf, lower_bound), 1)

    return {
        "mauchly_w": w,
        "mauchly_pval": mauchly_pval,
        "gg_eps": gg,
        "hf_eps": hf,
    }


def sa_rm_anova(mat: np.ndarray) -> dict[str, float]:
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    if n < 2:
        raise ValueError(f"needs at least 2 complete subjects, got {n}.")

    grand = float(np.mean(mat))
    condition_means = np.mean(mat, axis=0)
    subject_means = np.mean(mat, axis=1)

    ss_condition = float(n * np.sum((condition_means - grand) ** 2))
    ss_subject = float(k * np.sum((subject_means - grand) ** 2))
    ss_total = float(np.sum((mat - grand) ** 2))
    ss_error = ss_total - ss_condition - ss_subject

    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    if ss_error <= 0:
        raise ValueError(
            "the subject-by-condition residuals are all zero, leaving the F "
            "statistic undefined."
        )
    ms_error = ss_error / df2
    f_stat = (ss_condition / df1) / ms_error

    sph = sa_sphericity(mat)
    gg = sph["gg_eps"]
    hf = sph["hf_eps"]

    return {
        "n_used": float(n),
        "n_groups": float(k),
        "f_stat": float(f_stat),
        "df1": float(df1),
        "df2": float(df2),
        "partial_eta_sq": ss_condition / (ss_condition + ss_error),
        "gen_eta_sq": ss_condition / (ss_condition + ss_subject + ss_error),
        "mauchly_w": sph["mauchly_w"],
        "mauchly_pval": sph["mauchly_pval"],
        "gg_eps": gg,
        "pval_gg": float(stats.f.sf(f_stat, df1 * gg, df2 * gg)),
        "hf_eps": hf,
        "pval_hf": float(stats.f.sf(f_stat, df1 * hf, df2 * hf)),
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }


def sa_friedman(mat: np.ndarray) -> dict[str, float]:
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    if np.all(np.apply_along_axis(lambda row: np.unique(row).size < 2, 1, mat)):
        raise ValueError(
            "no subject distinguishes the conditions, so the within-subject "
            "ranks carry no information."
        )
    res = stats.friedmanchisquare(*mat.T)
    chi_sq = float(res.statistic)
    df = float(mat.shape[1] - 1)

    return {
        "n_used": float(n),
        "n_groups": float(k),
        "chi_sq": chi_sq,
        "df": df,
        "kendalls_w": chi_sq / (n * (k - 1)),
        "pval": float(res.pvalue),
        "lower_conf": np.nan,
        "upper_conf": np.nan,
    }
