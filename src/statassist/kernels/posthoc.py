"""Pairwise post-hoc kernels (Tukey, Games-Howell, Dunn, Yuen, paired t, Conover)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from statassist.kernels.robust import sa_t_ci, sa_t_pval, sa_winsorize
from statassist.utils.validate import sa_level_pairs, sa_row


def sa_posthoc_columns() -> list[str]:
    return [
        "n1",
        "n2",
        "estimate",
        "stderr",
        "statistic",
        "df",
        "pval",
        "lower_conf",
        "upper_conf",
    ]


def sa_pair_matrix(
    group_lv: list[str],
    samples: dict[str, np.ndarray],
    fun: Callable[[int, int], pd.Series],
) -> np.ndarray:
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, row in pairs.iterrows():
        i, j = int(row["i"]), int(row["j"])
        res = fun(i, j)
        rows.append(res[sa_posthoc_columns()].to_numpy())
    return np.array(rows, dtype=float)


def sa_tukey(
    samples: dict[str, np.ndarray],
    conf_level: float = 0.95,
) -> np.ndarray:
    return _tukey_pairs(samples, conf_level)


def _tukey_pairs(samples: dict[str, np.ndarray], conf_level: float) -> np.ndarray:
    group_lv = list(samples.keys())
    k = len(group_lv)
    n = np.array([samples[lv].size for lv in group_lv])
    means = np.array([float(np.mean(samples[lv])) for lv in group_lv])
    df = float(n.sum() - k)

    ss_within = sum(
        float(np.sum((samples[lv] - means[i]) ** 2))
        for i, lv in enumerate(group_lv)
    )
    ms_within = ss_within / df
    if ms_within <= 0:
        raise ValueError(
            "the pooled mean square error is zero, so no pairwise comparison can be scaled."
        )
    q_crit = float(stats.studentized_range.ppf(conf_level, k, df))
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        estimate = means[i] - means[j]
        stderr = float(np.sqrt(ms_within / 2 * (1 / n[i] + 1 / n[j])))
        q_stat = estimate / stderr
        rows.append(
            sa_row(
                n1=n[i],
                n2=n[j],
                estimate=estimate,
                stderr=stderr,
                statistic=q_stat,
                df=df,
                pval=float(stats.studentized_range.sf(abs(q_stat), k, df)),
                lower_conf=estimate - q_crit * stderr,
                upper_conf=estimate + q_crit * stderr,
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)


def tukey_hsd(samples: dict[str, np.ndarray], conf_level: float = 0.95) -> np.ndarray:
    return _tukey_pairs(samples, conf_level)


def sa_games_howell(
    samples: dict[str, np.ndarray],
    conf_level: float = 0.95,
) -> np.ndarray:
    group_lv = list(samples.keys())
    k = len(group_lv)
    n = np.array([samples[lv].size for lv in group_lv])
    means = np.array([float(np.mean(samples[lv])) for lv in group_lv])
    vars_ = np.array([float(np.var(samples[lv], ddof=1)) for lv in group_lv])

    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        v_i = vars_[i] / n[i]
        v_j = vars_[j] / n[j]
        if v_i + v_j <= 0:
            raise ValueError(
                f"both groups of the pair {pr['group1']} - {pr['group2']} have zero variance."
            )
        estimate = means[i] - means[j]
        stderr = float(np.sqrt((v_i + v_j) / 2))
        df = float((v_i + v_j) ** 2 / (v_i**2 / (n[i] - 1) + v_j**2 / (n[j] - 1)))
        q_stat = estimate / stderr
        q_crit = float(stats.studentized_range.ppf(conf_level, k, df))
        rows.append(
            sa_row(
                n1=n[i],
                n2=n[j],
                estimate=estimate,
                stderr=stderr,
                statistic=q_stat,
                df=df,
                pval=float(stats.studentized_range.sf(abs(q_stat), k, df)),
                lower_conf=estimate - q_crit * stderr,
                upper_conf=estimate + q_crit * stderr,
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)


def sa_dunn(samples: dict[str, np.ndarray], conf_level: float = 0.95) -> np.ndarray:
    group_lv = list(samples.keys())
    n = np.array([samples[lv].size for lv in group_lv])
    total = int(n.sum())
    pooled = np.concatenate([samples[lv] for lv in group_lv])
    ranks = stats.rankdata(pooled)
    split_at = np.repeat(np.arange(len(group_lv)), n)
    mean_ranks = np.array(
        [float(np.mean(ranks[split_at == i])) for i in range(len(group_lv))]
    )

    tie_sizes = np.unique(pooled, return_counts=True)[1]
    tie_term = float(np.sum(tie_sizes**3 - tie_sizes) / (12 * (total - 1)))
    base_var = total * (total + 1) / 12 - tie_term
    if base_var <= 0:
        raise ValueError("every observation is tied, leaving the rank variance at zero.")

    z_crit = float(stats.norm.ppf(1 - (1 - conf_level) / 2))
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        estimate = mean_ranks[i] - mean_ranks[j]
        stderr = float(np.sqrt(base_var * (1 / n[i] + 1 / n[j])))
        z_stat = estimate / stderr
        rows.append(
            sa_row(
                n1=n[i],
                n2=n[j],
                estimate=estimate,
                stderr=stderr,
                statistic=z_stat,
                df=np.nan,
                pval=float(2 * stats.norm.sf(abs(z_stat))),
                lower_conf=estimate - z_crit * stderr,
                upper_conf=estimate + z_crit * stderr,
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)


def sa_yuen_independent(
    x: np.ndarray,
    y: np.ndarray,
    tr: float = 0.2,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x, n_y = x.size, y.size
    h_x = n_x - 2 * int(np.floor(tr * n_x))
    h_y = n_y - 2 * int(np.floor(tr * n_y))
    if h_x < 2 or h_y < 2:
        raise ValueError(
            f"fewer than 2 observations survive trimming {tr} from each tail "
            f"({h_x} and {h_y})."
        )

    d_x = (n_x - 1) * float(np.var(sa_winsorize(x, tr), ddof=1)) / (h_x * (h_x - 1))
    d_y = (n_y - 1) * float(np.var(sa_winsorize(y, tr), ddof=1)) / (h_y * (h_y - 1))
    stderr = float(np.sqrt(d_x + d_y))
    if not np.isfinite(stderr) or stderr <= 0:
        raise ValueError(
            "both winsorised samples are constant, leaving the standard error at "
            "zero and the statistic undefined."
        )

    df = (d_x + d_y) ** 2 / (d_x**2 / (h_x - 1) + d_y**2 / (h_y - 1))
    x_trim_mean = float(stats.trim_mean(x, tr))
    y_trim_mean = float(stats.trim_mean(y, tr))
    trim_diff = x_trim_mean - y_trim_mean
    yuen_stat = trim_diff / stderr
    ci = sa_t_ci(trim_diff, stderr, df, alternative, conf_level)

    return {
        "x_trim_mean": x_trim_mean,
        "y_trim_mean": y_trim_mean,
        "trim_diff": float(trim_diff),
        "stderr": stderr,
        "yuen_stat": float(yuen_stat),
        "df": float(df),
        "pval": sa_t_pval(yuen_stat, df, alternative),
        "lower_conf": float(ci[0]),
        "upper_conf": float(ci[1]),
    }


def sa_pairwise_yuen(
    samples: dict[str, np.ndarray],
    tr: float = 0.2,
    conf_level: float = 0.95,
) -> np.ndarray:
    group_lv = list(samples.keys())
    n = np.array([samples[lv].size for lv in group_lv])
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        res = sa_yuen_independent(samples[group_lv[i]], samples[group_lv[j]], tr=tr, conf_level=conf_level)
        rows.append(
            sa_row(
                n1=n[i],
                n2=n[j],
                estimate=res["trim_diff"],
                stderr=res["stderr"],
                statistic=res["yuen_stat"],
                df=res["df"],
                pval=res["pval"],
                lower_conf=res["lower_conf"],
                upper_conf=res["upper_conf"],
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)


def sa_pairwise_paired_t(mat: np.ndarray, conf_level: float = 0.95) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    group_lv = list(mat.columns) if hasattr(mat, "columns") else [str(i) for i in range(mat.shape[1])]
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        col_i = mat[:, i] if isinstance(mat, np.ndarray) else mat.iloc[:, i].to_numpy()
        col_j = mat[:, j] if isinstance(mat, np.ndarray) else mat.iloc[:, j].to_numpy()
        res = stats.ttest_rel(col_i, col_j)
        se = float(np.std(col_i - col_j, ddof=1) / np.sqrt(n))
        alpha = 1 - conf_level
        df = n - 1
        diff = float(np.mean(col_i) - np.mean(col_j))
        q = stats.t.ppf(1 - alpha / 2, df)
        rows.append(
            sa_row(
                n1=n,
                n2=n,
                estimate=diff,
                stderr=se,
                statistic=float(res.statistic),
                df=float(df),
                pval=float(res.pvalue),
                lower_conf=diff - q * se,
                upper_conf=diff + q * se,
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)


def sa_conover(mat: np.ndarray | pd.DataFrame, conf_level: float = 0.95) -> np.ndarray:
    if isinstance(mat, pd.DataFrame):
        group_lv = list(mat.columns)
        arr = mat.to_numpy(dtype=float)
    else:
        arr = np.asarray(mat, dtype=float)
        group_lv = [str(i) for i in range(arr.shape[1])]
    n, k = arr.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, arr)
    rank_sums = np.sum(ranks, axis=0)
    a = float(np.sum(ranks**2))
    b = float(np.sum(rank_sums**2) / n)
    df = (n - 1) * (k - 1)
    variance = 2 * n * (a - b) / df
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(
            "every subject ranks the conditions identically, leaving the "
            "residual rank variance at zero."
        )
    stderr = float(np.sqrt(variance))
    t_crit = float(stats.t.ppf(1 - (1 - conf_level) / 2, df))
    pairs = sa_level_pairs(group_lv)
    rows = []
    for _, pr in pairs.iterrows():
        i = group_lv.index(pr["group1"])
        j = group_lv.index(pr["group2"])
        estimate = rank_sums[i] - rank_sums[j]
        t_stat = estimate / stderr
        rows.append(
            sa_row(
                n1=n,
                n2=n,
                estimate=estimate,
                stderr=stderr,
                statistic=t_stat,
                df=float(df),
                pval=float(2 * stats.t.sf(abs(t_stat), df)),
                lower_conf=estimate - t_crit * stderr,
                upper_conf=estimate + t_crit * stderr,
            )[sa_posthoc_columns()].to_numpy()
        )
    return np.array(rows, dtype=float)
