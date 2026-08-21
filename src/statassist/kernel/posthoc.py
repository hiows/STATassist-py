"""Pairwise kernels, each paired with the omnibus test that shares its
assumptions.

Port of ``R/kernel_posthoc.R``. A rank-based omnibus test is never followed by a
parametric comparison, which is the whole reason these are not interchangeable.

Two contracts run through the file.

Every function returns a frame with one row per pair, in the row order of
:func:`~statassist.core.level_pairs`, and the nine columns of
:func:`posthoc_columns`. The estimate reads as ``group_lv[j] - group_lv[i]`` with
``i < j``: the reference is the first level, so it is the one being subtracted,
and a level that raised a feature is positive against the control in both the
post-hoc table and the fold change.

Only Tukey's and Games-Howell's p-values are family-wise. Both judge every
contrast against the studentised range, which controls the error rate over the
whole set at once, so adjusting them again would be adjusting twice. Dunn,
Conover, pairwise Yuen and pairwise paired t all return unadjusted p-values and
expect the caller to adjust across the pairs of one feature - which is what
:func:`~statassist.core.posthoc_table` does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaValueError
from ..core.tables import level_pairs, stat_row
from ._shared import as_matrix, as_sample, as_samples, condition_names
from .robust import t_ci, t_pval, trimmed_mean, winsorize

__all__ = [
    "conover",
    "dunn",
    "games_howell",
    "pair_matrix",
    "pairwise_paired_t",
    "pairwise_yuen",
    "posthoc_columns",
    "tukey",
    "yuen_independent",
]


def posthoc_columns() -> list[str]:
    """Column layout of one post-hoc pair.

    Port of ``sa_posthoc_columns()``.

    >>> posthoc_columns()[:3]
    ['n1', 'n2', 'estimate']
    """
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


def pair_matrix(
    group_lv: Any,
    fun: Callable[[int, int], dict[str, float]],
) -> pd.DataFrame:
    """Assemble a pairwise result table from a per-pair function.

    Port of ``sa_pair_matrix()``. R returns a matrix here; a frame is returned
    instead, because :func:`~statassist.core.posthoc_table` selects its columns by
    name and a frame is what carries them.

    Args:
        group_lv: Group levels, fixing the pair order.
        fun: Called with the two **zero-based** level indices - R passes the
            one-based ones - and returning the columns of
            :func:`posthoc_columns`.
    """
    wanted = posthoc_columns()
    pairs = level_pairs(group_lv)
    rows = []
    for i, j in zip(pairs["i"], pairs["j"], strict=False):
        produced = fun(int(i), int(j))
        absent = [column for column in wanted if column not in produced]
        if absent:
            raise SaValueError("a post-hoc row is missing column(s): " + ", ".join(absent) + ".")
        rows.append({column: produced[column] for column in wanted})
    return pd.DataFrame(rows, columns=wanted, dtype=float)


def tukey(samples: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Tukey's honestly significant difference.

    Port of ``sa_tukey()``. All pairwise mean differences judged against the
    studentised range, which controls the error rate over the whole set of
    comparisons at once. The p-values are therefore already family-wise and must
    not be adjusted again.

    Assumes equal variances, since every pair is judged against the same pooled
    mean square error. That is the assumption it shares with the one-way ANOVA it
    follows.

    R's ``ptukey`` and ``qtukey`` become
    :class:`scipy.stats.studentized_range`, whose ``sf`` and ``ppf`` are the same
    two functions under different names.

    Args:
        samples: One sample per group level, no missing values, in ``group_lv``
            order.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair, with the columns of :func:`posthoc_columns`.

    Raises:
        SaValueError: If the pooled mean square error is zero.

    References:
        Tukey, J. W. (1949). Comparing individual means in the analysis of
        variance. *Biometrics*, 5(2), 99-114.
    """
    names, arrays = as_samples(samples)
    k = len(arrays)
    sizes = [array.size for array in arrays]
    means = [float(np.mean(array)) for array in arrays]
    df = sum(sizes) - k

    ss_within = float(
        sum(float(np.sum((array - mean) ** 2)) for array, mean in zip(arrays, means, strict=False))
    )
    ms_within = ss_within / df
    if ms_within <= 0:
        raise SaValueError(
            "the pooled mean square error is zero, so no pairwise comparison can be scaled."
        )

    q_crit = float(stats.studentized_range.ppf(conf_level, k, df))

    def one_pair(i: int, j: int) -> dict[str, float]:
        estimate = means[i] - means[j]
        # The studentised range is the range of k means over the standard error
        # of one mean, so the divisor carries a 1/2 that a two-sample t does not.
        stderr = math.sqrt(ms_within / 2 * (1 / sizes[i] + 1 / sizes[j]))
        q_stat = estimate / stderr
        return stat_row(
            n1=sizes[i],
            n2=sizes[j],
            estimate=estimate,
            stderr=stderr,
            statistic=q_stat,
            df=df,
            pval=float(stats.studentized_range.sf(abs(q_stat), k, df)),
            lower_conf=estimate - q_crit * stderr,
            upper_conf=estimate + q_crit * stderr,
        )

    return pair_matrix(names, one_pair)


def games_howell(samples: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Games-Howell pairwise comparisons.

    Port of ``sa_games_howell()``. Tukey's procedure with the pooled variance
    replaced by a per-pair Welch standard error and Welch degrees of freedom, so
    it stays valid when the groups differ in spread or in size. The post-hoc
    partner of Welch's ANOVA.

    Like Tukey's test the p-values come from the studentised range and are
    already family-wise, so they must not be adjusted again.

    Args:
        samples: One sample per group level, no missing values, in ``group_lv``
            order.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair, with the columns of :func:`posthoc_columns`.

    Raises:
        SaValueError: If both groups of a pair have zero variance.

    References:
        Games, P. A. and Howell, J. F. (1976). Pairwise multiple comparison
        procedures with unequal n's and/or variances. *Journal of Educational
        Statistics*, 1(2), 113-125.
    """
    names, arrays = as_samples(samples)
    k = len(arrays)
    sizes = [array.size for array in arrays]
    means = [float(np.mean(array)) for array in arrays]
    variances = [float(np.var(array, ddof=1)) for array in arrays]

    def one_pair(i: int, j: int) -> dict[str, float]:
        v_i = variances[i] / sizes[i]
        v_j = variances[j] / sizes[j]
        if v_i + v_j <= 0:
            raise SaValueError(
                f"both groups of the pair {names[i]} - {names[j]} have zero variance."
            )
        estimate = means[i] - means[j]
        stderr = math.sqrt((v_i + v_j) / 2)
        df = (v_i + v_j) ** 2 / (v_i**2 / (sizes[i] - 1) + v_j**2 / (sizes[j] - 1))
        q_stat = estimate / stderr
        q_crit = float(stats.studentized_range.ppf(conf_level, k, df))
        return stat_row(
            n1=sizes[i],
            n2=sizes[j],
            estimate=estimate,
            stderr=stderr,
            statistic=q_stat,
            df=df,
            pval=float(stats.studentized_range.sf(abs(q_stat), k, df)),
            lower_conf=estimate - q_crit * stderr,
            upper_conf=estimate + q_crit * stderr,
        )

    return pair_matrix(names, one_pair)


def dunn(samples: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Dunn's pairwise rank comparisons.

    Port of ``sa_dunn()``. Compares mean ranks taken from the pooled ranking the
    Kruskal-Wallis test already computed, rather than re-ranking each pair on its
    own. Using the pooled ranks is what keeps the post-hoc conclusions consistent
    with the omnibus one; a set of separate rank-sum tests can contradict it.

    The variance carries a tie correction, so midranks do not inflate the
    statistic. Unlike Tukey's test these p-values are not family-wise on their
    own and are meant to be adjusted by the caller.

    Args:
        samples: One sample per group level, no missing values, in ``group_lv``
            order.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair. ``estimate`` is the mean rank difference and ``df`` is
        missing, the statistic being standard normal.

    Raises:
        SaValueError: If every observation is tied.

    References:
        Dunn, O. J. (1964). Multiple comparisons using rank sums.
        *Technometrics*, 6(3), 241-252.
    """
    names, arrays = as_samples(samples)
    sizes = [array.size for array in arrays]
    total = sum(sizes)
    pooled = np.concatenate(arrays)
    ranks = stats.rankdata(pooled)
    edges = np.concatenate(([0], np.cumsum(sizes)))
    mean_ranks = [float(np.mean(ranks[edges[i] : edges[i + 1]])) for i in range(len(arrays))]

    tie_sizes = np.unique(pooled, return_counts=True)[1].astype(float)
    tie_term = float(np.sum(tie_sizes**3 - tie_sizes)) / (12 * (total - 1))
    base_var = total * (total + 1) / 12 - tie_term
    if base_var <= 0:
        raise SaValueError("every observation is tied, leaving the rank variance at zero.")

    z_crit = float(stats.norm.ppf(1 - (1 - conf_level) / 2))

    def one_pair(i: int, j: int) -> dict[str, float]:
        estimate = mean_ranks[i] - mean_ranks[j]
        stderr = math.sqrt(base_var * (1 / sizes[i] + 1 / sizes[j]))
        z_stat = estimate / stderr
        return stat_row(
            n1=sizes[i],
            n2=sizes[j],
            estimate=estimate,
            stderr=stderr,
            statistic=z_stat,
            df=float("nan"),
            pval=float(2 * stats.norm.cdf(-abs(z_stat))),
            lower_conf=estimate - z_crit * stderr,
            upper_conf=estimate + z_crit * stderr,
        )

    return pair_matrix(names, one_pair)


def yuen_independent(
    x: Any,
    y: Any,
    tr: float = 0.2,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Yuen's trimmed mean test for two independent samples.

    Port of ``sa_yuen_independent()``. The independent counterpart of
    :func:`~statassist.kernel.robust.yuen_paired`. Trimming both tails before
    comparing means, and building the standard error from winsorised variances,
    keeps a few extreme observations from deciding the result.

    Args:
        x: First sample, no missing values.
        y: Second sample, no missing values. Need not be the same length.
        tr: Proportion trimmed at each tail, in ``[0, 0.5)``.
        alternative: One of :data:`~statassist.kernel.robust.ALTERNATIVES`, where
            ``"greater"`` tests whether ``x`` exceeds ``y``.
        conf_level: Confidence level of the reported interval.

    Returns:
        ``x_trim_mean``, ``y_trim_mean``, ``trim_diff``, ``stderr``,
        ``yuen_stat``, ``df``, ``pval``, ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If fewer than 2 observations survive trimming on either
            side, or if both winsorised samples are constant.

    References:
        Yuen, K. K. (1974). The two-sample trimmed t for unequal population
        variances. *Biometrika*, 61(1), 165-170.
    """
    sample_x = as_sample(x, "x")
    sample_y = as_sample(y, "y")
    n_x = sample_x.size
    n_y = sample_y.size
    h_x = n_x - 2 * math.floor(tr * n_x)
    h_y = n_y - 2 * math.floor(tr * n_y)
    if h_x < 2 or h_y < 2:
        raise SaValueError(
            f"fewer than 2 observations survive trimming {tr} from each tail ({h_x} and {h_y})."
        )

    d_x = (n_x - 1) * float(np.var(winsorize(sample_x, tr), ddof=1)) / (h_x * (h_x - 1))
    d_y = (n_y - 1) * float(np.var(winsorize(sample_y, tr), ddof=1)) / (h_y * (h_y - 1))
    stderr = math.sqrt(d_x + d_y)
    if not math.isfinite(stderr) or stderr <= 0:
        raise SaValueError(
            "both winsorised samples are constant, leaving the standard error at "
            "zero and the statistic undefined."
        )

    df = (d_x + d_y) ** 2 / (d_x**2 / (h_x - 1) + d_y**2 / (h_y - 1))
    x_trim_mean = trimmed_mean(sample_x, tr)
    y_trim_mean = trimmed_mean(sample_y, tr)
    trim_diff = x_trim_mean - y_trim_mean
    yuen_stat = trim_diff / stderr
    lower, upper = t_ci(trim_diff, stderr, df, alternative, conf_level)

    return {
        "x_trim_mean": x_trim_mean,
        "y_trim_mean": y_trim_mean,
        "trim_diff": trim_diff,
        "stderr": stderr,
        "yuen_stat": yuen_stat,
        "df": df,
        "pval": t_pval(yuen_stat, df, alternative),
        "lower_conf": lower,
        "upper_conf": upper,
    }


def pairwise_yuen(
    samples: Any,
    tr: float = 0.2,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Pairwise Yuen comparisons between independent groups.

    Port of ``sa_pairwise_yuen()``. The post-hoc partner of the trimmed mean
    ANOVA, run pair by pair on the same trimming proportion the omnibus test
    used. Its p-values are not family-wise and are meant to be adjusted by the
    caller.

    Args:
        samples: One sample per group level, no missing values, in ``group_lv``
            order.
        tr: Proportion trimmed at each tail, in ``[0, 0.5)``.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair, with the columns of :func:`posthoc_columns`.
    """
    names, arrays = as_samples(samples)
    sizes = [array.size for array in arrays]

    def one_pair(i: int, j: int) -> dict[str, float]:
        result = yuen_independent(arrays[i], arrays[j], tr=tr, conf_level=conf_level)
        return stat_row(
            n1=sizes[i],
            n2=sizes[j],
            estimate=result["trim_diff"],
            stderr=result["stderr"],
            statistic=result["yuen_stat"],
            df=result["df"],
            pval=result["pval"],
            lower_conf=result["lower_conf"],
            upper_conf=result["upper_conf"],
        )

    return pair_matrix(names, one_pair)


def pairwise_paired_t(mat: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Pairwise paired t-tests between repeated conditions.

    Port of ``sa_pairwise_paired_t()``. The post-hoc partner of repeated measures
    ANOVA. Each pair is tested on its own differences rather than against the
    pooled residual error, so a pair whose difference happens to be far more
    variable than the rest is not judged as though it were not. Its p-values are
    meant to be adjusted by the caller.

    ``estimate`` is the difference of the two column means rather than the mean
    of the differences. On a complete matrix the two are the same number, and the
    column form is the one that matches how the omnibus row reports the condition
    means.

    Args:
        mat: Subjects-by-conditions matrix, complete, columns in ``group_lv``
            order. A :class:`pandas.DataFrame` names its own levels; anything
            else is labelled by position.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair, with the columns of :func:`posthoc_columns`.

    Raises:
        SaValueError: If some pair's differences are constant. R reaches the same
            refusal through ``t.test``, which calls such data "essentially
            constant"; the pair is named here, since the caller has to know which
            of the pairs it was.
    """
    names = condition_names(mat)
    array = as_matrix(mat)
    n = array.shape[0]

    def one_pair(i: int, j: int) -> dict[str, float]:
        differences = array[:, i] - array[:, j]
        df = n - 1
        stderr = math.sqrt(float(np.var(differences, ddof=1)) / n)
        if stderr <= 0:
            raise SaValueError(
                f"in the pair {names[i]} - {names[j]} the differences are constant, "
                "leaving the standard error at zero and the statistic undefined."
            )
        estimate = float(np.mean(array[:, i])) - float(np.mean(array[:, j]))
        statistic = float(np.mean(differences)) / stderr
        lower, upper = t_ci(float(np.mean(differences)), stderr, df, "two.sided", conf_level)
        return stat_row(
            n1=n,
            n2=n,
            estimate=estimate,
            stderr=stderr,
            statistic=statistic,
            df=df,
            pval=t_pval(statistic, df, "two.sided"),
            lower_conf=lower,
            upper_conf=upper,
        )

    return pair_matrix(names, one_pair)


def conover(mat: Any, conf_level: float = 0.95) -> pd.DataFrame:
    """Conover's pairwise comparisons after a Friedman test.

    Port of ``sa_conover()``. Compares the within-subject rank sums the Friedman
    test already formed, scaled by the residual variability of those same ranks
    and judged against a t distribution. Working from the within-block ranking is
    what keeps the post-hoc conclusions consistent with the omnibus one.

    Its p-values are not family-wise and are meant to be adjusted by the caller.

    Args:
        mat: Subjects-by-conditions matrix, complete, columns in ``group_lv``
            order.
        conf_level: Confidence level for the reported intervals.

    Returns:
        One row per pair. ``estimate`` is the rank sum difference.

    Raises:
        SaValueError: If every subject ranks the conditions identically.

    References:
        Conover, W. J. (1999). *Practical Nonparametric Statistics*, 3rd edition.
    """
    names = condition_names(mat)
    array = as_matrix(mat)
    n, k = array.shape
    # Ranked within each subject, so only the ordering a subject produces counts.
    ranks = np.vstack([stats.rankdata(row) for row in array])
    rank_sums = ranks.sum(axis=0)

    # a - b is the sum of squares of the ranks left after the condition rank sums
    # are accounted for, which is the residual the pairwise scale is built from.
    a = float(np.sum(ranks**2))
    b = float(np.sum(rank_sums**2)) / n

    df = (n - 1) * (k - 1)
    variance = 2 * n * (a - b) / df
    if not math.isfinite(variance) or variance <= 0:
        raise SaValueError(
            "every subject ranks the conditions identically, leaving the "
            "residual rank variance at zero."
        )
    stderr = math.sqrt(variance)
    t_crit = float(stats.t.ppf(1 - (1 - conf_level) / 2, df))

    def one_pair(i: int, j: int) -> dict[str, float]:
        estimate = float(rank_sums[i] - rank_sums[j])
        t_stat = estimate / stderr
        return stat_row(
            n1=n,
            n2=n,
            estimate=estimate,
            stderr=stderr,
            statistic=t_stat,
            df=df,
            pval=float(2 * stats.t.cdf(-abs(t_stat), df)),
            lower_conf=estimate - t_crit * stderr,
            upper_conf=estimate + t_crit * stderr,
        )

    return pair_matrix(names, one_pair)
