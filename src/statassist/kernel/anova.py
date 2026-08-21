"""Omnibus kernels for three or more groups.

Port of ``R/kernel_anova.R``.

None of the omnibus tests reports a confidence interval. An omnibus test says
that the conditions are not all alike; it does not say by how much, and there is
no single quantity for an interval to be about. The intervals of a multi-group
comparison live in the post-hoc table, where each row is one contrast and does
have a scale of its own. ``lower_conf`` and ``upper_conf`` are therefore present
and missing in every row here, which the result contract allows: it requires
that the columns exist, not that they are finite.

``kruskal`` and ``friedman`` reproduce ``stats::kruskal.test()`` and
``stats::friedman.test()`` rather than calling anything: SciPy's
:func:`scipy.stats.kruskal` applies the same tie correction but
:func:`scipy.stats.friedmanchisquare` applies none at all, so a within-subject
tie would put the two languages a long way apart.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import SaValueError
from ._shared import as_matrix, as_samples
from .robust import trimmed_mean, winsorize

__all__ = [
    "friedman",
    "kruskal",
    "oneway_anova",
    "rm_anova",
    "sphericity",
    "split_groups",
    "welch_anova",
    "yuen_anova",
]


def split_groups(
    values: Any,
    group: Any,
    n_min: int = 2,
) -> dict[str, np.ndarray]:
    """Split a numeric vector into one sample per group level.

    Port of ``sa_split_groups()``. Missing values are dropped inside each level,
    which is the independent-sample rule the rest of the package uses. Levels
    left with too few observations are an error rather than a silently smaller
    design.

    Args:
        values: Numeric vector, missing values included.
        group: The levels to split on, same length as ``values``. A
            :class:`pandas.Categorical` - what
            :func:`~statassist.core.validate_wide_input` hands back - carries the
            display order in its categories. Anything else is read as a plain
            vector and its levels are the sorted unique values, which is what
            R's ``factor()`` would have done to it.
        n_min: Smallest acceptable number of usable observations per level.

    Returns:
        One array per level, in level order, missing values gone.

    Raises:
        SaValueError: If any level is left with fewer than ``n_min``
            observations. Every short level is named, so a design with several
            of them is reported once.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    categorical = group if isinstance(group, pd.Categorical) else pd.Categorical(group)
    if len(categorical) != array.size:
        raise SaValueError(
            f"`group` must have one entry per value, got {len(categorical)} for {array.size}."
        )

    codes = np.asarray(categorical.codes)
    samples: dict[str, np.ndarray] = {}
    for index, level in enumerate(categorical.categories):
        sample = array[codes == index]
        samples[str(level)] = sample[~np.isnan(sample)]

    short = [(name, sample.size) for name, sample in samples.items() if sample.size < n_min]
    if short:
        detail = ", ".join(f"{name} = {size}" for name, size in short)
        raise SaValueError(f"needs at least {n_min} usable observation(s) per group; {detail}.")
    return samples


def oneway_anova(samples: Any) -> dict[str, float]:
    """One-way analysis of variance.

    Port of ``sa_oneway_anova()``. The equal-variance omnibus F test, written out
    rather than taken from ``stats::oneway.test()`` because the sums of squares
    are needed anyway: Tukey's post-hoc test runs on the same mean square error,
    and computing it twice from two different code paths is how the two end up
    disagreeing.

    Args:
        samples: One sample per group level, no missing values.

    Returns:
        ``n_used``, ``n_groups``, ``f_stat``, ``df1``, ``df2``, ``eta_sq``,
        ``omega_sq``, ``pval``, ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If there are no residual degrees of freedom left, or if
            every group has zero variance.

    References:
        Fisher, R. A. (1925). *Statistical Methods for Research Workers*.

        Okada, K. (2013). Is omega squared less biased? *Behaviormetrika*,
        40(2), 129-147.
    """
    _, arrays = as_samples(samples)
    k = len(arrays)
    sizes = np.array([array.size for array in arrays], dtype=float)
    total = float(sizes.sum())
    df1 = k - 1
    df2 = int(total) - k
    if df2 < 1:
        raise SaValueError(
            "needs more observations than groups to leave any residual degrees of freedom."
        )

    means = np.array([float(np.mean(array)) for array in arrays])
    grand = float(np.sum(sizes * means) / total)

    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(
        sum(float(np.sum((array - mean) ** 2)) for array, mean in zip(arrays, means, strict=True))
    )
    ss_total = ss_between + ss_within

    if ss_within <= 0:
        raise SaValueError(
            "every group has zero variance, leaving the mean square error at "
            "zero and the F statistic undefined."
        )

    ms_within = ss_within / df2
    f_stat = (ss_between / df1) / ms_within

    return {
        "n_used": total,
        "n_groups": float(k),
        "f_stat": f_stat,
        "df1": float(df1),
        "df2": float(df2),
        "eta_sq": ss_between / ss_total,
        # Omega squared subtracts the variance the grouping would explain by
        # chance alone, so it can go negative when the groups are
        # indistinguishable. That is not clipped: a negative value is the
        # estimate saying so.
        "omega_sq": (ss_between - df1 * ms_within) / (ss_total + ms_within),
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }


def welch_anova(samples: Any) -> dict[str, float]:
    """Welch's heteroscedastic one-way analysis of variance.

    Port of ``sa_welch_anova()``. Weights each group by its own precision instead
    of pooling the variances, so unequal spreads or unequal group sizes no longer
    distort the test. The effect size columns are the ordinary sums-of-squares
    ones: they describe how far apart the group means are, which does not change
    with the choice of test.

    Args:
        samples: One sample per group level, no missing values, at least 2 each.

    Returns:
        The same columns as :func:`oneway_anova`.

    Raises:
        SaValueError: If a group holds fewer than 2 observations, or if any group
            has zero variance and so an infinite Welch weight.

    References:
        Welch, B. L. (1951). On the comparison of several mean values.
        *Biometrika*, 38(3-4), 330-336.
    """
    names, arrays = as_samples(samples)
    k = len(arrays)
    sizes = np.array([array.size for array in arrays], dtype=float)
    if (sizes < 2).any():
        raise SaValueError(
            "Welch's ANOVA needs at least 2 observations per group to estimate "
            "a within-group variance."
        )

    means = np.array([float(np.mean(array)) for array in arrays])
    variances = np.array([float(np.var(array, ddof=1)) for array in arrays])
    if (variances <= 0).any():
        zero = ", ".join(name for name, var in zip(names, variances, strict=False) if var <= 0)
        raise SaValueError(f"group(s) with zero variance leave the Welch weight infinite: {zero}.")

    weights = sizes / variances
    sum_w = float(weights.sum())
    weighted_mean = float(np.sum(weights * means) / sum_w)

    lam = float(np.sum((1 - weights / sum_w) ** 2 / (sizes - 1)))
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2)) / (k - 1)
    denominator = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    f_stat = numerator / denominator

    df1 = k - 1
    df2 = 1 / (3 / (k**2 - 1) * lam)

    reference = oneway_anova(samples)

    return {
        "n_used": float(sizes.sum()),
        "n_groups": float(k),
        "f_stat": f_stat,
        "df1": float(df1),
        "df2": df2,
        "eta_sq": reference["eta_sq"],
        "omega_sq": reference["omega_sq"],
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }


def yuen_anova(samples: Any, tr: float = 0.2) -> dict[str, float]:
    """Yuen's trimmed mean one-way analysis of variance.

    Port of ``sa_yuen_anova()``. The robust member of the independent omnibus
    family: Welch's construction applied to trimmed means and winsorised
    variances, so heavy tails and stray observations lose the leverage they have
    over an ordinary F test.

    ``robust_eta_sq`` is defined here rather than borrowed. It is the spread of
    the trimmed means around their unweighted centre, divided by that spread plus
    the mean rescaled winsorised variance, where the rescaling by
    ``(1 - 2 * tr) ** 2`` puts a winsorised variance back on the scale of the
    trimmed mean it belongs to. It is zero when the trimmed means coincide,
    approaches one as they separate, and does not change if every observation is
    multiplied by a constant. It is not Wilcox's explanatory measure and does not
    reproduce it.

    Args:
        samples: One sample per group level, no missing values.
        tr: Proportion trimmed at each tail, in ``[0, 0.5)``.

    Returns:
        ``n_used``, ``n_groups``, ``f_stat``, ``df1``, ``df2``,
        ``robust_eta_sq``, ``pval``, ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If fewer than 2 observations survive trimming in some
            group, or if a group's winsorised values are constant.

    References:
        Yuen, K. K. (1974). The two-sample trimmed t for unequal population
        variances. *Biometrika*, 61(1), 165-170.
    """
    names, arrays = as_samples(samples)
    k = len(arrays)
    sizes = np.array([array.size for array in arrays], dtype=float)

    h = np.array([array.size - 2 * math.floor(tr * array.size) for array in arrays], dtype=float)
    if (h < 2).any():
        short = ", ".join(name for name, kept in zip(names, h, strict=False) if kept < 2)
        raise SaValueError(
            f"fewer than 2 observations survive trimming {tr} from each tail in group(s): {short}."
        )

    trim_means = np.array([trimmed_mean(array, tr) for array in arrays])
    win_vars = np.array([float(np.var(winsorize(array, tr), ddof=1)) for array in arrays])
    if (win_vars <= 0).any():
        flat = ", ".join(name for name, var in zip(names, win_vars, strict=False) if var <= 0)
        raise SaValueError(
            "group(s) whose winsorised values are constant leave the trimmed "
            f"weight infinite: {flat}."
        )

    # d is the squared standard error of one trimmed mean; its reciprocal is the
    # Welch weight, exactly as `welch_anova` uses n / var.
    d = (sizes - 1) * win_vars / (h * (h - 1))
    weights = 1 / d
    sum_w = float(weights.sum())
    weighted_mean = float(np.sum(weights * trim_means) / sum_w)

    lam = float(np.sum((1 - weights / sum_w) ** 2 / (h - 1)))
    numerator = float(np.sum(weights * (trim_means - weighted_mean) ** 2)) / (k - 1)
    f_stat = numerator / (1 + 2 * (k - 2) / (k**2 - 1) * lam)

    df1 = k - 1
    df2 = 1 / (3 / (k**2 - 1) * lam)

    between = float(np.sum((trim_means - np.mean(trim_means)) ** 2)) / k
    within = float(np.mean(win_vars / (1 - 2 * tr) ** 2))

    return {
        "n_used": float(sizes.sum()),
        "n_groups": float(k),
        "f_stat": f_stat,
        "df1": float(df1),
        "df2": df2,
        "robust_eta_sq": between / (between + within),
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }


def _tie_correction(values: np.ndarray) -> float:
    """``sum(t^3 - t)`` over the tie group sizes of a pooled sample."""
    counts = np.unique(values, return_counts=True)[1].astype(float)
    return float(np.sum(counts**3 - counts))


def kruskal(samples: Any) -> dict[str, float]:
    """Kruskal-Wallis rank sum test.

    Port of ``sa_kruskal()``. The rank-based omnibus test. It asks whether one
    group tends to produce larger values than another, which is only a statement
    about medians when the distributions share a shape.

    The statistic reproduces ``stats::kruskal.test()``: the group rank sums are
    squared and divided by the group sizes, and the result is corrected by the
    pooled tie term. R would return ``NaN`` in silence when every value is tied,
    since the correction divides by zero, so that case is refused here and
    becomes a missing row with a named reason like every other test that cannot
    run.

    Args:
        samples: One sample per group level, no missing values.

    Returns:
        ``n_used``, ``n_groups``, ``h_stat``, ``df``, ``epsilon_sq``,
        ``eta_sq_rank``, ``pval``, ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If every observation takes the same value.

    References:
        Kruskal, W. H. and Wallis, W. A. (1952). Use of ranks in one-criterion
        variance analysis. *JASA*, 47(260), 583-621.

        Tomczak, M. and Tomczak, E. (2014). The need to report effect size
        estimates revisited. *Trends in Sport Sciences*, 1(21), 19-25.
    """
    _, arrays = as_samples(samples)
    k = len(arrays)
    pooled = np.concatenate(arrays)
    total = pooled.size
    if np.unique(pooled).size < 2:
        raise SaValueError(
            "every observation takes the same value, so the ranks carry no "
            "information and the tie correction is undefined."
        )

    ranks = stats.rankdata(pooled)
    sizes = np.array([array.size for array in arrays], dtype=float)
    edges = np.concatenate(([0], np.cumsum(sizes).astype(int)))
    rank_sums = np.array(
        [float(np.sum(ranks[edges[i] : edges[i + 1]])) for i in range(k)], dtype=float
    )

    raw = float(np.sum(rank_sums**2 / sizes))
    h_stat = (12 * raw / (total * (total + 1)) - 3 * (total + 1)) / (
        1 - _tie_correction(pooled) / (total**3 - total)
    )
    df = k - 1

    return {
        "n_used": float(total),
        "n_groups": float(k),
        "h_stat": h_stat,
        "df": float(df),
        # Both rescale H onto [0, 1]; epsilon squared divides by the largest
        # value H could take, eta squared by the residual degrees of freedom.
        "epsilon_sq": h_stat * (total + 1) / (total**2 - 1),
        "eta_sq_rank": (h_stat - k + 1) / (total - k),
        "pval": float(stats.chi2.sf(h_stat, df)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }


def sphericity(mat: Any) -> dict[str, float]:
    """Mauchly's sphericity test and the two epsilon corrections.

    Port of ``sa_sphericity()``. All three quantities come from the same
    orthonormal contrast of the condition covariance matrix, so they are computed
    together.

    The Huynh-Feldt epsilon can exceed one, which is meaningless as a correction
    factor, so it is capped. Both epsilons are also floored at the lower bound
    ``1 / (k - 1)``, the value they take when sphericity fails as badly as it
    can.

    Two things to know before comparing this with anything else.

    ``stats::mauchly.test()`` evaluates the same p-value expression with the
    number of conditions where the published form has the contrast rank, so R's
    own two functions disagree in the fourth decimal place. The single-symbol
    form is the published one and is what both the R kernel and this port use, so
    the number to compare against is ``sa_sphericity()`` and not
    ``mauchly.test()``.

    The contrast basis comes from a QR decomposition, and the one NumPy produces
    is not column-for-column the one R produces. It does not have to be: every
    quantity below is a function of the eigenvalues of ``C' S C``, and changing
    ``C`` for another orthonormal basis of the same subspace conjugates that
    matrix by an orthogonal factor, which leaves its eigenvalues alone.

    Args:
        mat: Subjects-by-conditions numeric matrix, complete.

    Returns:
        ``mauchly_w``, ``mauchly_pval``, ``gg_eps``, ``hf_eps``. The first two
        are missing, and both epsilons sit at the lower bound, when the condition
        covariance is singular.
    """
    array = as_matrix(mat)
    n, k = array.shape
    p = k - 1
    f = n - 1

    lower_bound = 1 / p
    fallback = {
        "mauchly_w": float("nan"),
        "mauchly_pval": float("nan"),
        "gg_eps": lower_bound,
        "hf_eps": lower_bound,
    }
    if n <= k:
        # The condition covariance is singular below this point, so both the
        # determinant and its eigenvalues stop meaning anything. Falling back to
        # the lower bound applies the most conservative correction available
        # rather than reporting no correction at all.
        return fallback

    covariance = np.cov(array, rowvar=False, ddof=1)

    # Orthonormal contrasts spanning the k - 1 differences between conditions.
    # The QR of the deviation basis gives a basis orthogonal to the unit vector,
    # which is what sphericity is defined relative to.
    contrasts = np.linalg.qr(np.eye(k) - 1 / k)[0][:, :p]
    transformed = contrasts.T @ covariance @ contrasts
    eigenvalues = np.linalg.eigvalsh(transformed)
    if (eigenvalues <= 0).any():
        return fallback

    total = float(eigenvalues.sum())
    w = float(np.prod(eigenvalues)) / (total / p) ** p

    # Mauchly's statistic is only asymptotically chi-square, so the p-value uses
    # the two-term expansion rather than the leading term alone.
    rho = 1 - (2 * p**2 + p + 2) / (6 * p * f)
    chi_sq = -f * rho * math.log(w)
    chi_df = p * (p + 1) / 2 - 1
    weight = (
        (p + 2) * (p - 1) * (p - 2) * (2 * p**3 + 6 * p**2 + 3 * p + 2) / (288 * (f * p * rho) ** 2)
    )
    lead = float(stats.chi2.sf(chi_sq, chi_df))
    correction = float(stats.chi2.sf(chi_sq, chi_df + 4))

    gg = total**2 / (p * float(np.sum(eigenvalues**2)))
    gg = min(max(gg, lower_bound), 1.0)
    hf = (n * p * gg - 2) / (p * (f - p * gg))
    hf = min(max(hf, lower_bound), 1.0)

    return {
        "mauchly_w": w,
        "mauchly_pval": lead + weight * (correction - lead),
        "gg_eps": gg,
        "hf_eps": hf,
    }


def rm_anova(mat: Any) -> dict[str, float]:
    """One-way repeated measures analysis of variance.

    Port of ``sa_rm_anova()``. Removes the between-subject variation before
    testing the conditions, which is what makes a within-subject design more
    sensitive than the same number of independent observations.

    The uncorrected F test assumes sphericity: that every pair of conditions has
    the same variance of differences. Mauchly's test reports whether that holds
    and the Greenhouse-Geisser and Huynh-Feldt epsilons say how badly it fails.
    Both corrected p-values are returned alongside the uncorrected one instead of
    one being chosen, since which to trust is a judgement about the design.

    Args:
        mat: Subjects-by-conditions numeric matrix, complete, at least 2 rows.

    Returns:
        ``n_used``, ``n_groups``, ``f_stat``, ``df1``, ``df2``,
        ``partial_eta_sq``, ``gen_eta_sq``, ``mauchly_w``, ``mauchly_pval``,
        ``gg_eps``, ``pval_gg``, ``hf_eps``, ``pval_hf``, ``pval``,
        ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If there are fewer than 2 complete subjects, or if the
            subject-by-condition residuals are all zero.

    References:
        Mauchly, J. W. (1940). Significance test for sphericity. *Annals of
        Mathematical Statistics*, 11(2), 204-209.

        Greenhouse, S. W. and Geisser, S. (1959). On methods in the analysis of
        profile data. *Psychometrika*, 24(2), 95-112.

        Huynh, H. and Feldt, L. S. (1976). Estimation of the Box correction.
        *Journal of Educational Statistics*, 1(1), 69-82.

        Bakeman, R. (2005). Recommended effect size statistics for repeated
        measures designs. *Behavior Research Methods*, 37(3), 379-384.
    """
    array = as_matrix(mat)
    n, k = array.shape
    if n < 2:
        raise SaValueError(f"needs at least 2 complete subjects, got {n}.")

    grand = float(np.mean(array))
    condition_means = np.mean(array, axis=0)
    subject_means = np.mean(array, axis=1)

    ss_condition = n * float(np.sum((condition_means - grand) ** 2))
    ss_subject = k * float(np.sum((subject_means - grand) ** 2))
    ss_total = float(np.sum((array - grand) ** 2))
    ss_error = ss_total - ss_condition - ss_subject

    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    if ss_error <= 0:
        raise SaValueError(
            "the subject-by-condition residuals are all zero, leaving the F statistic undefined."
        )
    ms_error = ss_error / df2
    f_stat = (ss_condition / df1) / ms_error

    spread = sphericity(array)
    gg = spread["gg_eps"]
    hf = spread["hf_eps"]

    return {
        "n_used": float(n),
        "n_groups": float(k),
        "f_stat": f_stat,
        "df1": float(df1),
        "df2": float(df2),
        "partial_eta_sq": ss_condition / (ss_condition + ss_error),
        # Generalised eta squared keeps the between-subject variance in the
        # denominator, which is what makes it comparable with the eta squared of
        # an independent design measuring the same thing.
        "gen_eta_sq": ss_condition / (ss_condition + ss_subject + ss_error),
        "mauchly_w": spread["mauchly_w"],
        "mauchly_pval": spread["mauchly_pval"],
        "gg_eps": gg,
        "pval_gg": float(stats.f.sf(f_stat, df1 * gg, df2 * gg)),
        "hf_eps": hf,
        "pval_hf": float(stats.f.sf(f_stat, df1 * hf, df2 * hf)),
        "pval": float(stats.f.sf(f_stat, df1, df2)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }


def friedman(mat: Any) -> dict[str, float]:
    """Friedman rank sum test.

    Port of ``sa_friedman()``. The rank-based counterpart of repeated measures
    ANOVA: observations are ranked within each subject, so only the ordering of
    the conditions inside a subject matters and no distributional assumption is
    made across subjects.

    The statistic reproduces ``stats::friedman.test()``, tie correction included.
    :func:`scipy.stats.friedmanchisquare` applies no correction at all, so a
    subject who gives two conditions the same value would put the two languages
    a long way apart.

    Args:
        mat: Subjects-by-conditions numeric matrix, complete.

    Returns:
        ``n_used``, ``n_groups``, ``chi_sq``, ``df``, ``kendalls_w``, ``pval``,
        ``lower_conf``, ``upper_conf``.

    Raises:
        SaValueError: If no subject distinguishes the conditions.

    References:
        Friedman, M. (1937). The use of ranks to avoid the assumption of
        normality implicit in the analysis of variance. *JASA*, 32(200), 675-701.

        Kendall, M. G. and Babington Smith, B. (1939). The problem of m
        rankings. *Annals of Mathematical Statistics*, 10(3), 275-287.
    """
    array = as_matrix(mat)
    n, k = array.shape
    # A subject who gives every condition the same value contributes only ties.
    # With no subject ranking anything, R returns NaN in silence; refusing turns
    # that into a missing row with a reason attached.
    if all(np.unique(row).size < 2 for row in array):
        raise SaValueError(
            "no subject distinguishes the conditions, so the within-subject "
            "ranks carry no information."
        )

    ranks = np.vstack([stats.rankdata(row) for row in array])
    # The tie term is per subject, not pooled: a value shared by two subjects is
    # not a tie in a within-subject ranking.
    ties = float(sum(_tie_correction(row) for row in array))

    chi_sq = (
        12
        * float(np.sum((ranks.sum(axis=0) - n * (k + 1) / 2) ** 2))
        / (n * k * (k + 1) - ties / (k - 1))
    )
    df = k - 1

    return {
        "n_used": float(n),
        "n_groups": float(k),
        "chi_sq": chi_sq,
        "df": float(df),
        # Kendall's W is the Friedman statistic expressed as agreement between
        # subjects: 0 when the subjects rank the conditions independently, 1 when
        # they all produce the same ranking.
        "kendalls_w": chi_sq / (n * (k - 1)),
        "pval": float(stats.chi2.sf(chi_sq, df)),
        "lower_conf": float("nan"),
        "upper_conf": float("nan"),
    }
