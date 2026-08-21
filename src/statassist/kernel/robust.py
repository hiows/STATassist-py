"""Robust test kernels.

Port of ``R/kernel_robust.R``. The R file explains why these are written out
rather than wrapped: ``scipy.stats.brunnermunzel`` reports no interval, and
neither SciPy nor statsmodels covers the dependent-samples Yuen test at all, so
one of the two languages would have had to be the odd one out anyway.

Two of the R idioms here need saying out loud, because they are what a naive
translation gets wrong.

``stats::var()`` and ``stats::sd()`` are the *sample* statistics, so every
variance in this module carries ``ddof=1``. NumPy's default of ``ddof=0`` would
be off by a factor of ``n / (n - 1)``, which is small enough to survive a casual
look at the output and large enough to be wrong.

``mean(v, trim = tr)`` drops ``floor(n * tr)`` values from each end of the
*sorted* vector, which is not the same as trimming the tails by value. It shares
that count with :func:`winsorize`, which is why the two agree about which
observations are extreme.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core.errors import SaValueError
from ._shared import ALTERNATIVES, as_sample, check_alternative

__all__ = [
    "ALTERNATIVES",
    "brunner_munzel",
    "t_ci",
    "t_pval",
    "trimmed_mean",
    "winsorize",
    "winsorized_normal_var",
    "yuen_paired",
]


def t_pval(stat: float, df: float, alternative: str) -> float:
    """Two-sided or one-sided p-value from a t-distributed statistic.

    Port of ``sa_t_pval()``. ``"greater"`` always means the first sample exceeds
    the second, so the caller must hand over a statistic that is positive in
    that case.

    R's ``switch()`` returns ``NULL`` for a value it does not recognise, which
    then vanishes from the result vector; here an unrecognised ``alternative``
    is refused by name.

    >>> round(t_pval(2.31, 7, "two.sided"), 6)
    0.054187
    >>> round(t_pval(2.31, 7, "greater"), 6)
    0.027093
    """
    check_alternative(alternative)
    if alternative == "two.sided":
        return float(2 * stats.t.cdf(-abs(stat), df))
    if alternative == "greater":
        return float(stats.t.sf(stat, df))
    return float(stats.t.cdf(stat, df))


def t_ci(
    est: float,
    se: float,
    df: float,
    alternative: str,
    conf_level: float,
    bounds: tuple[float, float] = (-math.inf, math.inf),
) -> tuple[float, float]:
    """Confidence interval for an estimate with a t-distributed pivot.

    Port of ``sa_t_ci()``. A one-sided alternative leaves the side it does not
    test open, which is what ``t.test()`` and ``wilcox.test()`` do, so the three
    tables of a comparison can be read the same way. ``bounds`` sets what "open"
    means: an unbounded quantity runs to infinity, a probability stops at 0 and
    1.

    >>> lower, upper = t_ci(1.25, 0.4, 9, "greater", 0.95)
    >>> round(lower, 6), upper
    (0.516755, inf)
    """
    check_alternative(alternative)
    alpha = 1 - conf_level
    if alternative == "two.sided":
        half = float(stats.t.ppf(1 - alpha / 2, df)) * se
        return est - half, est + half
    if alternative == "greater":
        return est - float(stats.t.ppf(1 - alpha, df)) * se, bounds[1]
    return bounds[0], est + float(stats.t.ppf(1 - alpha, df)) * se


def _trim_count(n: int, tr: float) -> int:
    """How many observations each tail loses.

    ``floor(tr * n)``, shared by :func:`winsorize` and :func:`trimmed_mean` so
    that the two cannot disagree about which observations are extreme.
    """
    return int(math.floor(tr * n))


def winsorize(v: Any, tr: float) -> np.ndarray:
    """Winsorise the tails of a sample.

    Port of ``sa_winsorize()``. The ``floor(tr * n)`` smallest values are pulled
    up to the next one and the same number of largest values pulled down, leaving
    the length unchanged. Order is preserved, which is what lets the result be
    used for a covariance as well as for a variance.

    Args:
        v: Numeric vector without missing values.
        tr: Proportion winsorised at each tail, in ``[0, 0.5)``.

    >>> winsorize([1.0, 2.0, 3.0, 4.0, 10.0], 0.2)
    array([2., 2., 3., 4., 4.])
    """
    array = np.asarray(v, dtype=float).reshape(-1).copy()
    n = array.size
    g = _trim_count(n, tr)
    sorted_values = np.sort(array)
    lower = sorted_values[g]
    upper = sorted_values[n - g - 1]
    np.clip(array, lower, upper, out=array)
    return array


def trimmed_mean(v: Any, tr: float) -> float:
    """Mean of the sample with ``floor(tr * n)`` values gone from each tail.

    Port of R's ``mean(v, trim = tr)``, written out because it is the trimming
    count rather than a quantile that has to line up with :func:`winsorize`.

    >>> trimmed_mean([1.0, 2.0, 3.0, 4.0, 10.0], 0.2)
    3.0
    """
    array = np.asarray(v, dtype=float).reshape(-1)
    n = array.size
    g = _trim_count(n, tr)
    kept = np.sort(array)[g : n - g] if g > 0 else array
    return float(np.mean(kept))


def winsorized_normal_var(tr: float) -> float:
    """Variance of a winsorised standard normal sample.

    Port of ``sa_winsorized_normal_var()``. Winsorising shortens the tails, so a
    winsorised variance underestimates the variance of the underlying normal;
    dividing by this factor rescales it back, which is what makes the robust
    effect size of :func:`yuen_paired` comparable to an ordinary standardised
    difference.

    >>> round(winsorized_normal_var(0.2), 7)
    0.4120867
    """
    if tr <= 0:
        return 1.0
    z = float(stats.norm.ppf(tr))
    return (1 - 2 * tr) + 2 * z * float(stats.norm.pdf(z)) + 2 * z**2 * tr


def brunner_munzel(
    x: Any,
    y: Any,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Brunner-Munzel test for two independent samples.

    Port of ``sa_brunner_munzel()``. The nonparametric Behrens-Fisher problem:
    unlike the Wilcoxon rank-sum test it does not assume the two distributions
    share a shape, so it stays valid when the groups differ in spread.

    ``relative_effect`` is ``P(X > Y) + 0.5 * P(X = Y)``, above 0.5 when ``x``
    tends to be the larger sample, and ``bm_stat`` is positive in that same case.
    Published presentations of the test state the estimate the other way round;
    the two are complements and the p-value is identical either way. This
    direction is used so that every estimate in a result table points the same
    way.

    :func:`scipy.stats.brunnermunzel` computes the same statistic but reports no
    interval, which is the whole reason this is written out: the interval comes
    from :func:`t_ci` with the relative effect held inside ``[0, 1]``.

    Args:
        x: First sample, no missing values, at least 2 long.
        y: Second sample, same.
        alternative: One of :data:`ALTERNATIVES`, where ``"greater"`` tests
            whether ``x`` exceeds ``y``.
        conf_level: Confidence level of the reported interval.

    Returns:
        ``relative_effect``, ``bm_stat``, ``df``, ``pval``, ``lower_conf``,
        ``upper_conf``.

    Raises:
        SaValueError: If the groups do not overlap, which leaves the variance
            estimate at zero and the statistic undefined.

    References:
        Brunner, E. and Munzel, U. (2000). The nonparametric Behrens-Fisher
        problem: asymptotic theory and a small-sample approximation.
        *Biometrical Journal*, 42(1), 17-25.
    """
    sample_x = as_sample(x, "x")
    sample_y = as_sample(y, "y")
    n_x = sample_x.size
    n_y = sample_y.size

    # Ranks within each sample and within the pooled sample. The placement
    # r_pooled - r_within is what carries the information about the other group.
    r_x = stats.rankdata(sample_x)
    r_y = stats.rankdata(sample_y)
    r_pooled = stats.rankdata(np.concatenate((sample_x, sample_y)))
    r_pooled_x = r_pooled[:n_x]
    r_pooled_y = r_pooled[n_x:]

    m_x = float(np.mean(r_pooled_x))
    m_y = float(np.mean(r_pooled_y))

    v_x = float(np.sum((r_pooled_x - r_x - m_x + (n_x + 1) / 2) ** 2)) / (n_x - 1)
    v_y = float(np.sum((r_pooled_y - r_y - m_y + (n_y + 1) / 2) ** 2)) / (n_y - 1)

    pooled_var = n_x * v_x + n_y * v_y
    if pooled_var <= 0:
        raise SaValueError(
            "the groups do not overlap, leaving the Brunner-Munzel variance "
            "estimate at zero and the statistic undefined."
        )

    # (m_x - m_y) rather than (m_y - m_x), so a positive statistic means x is the
    # larger sample.
    bm_stat = n_x * n_y * (m_x - m_y) / (n_x + n_y) / math.sqrt(pooled_var)
    df = pooled_var**2 / ((n_x * v_x) ** 2 / (n_x - 1) + (n_y * v_y) ** 2 / (n_y - 1))

    relative_effect = 1 - (m_y - (n_y + 1) / 2) / n_x
    se = math.sqrt(v_x / (n_x * n_y**2) + v_y / (n_y * n_x**2))
    lower, upper = t_ci(relative_effect, se, df, alternative, conf_level, bounds=(0.0, 1.0))

    return {
        "relative_effect": relative_effect,
        "bm_stat": bm_stat,
        "df": df,
        "pval": t_pval(bm_stat, df, alternative),
        "lower_conf": lower,
        "upper_conf": upper,
    }


def yuen_paired(
    x: Any,
    y: Any,
    tr: float = 0.2,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Yuen's trimmed mean test for two dependent samples.

    Port of ``sa_yuen_paired()``. Compares trimmed means using a standard error
    built from the winsorised variances and their covariance, so the pairing is
    kept while outliers in either sample lose their leverage.

    ``robust_dz`` is the trimmed mean difference over a robust estimate of the
    standard deviation of the paired differences, rescaled by
    :func:`winsorized_normal_var` so that it reads on the same scale as Cohen's
    ``dz`` when the differences are normal. ``WRS2::yuend()`` reports an
    explanatory power measure here instead, computed as though the two samples
    were independent and returned unsigned; a signed, pairing-aware quantity is
    more use in a table where every other estimate is signed.

    Args:
        x: First sample of complete pairs, no missing values.
        y: Second sample, the same length.
        tr: Proportion trimmed at each tail, in ``[0, 0.5)``.
        alternative: One of :data:`ALTERNATIVES`, where ``"greater"`` tests
            whether ``x`` exceeds ``y``.
        conf_level: Confidence level of the reported interval.

    Returns:
        ``x_trim_mean``, ``y_trim_mean``, ``trim_diff``, ``stderr``,
        ``yuen_stat``, ``df``, ``pval``, ``lower_conf``, ``upper_conf``,
        ``robust_dz``.

    Raises:
        SaValueError: If the winsorised paired differences have zero variance.

    References:
        Yuen, K. K. (1974). The two-sample trimmed t for unequal population
        variances. *Biometrika*, 61(1), 165-170.

        Algina, J., Keselman, H. J. and Penfield, R. D. (2005). An alternative
        to Cohen's standardized mean difference effect size. *Psychological
        Methods*, 10(3), 317-328.
    """
    sample_x = as_sample(x, "x")
    sample_y = as_sample(y, "y")
    if sample_x.size != sample_y.size:
        raise SaValueError(
            f"`x` and `y` must be complete pairs of the same length, got "
            f"{sample_x.size} and {sample_y.size}."
        )

    n_pairs = sample_x.size
    h = n_pairs - 2 * _trim_count(n_pairs, tr)

    win_x = winsorize(sample_x, tr)
    win_y = winsorize(sample_y, tr)

    # Sums of squared deviations, not variances: the h in the denominator below
    # replaces n, which is the whole point of the trimmed test.
    ss_x = (n_pairs - 1) * float(np.var(win_x, ddof=1))
    ss_y = (n_pairs - 1) * float(np.var(win_y, ddof=1))
    ss_xy = (n_pairs - 1) * float(np.cov(win_x, win_y, ddof=1)[0, 1])

    # ss_x + ss_y - 2 * ss_xy is the sum of squares of (win_x - win_y), so it can
    # only reach zero when the winsorised differences are constant.
    stderr = math.sqrt(max((ss_x + ss_y - 2 * ss_xy) / (h * (h - 1)), 0.0))
    if not math.isfinite(stderr) or stderr <= 0:
        raise SaValueError(
            "the winsorised paired differences have zero variance, leaving the "
            "standard error at zero and the statistic undefined."
        )

    df = float(h - 1)
    x_trim_mean = trimmed_mean(sample_x, tr)
    y_trim_mean = trimmed_mean(sample_y, tr)
    trim_diff = x_trim_mean - y_trim_mean
    yuen_stat = trim_diff / stderr
    lower, upper = t_ci(trim_diff, stderr, df, alternative, conf_level)

    win_diff_var = float(np.var(winsorize(sample_x - sample_y, tr), ddof=1))
    robust_dz = (
        trim_diff / math.sqrt(win_diff_var / winsorized_normal_var(tr))
        if win_diff_var > 0
        else float("nan")
    )

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
        "robust_dz": robust_dz,
    }
