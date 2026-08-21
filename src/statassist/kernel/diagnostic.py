"""Assumption-check kernels.

Port of ``R/kernel_diagnostic.R``. Two callers use them:
:func:`~statassist.diagnose_distribution`, where they are the analysis, and the
comparison scenarios, where they fill the ``diagnostics`` slot so that an
assumption a test rests on is never silently ignored.

A failed check never changes what gets run. It changes which member of the
reported test family deserves the most weight, and that judgement stays with the
user.

Three of R's defaults are easy to lose in translation and are pinned here.

``stats::quantile()`` defaults to type 7, which is
``numpy.quantile(method="linear")``. NumPy's default happens to be the same one,
and it is written out anyway so that a future change of default cannot move the
outlier fences.

``stats::mad()`` scales by 1.4826 so that it estimates a standard deviation for
normal data. :func:`scipy.stats.median_abs_deviation` defaults to a scale of 1 and
its ``scale="normal"`` is the unrounded ``1 / qnorm(0.75)``, which is off from R's
literal by 1.5e-6 relative - a hundred times the golden tolerance, and visible on
the fixtures. :func:`~statassist.core.mad` carries R's constant instead.

``stats::ks.test()`` picks its p-value by a rule rather than a constant: exact
below 100 observations and with no ties, asymptotic otherwise. SciPy has the same
two methods under ``method=``, and the branch is reproduced rather than left to
either library's ``"auto"``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core.errors import SaValueError
from ..core.rstats import mad
from ._shared import as_sample, as_samples
from .anova import oneway_anova
from .robust import trimmed_mean

__all__ = [
    "LEVENE_CENTERS",
    "OUTLIER_CRITERIA",
    "bartlett",
    "flag_outliers",
    "grubbs",
    "ks_normal",
    "levene",
    "shapiro",
]

#: Where the Levene test may take each group's centre from.
LEVENE_CENTERS: tuple[str, ...] = ("median", "mean", "trimmed")

#: The three screening rules, which do not agree with each other. That is the
#: point of naming the one used rather than assuming it.
OUTLIER_CRITERIA: tuple[str, ...] = ("iqr", "robust_z", "grubbs")

#: The sample sizes ``shapiro.test()`` will accept. Checked here so the message
#: names the sample size rather than reporting a failure from inside the engine.
SHAPIRO_MIN, SHAPIRO_MAX = 3, 5000

#: Above this many observations R's Kolmogorov-Smirnov test leaves the exact
#: p-value for the asymptotic one, and so does this.
KS_EXACT_MAX = 100

#: Fewer usable observations than this and no screening rule runs at all: three
#: points cannot say which of them is out of place.
MIN_SCREENED = 3


def shapiro(v: Any) -> dict[str, float]:
    """Shapiro-Wilk normality test.

    Port of ``sa_shapiro()``. R and SciPy implement the same Royston AS R94
    algorithm, so the statistic agrees closely; the p-values can part company in
    the far tail, where the two use different polynomial approximations.

    Args:
        v: Numeric vector without missing values, 3 to 5000 long.

    Returns:
        ``shapiro_stat``, ``shapiro_pval``.

    Raises:
        SaValueError: If the sample is outside the size the test accepts, or if
            it is constant. R's engine refuses a constant sample outright, where
            :func:`scipy.stats.shapiro` returns a statistic of 1 and a p-value of
            1 with a warning. Reported as perfect normality, that would be a
            confident answer to a question the sample cannot answer, so the
            refusal is reinstated here.

    References:
        Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for
        normality (complete samples). *Biometrika*, 52(3-4), 591-611.
    """
    sample = as_sample(v)
    n = sample.size
    if n < SHAPIRO_MIN or n > SHAPIRO_MAX:
        raise SaValueError(
            f"Shapiro-Wilk needs between {SHAPIRO_MIN} and {SHAPIRO_MAX} observations, got {n}."
        )
    if np.ptp(sample) == 0:
        raise SaValueError("all `v` values are identical.")
    result = stats.shapiro(sample)
    return {
        "shapiro_stat": float(result.statistic),
        "shapiro_pval": float(result.pvalue),
    }


def ks_normal(v: Any) -> dict[str, float]:
    """Kolmogorov-Smirnov goodness-of-fit test against a fitted normal.

    Port of ``sa_ks_normal()``. The reference distribution is the normal with the
    sample's own mean and standard deviation. Estimating the parameters from the
    same data the test judges makes the p-value anti-conservative: it is too
    large, so the test rejects normality less often than its nominal level says.
    That is a real limitation of the test and the reason both normality checks
    are reported together rather than one of them alone.

    The exact-versus-asymptotic branch is R's: exact below
    :data:`KS_EXACT_MAX` observations and with no ties. R warns about the ties
    and the warning says nothing the caller can act on here, so it is not passed
    on; choosing the method explicitly is what replaces it.

    Args:
        v: Numeric vector without missing values, at least 2 long.

    Returns:
        ``ks_stat``, ``ks_pval``.

    Raises:
        SaValueError: If the sample is shorter than 2 or has no spread.
    """
    sample = as_sample(v)
    n = sample.size
    if n < 2:
        raise SaValueError(f"the Kolmogorov-Smirnov test needs at least 2 observations, got {n}.")
    spread = float(np.std(sample, ddof=1))
    if not math.isfinite(spread) or spread <= 0:
        raise SaValueError(
            "the sample is constant, so no normal reference distribution can be fitted to it."
        )

    has_ties = np.unique(sample).size < n
    method = "exact" if n < KS_EXACT_MAX and not has_ties else "asymp"
    result = stats.ks_1samp(
        sample,
        stats.norm.cdf,
        args=(float(np.mean(sample)), spread),
        method=method,
    )
    return {"ks_stat": float(result.statistic), "ks_pval": float(result.pvalue)}


def levene(samples: Any, center: str = "median", trim: float = 0.1) -> dict[str, float]:
    """Levene test for homogeneity of variance.

    Port of ``sa_levene()``. A one-way ANOVA on how far each observation sits from
    its own group centre. The default centre is the median, which is the
    Brown-Forsythe variant: it keeps the test honest when the groups are skewed,
    whereas centring on the mean makes the test itself sensitive to the
    non-normality it is meant to tolerate.

    The ANOVA is :func:`~statassist.kernel.anova.oneway_anova` rather than a
    second implementation, which is what keeps a Levene statistic and the F
    statistic of the same data on one code path.

    Args:
        samples: One sample per group level, no missing values.
        center: One of :data:`LEVENE_CENTERS`.
        trim: Trimming proportion used when ``center="trimmed"``.

    Returns:
        ``levene_stat``, ``levene_df1``, ``levene_df2``, ``levene_pval``.

    Raises:
        SaValueError: If ``center`` is not one of :data:`LEVENE_CENTERS`, or if
            the deviations leave the ANOVA nothing to work with.

    References:
        Brown, M. B. and Forsythe, A. B. (1974). Robust tests for the equality of
        variances. *JASA*, 69(346), 364-367.
    """
    if center not in LEVENE_CENTERS:
        raise SaValueError("`center` must be one of: " + ", ".join(LEVENE_CENTERS) + ".")

    names, arrays = as_samples(samples)

    def centre_of(array: np.ndarray) -> float:
        if center == "median":
            return float(np.median(array))
        if center == "mean":
            return float(np.mean(array))
        return trimmed_mean(array, trim)

    deviations = {
        name: np.abs(array - centre_of(array)) for name, array in zip(names, arrays, strict=False)
    }
    result = oneway_anova(deviations)

    return {
        "levene_stat": result["f_stat"],
        "levene_df1": result["df1"],
        "levene_df2": result["df2"],
        "levene_pval": result["pval"],
    }


def bartlett(samples: Any) -> dict[str, float]:
    """Bartlett test for homogeneity of variance.

    Port of ``sa_bartlett()``. More powerful than the Levene test when the groups
    really are normal, and misleading when they are not: it cannot tell unequal
    variances apart from heavy tails. Read it next to the Levene result rather
    than instead of it.

    The statistic is written out rather than taken from
    :func:`scipy.stats.bartlett`, which reports no degrees of freedom, so the
    ``bartlett_df`` column would have had to be recomputed anyway.

    Args:
        samples: One sample per group level, no missing values, at least 2 each.

    Returns:
        ``bartlett_stat``, ``bartlett_df``, ``bartlett_pval``.

    Raises:
        SaValueError: If a group holds fewer than 2 observations, or if one has
            no variance at all, which leaves the log undefined.

    References:
        Bartlett, M. S. (1937). Properties of sufficiency and statistical tests.
        *Proceedings of the Royal Society A*, 160(901), 268-282.
    """
    names, arrays = as_samples(samples)
    k = len(arrays)
    df_each = np.array([array.size - 1 for array in arrays], dtype=float)
    if (df_each <= 0).any():
        raise SaValueError("there must be at least 2 observations in each group.")

    variances = np.array([float(np.var(array, ddof=1)) for array in arrays])
    if (variances <= 0).any():
        flat = ", ".join(name for name, var in zip(names, variances, strict=False) if var <= 0)
        raise SaValueError(
            f"group(s) with zero variance leave the Bartlett statistic undefined: {flat}."
        )

    df_total = float(df_each.sum())
    pooled = float(np.sum(df_each * variances)) / df_total
    statistic = (df_total * math.log(pooled) - float(np.sum(df_each * np.log(variances)))) / (
        1 + (float(np.sum(1 / df_each)) - 1 / df_total) / (3 * (k - 1))
    )
    df = k - 1

    return {
        "bartlett_stat": statistic,
        "bartlett_df": float(df),
        "bartlett_pval": float(stats.chi2.sf(statistic, df)),
    }


def grubbs(v: Any) -> dict[str, float]:
    """Grubbs test for a single outlier.

    Port of ``sa_grubbs()``. Tests whether the observation furthest from the mean
    is further out than a normal sample of that size would produce. It assumes
    the rest of the sample is normal and it looks at one observation, so it is
    the weakest of the three screening rules and the only one that produces a
    p-value.

    Args:
        v: Numeric vector without missing values, at least 3 long.

    Returns:
        ``grubbs_stat``, ``grubbs_pval`` and ``grubbs_index``, the position of
        the most extreme observation. The index is **zero-based**, where R's is
        one-based.

    Raises:
        SaValueError: If the sample is shorter than 3 or constant.

    References:
        Grubbs, F. E. (1969). Procedures for detecting outlying observations in
        samples. *Technometrics*, 11(1), 1-21.
    """
    sample = as_sample(v)
    n = sample.size
    if n < 3:
        raise SaValueError(f"the Grubbs test needs at least 3 observations, got {n}.")
    spread = float(np.std(sample, ddof=1))
    if not math.isfinite(spread) or spread <= 0:
        raise SaValueError("the sample is constant, so no observation can be called extreme.")

    distance = np.abs(sample - float(np.mean(sample)))
    index = int(np.argmax(distance))
    g = float(distance[index]) / spread

    # Inverting the Grubbs statistic gives a t on n - 2 degrees of freedom. The
    # factor n is the Bonferroni correction for having looked at whichever of the
    # n observations turned out to be furthest out, so the product can exceed 1
    # and is capped.
    denominator = (n - 1) ** 2 - n * g**2
    if denominator <= 0:
        pval = 0.0
    else:
        t_stat = math.sqrt(n * (n - 2) * g**2 / denominator)
        pval = min(1.0, float(n * 2 * stats.t.cdf(-t_stat, n - 2)))

    return {"grubbs_stat": g, "grubbs_pval": pval, "grubbs_index": float(index)}


def flag_outliers(
    v: Any,
    criterion: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Flag outlying observations in one numeric vector.

    Port of ``sa_flag_outliers()``. Returns a flag per observation rather than a
    cleaned vector. Which observations to keep is a decision about the experiment,
    not about the arithmetic, so the package never makes it.

    Args:
        v: Numeric vector. Missing and infinite values are never flagged and
            score as missing.
        criterion: One of :data:`OUTLIER_CRITERIA`.
        iqr_multiplier: Fence width for ``criterion="iqr"``.
        z_threshold: Cut-off for ``criterion="robust_z"``.
        alpha: Significance level for ``criterion="grubbs"``.

    Returns:
        ``flag``, a boolean array the length of ``v``, and ``score``, the numeric
        quantity the rule thresholded.

    Raises:
        SaValueError: If ``criterion`` is not one of :data:`OUTLIER_CRITERIA`.
            R reaches this only after the short-sample shortcut, so a bad
            criterion on a sample of two returns quietly there; the argument is
            checked first here.

    References:
        Iglewicz, B. and Hoaglin, D. C. (1993). *How to Detect and Handle
        Outliers*.
    """
    if criterion not in OUTLIER_CRITERIA:
        raise SaValueError("`criterion` must be one of: " + ", ".join(OUTLIER_CRITERIA) + ".")

    array = np.asarray(v, dtype=float).reshape(-1)
    usable = np.isfinite(array)
    flag = np.zeros(array.size, dtype=bool)
    score = np.full(array.size, np.nan)
    clean = array[usable]
    if clean.size < MIN_SCREENED:
        return {"flag": flag, "score": score}

    if criterion == "iqr":
        q1, q3 = np.quantile(clean, [0.25, 0.75], method="linear")
        iqr = float(q3 - q1)
        if iqr > 0:
            # How far past the nearer quartile the value sits, measured in IQR
            # units, so the score itself does not depend on `iqr_multiplier` and
            # the two can be compared across calls that used different fences.
            # Values inside the box score negative.
            per_value = np.maximum(q1 - clean, clean - q3) / iqr
            score[usable] = per_value
            flag[usable] = per_value > iqr_multiplier
    elif criterion == "robust_z":
        # The median and MAD are used instead of the mean and SD because a single
        # extreme value inflates the SD enough to hide itself.
        spread = mad(clean)
        if spread > 0:
            per_value = np.abs(clean - float(np.median(clean))) / spread
            score[usable] = per_value
            flag[usable] = per_value > z_threshold
    else:
        try:
            result = grubbs(clean)
        except SaValueError:
            return {"flag": flag, "score": score}
        at = np.flatnonzero(usable)[int(result["grubbs_index"])]
        score[at] = result["grubbs_stat"]
        if result["grubbs_pval"] <= alpha:
            flag[at] = True

    return {"flag": flag, "score": score}
