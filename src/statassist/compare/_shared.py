"""The t-test family, wired to SciPy with R's reporting conventions.

Not a kernel. ``scipy.stats.ttest_ind``, ``ttest_rel`` and ``ttest_1samp`` are
already ``stats::t.test()``: the statistic, the Welch-Satterthwaite degrees of
freedom, the p-value and the interval agree with R to the last few digits, and
the interval is left open on the side a one-sided alternative does not test in
both. Nothing here recomputes any of that.

Two things are added, because SciPy does not report them.

``stderr`` is not on a SciPy result object at all, and it is a column of the
comparison contract, so it is formed from the same variances the statistic came
from - which is also what makes ``t_stat == mean_diff / stderr`` hold in the
table a reader sees.

R refuses a sample it cannot test. ``t.test()`` stops on "data are essentially
constant" where SciPy returns a missing statistic and carries on, and the
comparison layer turns a refusal into an all-missing row naming the feature.
Raising is therefore how the feature gets reported rather than silently filled.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core.errors import SaValueError
from ..kernel._shared import ALTERNATIVES, as_sample, check_alternative

__all__ = ["t_independent", "t_one_sample", "t_paired"]

#: How R spells the three alternatives against how SciPy spells them.
#:
#: R's spelling is the one the public functions take, since it is what a user of
#: the R package already writes and what the golden fixtures were generated with.
_SCIPY_ALTERNATIVE = dict(zip(ALTERNATIVES, ("two-sided", "less", "greater"), strict=True))

#: Slack in the "essentially constant" test, as R's ``t.test()`` writes it.
_CONSTANT_TOL = 10 * np.finfo(float).eps


def _refuse_constant(stderr: float, scale: float) -> None:
    """Raise where ``t.test()`` stops, so the feature is reported rather than NA.

    R compares the standard error against the size of the estimate rather than
    against zero, so a sample of ten identical large values is refused even
    though its variance is not exactly zero in floating point.
    """
    if stderr < _CONSTANT_TOL * scale:
        raise SaValueError("data are essentially constant.")


def _report(
    result: Any,
    stderr: float,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """The five columns every t-test row of the contract carries."""
    interval = result.confidence_interval(conf_level)
    return {
        "stderr": stderr,
        "t_stat": float(result.statistic),
        "df": float(result.df),
        "pval": float(result.pvalue),
        "lower_conf": float(interval.low),
        "upper_conf": float(interval.high),
    }


def t_independent(
    x: Any,
    y: Any,
    alternative: str = ALTERNATIVES[0],
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Welch's t-test, reported as ``stats::t.test()`` reports it.

    Args:
        x: The sample every difference reads in the direction of.
        y: The reference sample, the one subtracted.
        alternative: One of :data:`~statassist.kernel._shared.ALTERNATIVES`.
        conf_level: Confidence level of the reported interval.

    Returns:
        ``stderr``, ``t_stat``, ``df``, ``pval``, ``lower_conf``, ``upper_conf``.
    """
    check_alternative(alternative)
    first, second = as_sample(x, "x"), as_sample(y, "y")
    if first.size < 2 or second.size < 2:
        raise SaValueError(
            f"needs at least 2 usable observations per group, got {first.size} and {second.size}."
        )

    stderr = math.sqrt(
        float(np.var(first, ddof=1)) / first.size + float(np.var(second, ddof=1)) / second.size
    )
    _refuse_constant(stderr, max(abs(float(np.mean(first))), abs(float(np.mean(second)))))
    result = stats.ttest_ind(
        first,
        second,
        equal_var=False,
        alternative=_SCIPY_ALTERNATIVE[alternative],
    )
    return _report(result, stderr, alternative, conf_level)


def t_paired(
    x: Any,
    y: Any,
    alternative: str = ALTERNATIVES[0],
    conf_level: float = 0.95,
) -> dict[str, float]:
    """Paired t-test, reported as ``stats::t.test(paired = TRUE)`` reports it.

    ``x`` and ``y`` are already matched pair for pair, which is decided where the
    pairing is resolved rather than here.
    """
    check_alternative(alternative)
    first, second = as_sample(x, "x"), as_sample(y, "y")
    if first.size != second.size:
        raise SaValueError(
            f"needs complete pairs, got {first.size} and {second.size} observation(s)."
        )
    if first.size < 2:
        raise SaValueError(f"needs at least 2 complete pairs, got {first.size}.")

    differences = first - second
    stderr = float(np.std(differences, ddof=1)) / math.sqrt(differences.size)
    _refuse_constant(stderr, abs(float(np.mean(differences))))
    result = stats.ttest_rel(first, second, alternative=_SCIPY_ALTERNATIVE[alternative])
    return _report(result, stderr, alternative, conf_level)


def t_one_sample(
    x: Any,
    mu: float = 0.0,
    alternative: str = ALTERNATIVES[0],
    conf_level: float = 0.95,
) -> dict[str, float]:
    """One-sample t-test, reported as ``stats::t.test(x, mu = mu)`` reports it."""
    check_alternative(alternative)
    sample = as_sample(x, "x")
    if sample.size < 2:
        raise SaValueError(f"needs at least 2 usable observations, got {sample.size}.")

    stderr = float(np.std(sample, ddof=1)) / math.sqrt(sample.size)
    _refuse_constant(stderr, abs(float(np.mean(sample))))
    result = stats.ttest_1samp(sample, popmean=mu, alternative=_SCIPY_ALTERNATIVE[alternative])
    return _report(result, stderr, alternative, conf_level)
