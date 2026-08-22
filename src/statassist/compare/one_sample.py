"""Compare one sample against a hypothesised value.

Port of ``R/compare_one_sample.R``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core.errors import SaValueError, notify
from ..core.result import SaComparison, new_comparison
from ..core.tables import feature_table, stat_row
from ..core.validate import (
    UNSET,
    check_feat_names,
    check_flag,
    check_p_adjust,
    check_scalar_num,
)
from ..diagnose.distribution import diagnose_samples
from ..diagnose.outliers import split_for_screening
from ..kernel._shared import ALTERNATIVES, check_alternative
from ..kernel.wilcox import signed_rank
from ..transform._foldchange import INPUT_SCALES, fc_center, resolve_fc_mean
from ._shared import t_one_sample

__all__ = ["compare_one_sample", "one_sample_prop"]

#: What the single sample is called on the level axis every table shares.
#:
#: There is no grouping here, but the diagnosis and the descriptive tables are
#: keyed by level, so the one sample needs a name. R uses this one.
SAMPLE_LEVEL = "sample"

#: Largest number of distinct values a feature may take and still be binary.
BINARY_MAX_LEVELS = 2


def compare_one_sample(
    data: Any,
    feats: Any,
    mu: float = 0.0,
    p: float = 0.5,
    success: float = 1.0,
    alternative: str = ALTERNATIVES[0],
    conf_level: float = 0.95,
    fc_mean: Any = UNSET,
    input_scale: str = INPUT_SCALES[0],
    p_adjust: str = "BH",
    diagnose: bool = True,
) -> SaComparison:
    """Compare one sample against a hypothesised value.

    Tests each feature against ``mu`` and returns a parametric, a rank-based and
    a proportion result side by side, in the shape
    :func:`~statassist.compare_two_groups` uses. There is no second group here,
    so the reference is a number rather than a set of observations.

    ``prop_test`` only applies to a feature that is binary. One that is not comes
    back missing and named in a warning rather than being silently reduced to
    "equals ``success`` or not", which would produce a number that looks like a
    result.

    Args:
        data: Wide frame (or 2-D array), one row per observation.
        feats: Names of the numeric columns to test.
        mu: Hypothesised value for the mean and the pseudo-median, applied to
            every feature and read on the same scale as ``data``.
        p: Hypothesised proportion for ``prop_test``, in ``(0, 1)``.
        success: The value counted as a success when a feature is binary.
        alternative: One of :data:`~statassist.kernel._shared.ALTERNATIVES`.
            ``"greater"`` tests whether the sample exceeds ``mu``, and every
            reported quantity follows that direction.
        conf_level: Confidence level for all reported intervals.
        fc_mean: Which centre the fold change divides ``mu`` into. Left unset it
            is ``"geom"`` when ``input_scale="log2"`` and ``"arith"`` otherwise.
        input_scale: The scale ``data`` and ``mu`` arrive on. On the log2 scale
            both are raised back through ``2 ** x`` before the ratio is taken, so
            ``fold_change`` means what it does for raw input. This changes the
            ``effect`` table only, never the tests.
        p_adjust: Multiplicity adjustment applied across ``feats`` within each
            test table. ``"none"`` disables it.
        diagnose: Whether to attach the normality check the t-test rests on.

    Returns:
        A :class:`~statassist.core.result.SaComparison`. ``design`` carries
        ``mu``, ``p`` and ``success`` instead of ``group_lv``, since there are no
        groups, and ``effect`` holds ``n_used``, ``center``, ``mu``, ``diff``,
        ``fold_change`` and ``log2fc``. There is no ``posthoc`` or ``pairwise``
        slot: a single sample has no pair of levels to contrast.

    Raises:
        SaValueError: If an argument is unusable, or if ``2 ** mu`` overflows on
            the log2 scale.

    Notes:
        ``fold_change`` is ``center / mu``, so both it and ``log2fc`` are
        undefined when ``mu`` is zero - which is also its most common value.
        Both come back missing in that case, a note says so, and
        :func:`~statassist.estimate_significance` calls every feature undecided.
        Reporting an infinity would read as an infinitely large increase when
        what actually happened is that the question has no answer. The case
        cannot arise under ``input_scale="log2"``, where the reference is
        ``2 ** mu`` and so positive whatever ``mu`` is.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"a": [4.1, 5.2, 6.3, 3.8, 7.1], "flag": [1, 0, 1, 1, 1]})
        >>> res = compare_one_sample(data, ["a", "flag"], mu=4, diagnose=False)
        >>> list(res.tests)
        ['t_test', 'wilcox_test', 'prop_test']

        The proportion test reaches the binary feature and not the continuous
        one, whose row is missing rather than coerced.

        >>> prop = res.tests["prop_test"].set_index("features")
        >>> float(prop.loc["flag", "proportion"])
        0.8
        >>> bool(prop.loc["a"].isna().all())
        True
    """
    check_alternative(alternative)
    if input_scale not in INPUT_SCALES:
        raise SaValueError("`input_scale` must be one of: " + ", ".join(INPUT_SCALES) + ".")
    mean_type = resolve_fc_mean(fc_mean, input_scale)
    mu = check_scalar_num(mu, "mu")
    p = check_scalar_num(p, "p", 0, 1, lower_open=True, upper_open=True)
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    p_adjust = check_p_adjust(p_adjust, "p_adjust")
    diagnose = check_flag(diagnose, "diagnose")
    success = check_scalar_num(success, "success")

    # The same validation that guards the grouped comparisons, with no grouping:
    # numeric columns, matching lengths, no duplicate feature names.
    screening = split_for_screening(data, feats, group=None, group_lv=None)
    frame = screening.data
    names = check_feat_names(feats)

    samples = {}
    for name in names:
        column = frame[name].to_numpy(dtype=float)
        samples[name] = column[~np.isnan(column)]

    # The tests take `mu` as supplied, but a ratio needs both sides on the scale
    # the measurement was made on, so the `effect` table compares against this.
    mu_ref = 2.0**mu if input_scale == "log2" else mu
    if not math.isfinite(mu_ref):
        raise SaValueError(
            f"2^`mu` overflows to infinity, so `mu` = {mu} is not on the log2 "
            'scale; use `input_scale = "raw"` instead.'
        )

    effect = feature_table(
        names,
        ["n_used", "center", "mu", "diff", "fold_change", "log2fc"],
        "Fold change against mu",
        fun=lambda index: _effect_row(samples[names[index]], mu_ref, mean_type, input_scale),
        p_adjust_method=None,
    )
    if mu_ref == 0:
        notify(
            "`mu` is 0, so `fold_change` and `log2fc` are undefined and the "
            "`effect` table reports them as NA."
        )

    t_result = feature_table(
        names,
        [
            "n_used",
            "center",
            "mu",
            "diff",
            "stderr",
            "t_stat",
            "df",
            "cohens_d",
            "pval",
            "lower_conf",
            "upper_conf",
        ],
        "One-sample t-test",
        fun=lambda index: _t_row(samples[names[index]], mu, alternative, conf_level),
        p_adjust_method=p_adjust,
    )

    w_result = feature_table(
        names,
        ["n_used", "hl_shift", "v_stat", "pval", "lower_conf", "upper_conf"],
        "One-sample Wilcoxon signed-rank test",
        fun=lambda index: _wilcox_row(samples[names[index]], mu, alternative, conf_level),
        p_adjust_method=p_adjust,
    )

    prop_result = feature_table(
        names,
        [
            "n_used",
            "n_success",
            "proportion",
            "p",
            "diff",
            "chi_sq",
            "df",
            "cohens_h",
            "pval",
            "lower_conf",
            "upper_conf",
        ],
        "One-sample proportion test",
        fun=lambda index: one_sample_prop(
            samples[names[index]], p, success, alternative, conf_level
        ),
        p_adjust_method=p_adjust,
    )

    return new_comparison(
        analysis="one_sample_comparison",
        features=names,
        design={"mu": mu, "p": p, "success": success, "paired": False, "n_dropped": 0},
        parameters={
            "alternative": alternative,
            "conf_level": conf_level,
            "fc_mean": mean_type,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
        },
        effect=effect,
        tests={"t_test": t_result, "wilcox_test": w_result, "prop_test": prop_result},
        test_info={
            "t_test": {
                "id": "one_sample_t_test",
                "label": "One-sample t-test",
                "paired": False,
            },
            "wilcox_test": {
                "id": "one_sample_wilcoxon",
                "label": "One-sample Wilcoxon signed-rank test",
                "paired": False,
            },
            "prop_test": {
                "id": "one_sample_proportion",
                "label": "One-sample proportion test",
                "paired": False,
            },
        },
        diagnostics=(
            diagnose_samples(
                {name: {SAMPLE_LEVEL: samples[name]} for name in names},
                names,
                [SAMPLE_LEVEL],
                False,
            )
            if diagnose
            else None
        ),
        subclass="sa_one_sample",
    )


def _effect_row(
    values: np.ndarray,
    mu_ref: float,
    mean_type: str,
    input_scale: str,
) -> dict[str, float]:
    """One row of the fold change table, against the hypothesised value."""
    if values.size == 0:
        raise SaValueError("no usable observation left.")
    centre = fc_center(values, SAMPLE_LEVEL, mean_type, input_scale)
    ratio = math.nan if mu_ref == 0 else centre / mu_ref
    return stat_row(
        n_used=values.size,
        center=centre,
        mu=mu_ref,
        diff=centre - mu_ref,
        fold_change=ratio,
        log2fc=math.log2(ratio) if ratio > 0 else (-math.inf if ratio == 0 else math.nan),
    )


def _t_row(
    values: np.ndarray,
    mu: float,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """One row of the t-test table."""
    if values.size < 2:
        raise SaValueError(f"needs at least 2 usable observations, got {values.size}.")
    centre = float(np.mean(values))
    spread = float(np.std(values, ddof=1))
    return {
        **stat_row(
            n_used=values.size,
            center=centre,
            mu=mu,
            diff=centre - mu,
        ),
        **t_one_sample(values, mu=mu, alternative=alternative, conf_level=conf_level),
        # Cohen's d against the sample standard deviation, which is undefined
        # rather than infinite when the sample does not vary at all.
        "cohens_d": (centre - mu) / spread if spread > 0 else math.nan,
    }


def _wilcox_row(
    values: np.ndarray,
    mu: float,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """One row of the signed-rank table."""
    if values.size < 1:
        raise SaValueError("needs at least 1 usable observation.")
    produced = signed_rank(values, mu=mu, alternative=alternative, conf_level=conf_level)
    return stat_row(
        n_used=values.size,
        hl_shift=produced["hl_shift"],
        v_stat=produced["v_stat"],
        pval=produced["pval"],
        lower_conf=produced["lower_conf"],
        upper_conf=produced["upper_conf"],
    )


def one_sample_prop(
    v: Any,
    p: float,
    success: float,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """Score test and Wilson interval for one proportion.

    Port of ``sa_one_sample_prop()``, which is ``stats::prop.test()`` on one
    count with the continuity correction R applies by default. SciPy has no
    counterpart: ``binomtest`` is exact and reports a Clopper-Pearson interval,
    which is a different test and a different interval, so the score test is
    written out here.

    The interval is Wilson's rather than Wald's. A Wald interval on a proportion
    near 0 or 1 runs outside ``[0, 1]``, which is not a statement the data can
    make.

    Args:
        v: One sample, missing values already removed.
        p: Hypothesised proportion.
        success: Value counted as a success.
        alternative: One of :data:`~statassist.kernel._shared.ALTERNATIVES`.
        conf_level: Confidence level of the interval.

    Raises:
        SaValueError: If the feature is not binary, or if no observation equals
            ``success``. Both become a missing row and a named warning through
            :func:`~statassist.core.tables.feature_table`.
    """
    values = np.asarray(v, dtype=float).reshape(-1)
    observed = np.unique(values)
    if observed.size > BINARY_MAX_LEVELS:
        raise SaValueError(
            "the proportion test needs a binary feature, but this one takes "
            f"{observed.size} distinct values."
        )
    if not bool(np.isin(success, observed)):
        raise SaValueError(
            f"the value counted as a success, {success:g}, does not occur in this feature."
        )

    n = values.size
    n_success = int(np.sum(values == success))
    proportion = n_success / n

    # R caps the correction at the distance to the null count, so a proportion
    # already at `p` is not corrected past it and the statistic cannot go
    # negative.
    yates = min(0.5, abs(n_success - n * p))
    statistic = ((abs(n_success - n * p) - yates) ** 2) / (n * p) + (
        (abs((n - n_success) - n * (1 - p)) - yates) ** 2
    ) / (n * (1 - p))

    if alternative == "two.sided":
        pval = float(stats.chi2.sf(statistic, 1))
    else:
        z = math.copysign(math.sqrt(statistic), proportion - p)
        pval = float(stats.norm.cdf(z) if alternative == "less" else stats.norm.sf(z))

    lower, upper = _wilson(proportion, n, yates, alternative, conf_level)
    return stat_row(
        n_used=n,
        n_success=n_success,
        proportion=proportion,
        p=p,
        diff=proportion - p,
        chi_sq=statistic,
        # One count against one hypothesised proportion, so the statistic is
        # referred to a chi-square on one degree of freedom.
        df=1,
        # Cohen's h compares proportions on the arcsine scale, where a given
        # difference means the same thing near 0.5 and near the boundaries.
        cohens_h=2 * math.asin(math.sqrt(proportion)) - 2 * math.asin(math.sqrt(p)),
        pval=pval,
        lower_conf=lower,
        upper_conf=upper,
    )


def _wilson(
    proportion: float,
    n: int,
    yates: float,
    alternative: str,
    conf_level: float,
) -> tuple[float, float]:
    """The Wilson score interval, with the same correction the statistic used.

    A one-sided alternative leaves the side it does not test at the boundary of
    the parameter space, 0 or 1, rather than at an infinity: a proportion cannot
    be outside the unit interval, so that is what "open" means here.
    """
    z = float(stats.norm.ppf((1 + conf_level) / 2 if alternative == "two.sided" else conf_level))
    z22n = z**2 / (2 * n)

    def endpoint(centre: float, sign: float) -> float:
        spread = z * math.sqrt(centre * (1 - centre) / n + z22n / (2 * n))
        return (centre + z22n + sign * spread) / (1 + 2 * z22n)

    shifted_up = proportion + yates / n
    upper = 1.0 if shifted_up >= 1 else endpoint(shifted_up, 1.0)
    shifted_down = proportion - yates / n
    lower = 0.0 if shifted_down <= 0 else endpoint(shifted_down, -1.0)

    if alternative == "greater":
        return max(lower, 0.0), 1.0
    if alternative == "less":
        return 0.0, min(upper, 1.0)
    return max(lower, 0.0), min(upper, 1.0)
