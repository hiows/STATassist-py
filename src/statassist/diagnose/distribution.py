"""Check the assumptions a comparison rests on.

Port of ``R/diagnose_distribution.R``. Runs the normality tests, the homogeneity
of variance tests and the outlier screen together, and reports them as one
object.

A failed assumption never blocks an analysis and never causes a test to be
swapped for another one. It changes which member of the reported test family
deserves the most weight, and that judgement stays with the user.

Each assumption is checked twice on purpose, by tests that fail differently.
Shapiro-Wilk is the more powerful of the two normality tests and is the one to
read first; the Kolmogorov-Smirnov test is fitted against a normal with the
sample's own mean and standard deviation, which makes its p-value
anti-conservative, so it disagreeing with Shapiro-Wilk usually means the
departure is in the tails. The Levene test is centred on the median and tolerates
skew; Bartlett's is more powerful when the groups really are normal and cannot
tell unequal variances apart from heavy tails when they are not, so the two
disagreeing is itself evidence about normality.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaError, SaValueError
from ..core.result import SaDiagnosis, metadata
from ..core.validate import check_feat_names, check_scalar_num
from ..kernel.diagnostic import LEVENE_CENTERS, OUTLIER_CRITERIA, bartlett, ks_normal, levene
from ..kernel.diagnostic import shapiro as shapiro_test
from ..summarize.descriptive import kurtosis, skewness
from .outliers import SCREEN_COLUMNS, screen_outliers, split_for_screening

__all__ = [
    "NORMALITY_COLUMNS",
    "SUMMARY_COLUMNS",
    "VARIANCE_COLUMNS",
    "diagnose_distribution",
    "diagnose_samples",
    "new_diagnosis",
    "normality_table",
    "variance_table",
]

#: Columns of one row of the normality table.
#:
#: The level column is named ``group`` rather than ``level`` so that it lines up
#: with the output of :func:`~statassist.summarize_descriptive_stats`.
NORMALITY_COLUMNS = (
    "features",
    "group",
    "n_used",
    "shapiro_stat",
    "shapiro_pval",
    "ks_stat",
    "ks_pval",
    "skewness",
    "excess_kurtosis",
)

#: Columns of one row of the variance table.
VARIANCE_COLUMNS = (
    "features",
    "n_used",
    "n_groups",
    "levene_stat",
    "levene_df1",
    "levene_df2",
    "levene_pval",
    "bartlett_stat",
    "bartlett_df",
    "bartlett_pval",
)

#: Columns of one row of the summary table.
SUMMARY_COLUMNS = (
    "features",
    "n_levels",
    "n_outliers",
    "min_shapiro_pval",
    "normal_ok",
    "variance_ok",
)


#: Per feature, the samples by level. An ungrouped diagnosis has one sample per
#: feature and no level to name it by, which R writes as ``NA``.
Samples = dict[str, dict[str | None, np.ndarray]]


def normality_table(per_feature: Samples, feats: list[str]) -> pd.DataFrame:
    """Normality and shape, one row per feature and group level.

    Port of ``sa_normality_table()``. A level that cannot be tested, because it
    is too small or constant, yields a row of missing values rather than being
    left out. Its absence would otherwise be indistinguishable from the level not
    existing.

    Args:
        per_feature: Per feature, the samples by level.
        feats: Feature names, fixing the block order.
    """
    rows: list[dict[str, Any]] = []
    for name in feats:
        for level, values in per_feature[name].items():
            normality = _or_missing(shapiro_test, values, ("shapiro_stat", "shapiro_pval"))
            ks = _or_missing(ks_normal, values, ("ks_stat", "ks_pval"))
            rows.append(
                {
                    "features": name,
                    "group": level,
                    "n_used": values.size,
                    **normality,
                    **ks,
                    "skewness": skewness(values) if values.size else float("nan"),
                    "excess_kurtosis": kurtosis(values) if values.size else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=list(NORMALITY_COLUMNS))


def variance_table(
    per_feature: Samples,
    feats: list[str],
    grouped: bool,
    center: str = LEVENE_CENTERS[0],
    trim: float = 0.1,
) -> pd.DataFrame:
    """Homogeneity of variance, one row per feature.

    Port of ``sa_variance_table()``.

    Args:
        per_feature: Per feature, the samples by level.
        feats: Feature names, fixing the row order.
        grouped: Whether more than one sample per feature exists. Without a
            grouping there is nothing to compare variances across, so the table
            is empty rather than full of missing values.
        center: Centre the Levene test measures each deviation from.
        trim: Trimming proportion used when ``center="trimmed"``.
    """
    if not grouped:
        return pd.DataFrame({name: [] for name in VARIANCE_COLUMNS}).astype(
            {name: float for name in VARIANCE_COLUMNS if name != "features"}
        )

    rows: list[dict[str, Any]] = []
    for name in feats:
        samples = per_feature[name]
        rows.append(
            {
                "features": name,
                "n_used": sum(values.size for values in samples.values()),
                "n_groups": len(samples),
                **_or_missing(
                    levene,
                    samples,
                    ("levene_stat", "levene_df1", "levene_df2", "levene_pval"),
                    center,
                    trim,
                ),
                **_or_missing(
                    bartlett,
                    samples,
                    ("bartlett_stat", "bartlett_df", "bartlett_pval"),
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(VARIANCE_COLUMNS))


def new_diagnosis(
    features: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    normality: pd.DataFrame,
    variance: pd.DataFrame,
    outliers: pd.DataFrame,
    alpha: float,
) -> SaDiagnosis:
    """Assemble a diagnosis object.

    Port of ``sa_new_diagnosis()``. ``normal_ok`` and ``variance_ok`` are
    ``p > alpha``: equality is a failure, and a check that could not be run is
    missing rather than passing.
    """
    rows: list[dict[str, Any]] = []
    for name in features:
        of_feature = normality["shapiro_pval"][normality["features"] == name]
        # The worst level decides: a comparison is only as normal as its least
        # normal group, so taking the minimum is what makes the flag mean
        # something.
        worst = float("nan") if of_feature.isna().all() else float(of_feature.min())
        if len(variance.index) == 0:
            variance_pval = float("nan")
        else:
            matched = variance["levene_pval"][variance["features"] == name]
            variance_pval = float(matched.iloc[0]) if len(matched) else float("nan")
        rows.append(
            {
                "features": name,
                "n_levels": int((normality["features"] == name).sum()),
                "n_outliers": int((outliers["features"] == name).sum()),
                "min_shapiro_pval": worst,
                "normal_ok": _flag(worst, alpha),
                "variance_ok": None if len(variance.index) == 0 else _flag(variance_pval, alpha),
            }
        )

    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    summary["normal_ok"] = summary["normal_ok"].astype("boolean")
    summary["variance_ok"] = summary["variance_ok"].astype("boolean")

    return SaDiagnosis(
        {
            "analysis": "distribution_diagnosis",
            "features": features,
            "design": design,
            "parameters": parameters,
            "normality": normality,
            "variance": variance,
            "outliers": outliers,
            "summary": summary,
            "metadata": metadata(),
        }
    )


def diagnose_distribution(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
    alpha: float = 0.05,
    criterion: str = OUTLIER_CRITERIA[0],
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.5,
    center: str = LEVENE_CENTERS[0],
    trim: float = 0.1,
) -> SaDiagnosis:
    """Check the assumptions a comparison rests on.

    Args:
        data: Wide frame (or 2-D array), one row per observation.
        feats: Names of the numeric columns to check.
        group: Optional grouping vector with one entry per row of ``data``. When
            supplied, normality is checked within each level and the variance
            tests are run across them. Without it there is only one sample per
            feature, so the variance table is empty.
        group_lv: Levels to keep, in display order. Defaults to the sorted unique
            values of ``group``.
        alpha: Threshold applied to the p-values when setting the ``normal_ok``
            and ``variance_ok`` flags of the summary table.
        criterion: Outlier rule, passed to
            :func:`~statassist.screen_outliers`.
        iqr_multiplier: Fence width for the IQR rule.
        z_threshold: Cut-off for the robust z rule.
        center: Centre used by the Levene test.
        trim: Trimming proportion of the Levene test when ``center="trimmed"``.

    Returns:
        A :class:`~statassist.core.SaDiagnosis` carrying ``analysis``,
        ``features``, ``design``, ``parameters``, the four tables ``normality``,
        ``variance``, ``outliers`` and ``summary``, and ``metadata``.

    Raises:
        SaValueError: If ``criterion`` or ``center`` is unknown, or if ``alpha``
            or ``trim`` is out of range.

    References:
        Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for
        normality (complete samples). *Biometrika*, 52(3-4), 591-611.

        Brown, M. B. and Forsythe, A. B. (1974). Robust tests for the equality of
        variances. *JASA*, 69(346), 364-367.

    Examples:
        >>> from statassist import simulate_two_groups
        >>> sim = simulate_two_groups(n_feats=3, n_up=1, n_down=1, seed=1)
        >>> d = diagnose_distribution(**{
        ...     name: sim.args[name] for name in ("data", "feats", "group", "group_lv")
        ... })
        >>> list(d)
        ['analysis', 'features', 'design', 'parameters', 'normality', 'variance',
         'outliers', 'summary', 'metadata']

        One normality row per feature and group level, one variance row per
        feature.

        >>> len(d.normality.index), len(d.variance.index)
        (6, 3)

        Without a grouping there is only one sample per feature, so there is
        nothing to compare variances across.

        >>> ungrouped = diagnose_distribution(sim.args["data"], sim.args["feats"])
        >>> len(ungrouped.variance.index)
        0
    """
    if criterion not in OUTLIER_CRITERIA:
        raise SaValueError("`criterion` must be one of: " + ", ".join(OUTLIER_CRITERIA) + ".")
    if center not in LEVENE_CENTERS:
        raise SaValueError("`center` must be one of: " + ", ".join(LEVENE_CENTERS) + ".")
    check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)
    check_scalar_num(trim, "trim", 0, 0.5, upper_open=True)

    split = split_for_screening(data, feats, group, group_lv)
    names = [str(name) for name in check_feat_names(feats)]
    levels = list(split.rows) if split.grouped else [None]

    per_feature = {
        name: {
            level: _finite(split.data[name].to_numpy(dtype=float)[rows])
            for level, rows in zip(levels, split.rows.values(), strict=False)
        }
        for name in names
    }

    outliers = screen_outliers(
        data, feats, group, group_lv, criterion, iqr_multiplier, z_threshold, alpha
    )

    return new_diagnosis(
        features=names,
        design={
            "group_lv": list(split.rows) if split.grouped else None,
            "grouped": split.grouped,
        },
        parameters={
            "alpha": alpha,
            "criterion": criterion,
            "iqr_multiplier": iqr_multiplier,
            "z_threshold": z_threshold,
            "center": center,
            "trim": trim,
        },
        normality=normality_table(per_feature, names),
        variance=variance_table(per_feature, names, split.grouped, center, trim),
        outliers=outliers,
        alpha=alpha,
    )


def diagnose_samples(
    per_feature: dict[str, Any],
    feats: list[str],
    group_lv: list[str],
    paired: bool,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Attach the assumption checks a comparison rests on.

    Port of ``sa_diagnose_samples()``. Called by the comparison scenarios with
    the samples they actually tested, rather than with the original data. That
    matters: a paired design keeps complete cases only, and a diagnosis run on
    the full column would describe a different set of observations than the
    p-value it sits next to.

    Homogeneity of variance across independent groups is not the assumption a
    within-subject test makes; sphericity is, and the repeated measures ANOVA row
    already carries Mauchly's test and both epsilon corrections. So a paired
    design gets an empty variance table.

    Args:
        per_feature: Per feature, the samples by level or, when ``paired``, a
            subjects-by-conditions matrix.
        feats: Feature names.
        group_lv: Group levels.
        paired: Whether ``per_feature`` holds matrices.
        alpha: Threshold for the two summary flags.

    Returns:
        ``normality``, ``variance`` and ``summary``.
    """
    by_feature: Samples = {}
    for name in feats:
        held = per_feature[name]
        if paired:
            values = np.asarray(held, dtype=float)
            by_feature[name] = {level: values[:, index] for index, level in enumerate(group_lv)}
        else:
            by_feature[name] = {
                str(level): np.asarray(sample, dtype=float) for level, sample in held.items()
            }

    normality = normality_table(by_feature, feats)
    variance = variance_table(by_feature, feats, not paired)
    empty = pd.DataFrame({name: [] for name in SCREEN_COLUMNS})
    summary = new_diagnosis(
        features=feats,
        design={"group_lv": group_lv, "grouped": True},
        parameters={"alpha": alpha},
        normality=normality,
        variance=variance,
        outliers=empty,
        alpha=alpha,
    )["summary"]
    return {"normality": normality, "variance": variance, "summary": summary}


def _finite(values: np.ndarray) -> np.ndarray:
    """The observations a test can be run on."""
    usable: np.ndarray = values[np.isfinite(values)]
    return usable


def _or_missing(
    engine: Callable[..., dict[str, float]],
    argument: Any,
    columns: Sequence[str],
    *rest: Any,
) -> dict[str, float]:
    """Run one kernel, or fill its columns with missing values if it refuses.

    R wraps each call in ``tryCatch()`` because a level that is too small or
    constant is a fact about that level, not a reason to abandon the table.
    """
    try:
        return engine(argument, *rest)
    except SaError:
        return dict.fromkeys(columns, float("nan"))


def _flag(pvalue: float, alpha: float) -> bool | None:
    """Whether a check passed. Strictly greater: equality is a failure."""
    if pvalue != pvalue:  # nan
        return None
    return bool(pvalue > alpha)
