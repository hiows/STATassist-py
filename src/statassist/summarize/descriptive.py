"""Descriptive summary of several features, optionally split by group.

Port of ``R/summarize_descriptive_stats.R`` and the four helpers of
``R/utils_describe.R``.

The two shape estimators are written out rather than taken from SciPy.
:func:`scipy.stats.skew` and :func:`scipy.stats.kurtosis` default to the
uncorrected moment ratios, and their ``bias=False`` gives the G1 and G2 that SAS
and SPSS report - the same quantities as ``e1071::skewness(type = 2)``, which is
what R uses here. Written out, the formula is the specification and there is no
default left to drift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.rstats import mad
from ..core.tables import na_row
from ..core.validate import validate_wide_input

__all__ = [
    "describe_columns",
    "describe_vector",
    "kurtosis",
    "skewness",
    "summarize_descriptive_stats",
]

#: Multiplier of the interquartile range that puts the Tukey outlier fences.
#:
#: Fixed by the definition of the fences rather than chosen here, and shared with
#: :func:`~statassist.kernel.diagnostic.flag_outliers`, whose default
#: ``iqr_multiplier`` is the same 1.5 and is an argument there because the screen
#: is allowed to be stricter or looser than the fences a box plot draws.
FENCE_MULTIPLIER = 1.5

#: Observations the two shape estimators need before their bias correction is
#: defined. Below it the correction divides by zero, so the answer is missing.
SKEWNESS_MIN, KURTOSIS_MIN = 3, 4


def describe_columns() -> list[str]:
    """Column layout of one descriptive summary row.

    Port of ``sa_describe_columns()``. Named in one place so that the row builder
    and the all-missing fallback can never disagree about which columns exist or
    in what order.

    >>> describe_columns()[:2]
    ['n', 'n_missing']
    """
    return [
        "n",
        "n_missing",
        "mean",
        "sd",
        "var",
        "se",
        "cv",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "iqr",
        "out_lower_bound",
        "out_upper_bound",
        "mad",
        "skewness",
        "excess_kurtosis",
    ]


def skewness(v: Any) -> float:
    """Sample skewness, G1.

    Port of ``sa_skewness()``. The bias-corrected estimator SAS and SPSS report,
    matching ``e1071::skewness(type = 2)``. It needs at least
    :data:`SKEWNESS_MIN` observations and a non-zero spread; outside that the
    correction divides by zero, so the result is missing rather than a ``nan``
    that would travel on into a summary table.

    References:
        Joanes, D. N. and Gill, C. A. (1998). Comparing measures of sample
        skewness and kurtosis. *JRSS: Series D*, 47(1), 183-189.
    """
    array = np.asarray(v, dtype=float).reshape(-1)
    n = array.size
    centred = array - float(np.mean(array)) if n else array
    m2 = float(np.sum(centred**2)) / n if n else 0.0
    if n < SKEWNESS_MIN or m2 <= 0:
        return float("nan")
    g1 = (float(np.sum(centred**3)) / n) / m2**1.5
    return float(g1 * math.sqrt(n * (n - 1)) / (n - 2))


def kurtosis(v: Any) -> float:
    """Sample excess kurtosis, G2.

    Port of ``sa_kurtosis()``. The counterpart of :func:`skewness`, matching
    ``e1071::kurtosis(type = 2)``. Excess, so a normal sample sits near zero.
    Needs at least :data:`KURTOSIS_MIN` observations.
    """
    array = np.asarray(v, dtype=float).reshape(-1)
    n = array.size
    centred = array - float(np.mean(array)) if n else array
    m2 = float(np.sum(centred**2)) / n if n else 0.0
    if n < KURTOSIS_MIN or m2 <= 0:
        return float("nan")
    g2 = (float(np.sum(centred**4)) / n) / m2**2 - 3
    return ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))


def describe_vector(x: Any) -> dict[str, float]:
    """Describe one numeric vector.

    Port of ``sa_describe_vector()``.

    Args:
        x: Numeric vector, missing and non-finite values included. They are
            counted into ``n_missing`` and left out of every other statistic,
            which keeps a single infinity from turning the whole row into an
            infinity or a ``nan``.

    Returns:
        One value per column of :func:`describe_columns`.
    """
    array = np.asarray(x, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    n = finite.size

    if n == 0:
        row = na_row(describe_columns())
        row["n"] = 0.0
        row["n_missing"] = float(array.size)
        return row

    mean = float(np.mean(finite))
    # R's sd() and var() are the sample statistics; NumPy defaults to ddof=0.
    sd = float(np.std(finite, ddof=1)) if n > 1 else float("nan")
    variance = float(np.var(finite, ddof=1)) if n > 1 else float("nan")
    # stats::quantile()'s default type 7 is numpy's "linear".
    q1, median, q3 = (
        float(value) for value in np.quantile(finite, [0.25, 0.5, 0.75], method="linear")
    )
    iqr = q3 - q1

    return {
        "n": float(n),
        "n_missing": float(array.size - n),
        "mean": mean,
        "sd": sd,
        "var": variance,
        "se": sd / math.sqrt(n),
        "cv": sd / mean,
        "min": float(np.min(finite)),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(np.max(finite)),
        "iqr": iqr,
        "out_lower_bound": q1 - FENCE_MULTIPLIER * iqr,
        "out_upper_bound": q3 + FENCE_MULTIPLIER * iqr,
        "mad": mad(finite),
        "skewness": skewness(finite),
        "excess_kurtosis": kurtosis(finite),
    }


def summarize_descriptive_stats(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
) -> pd.DataFrame:
    """Descriptive summary of several features, optionally split by group.

    Reduces every feature to one row of sample size, central tendency,
    dispersion, quartiles, outlier fences and distribution shape. With a
    grouping vector the same row is produced per group level, so the summary
    lines up with the tests and plots that compare those levels.

    Missing and non-finite values are dropped per feature and per group before
    anything is computed, so one infinity cannot turn a whole row into an
    infinity or a ``nan``; ``n_missing`` records how many were left out. A
    feature with no finite value in a group gives an all-missing row rather than
    aborting the summary.

    ``cv`` is a ratio, so it only reads as relative dispersion when the values
    are positive. On data that crosses zero the mean shrinks towards it and the
    ratio explodes without the spread having changed.

    ``out_lower_bound`` and ``out_upper_bound`` are where the whiskers of a box
    plot may reach, not where they actually end.

    Args:
        data: Wide frame (or 2-D array), one row per observation and one column
            per feature.
        feats: Names of the numeric columns to summarise, in output order.
        group: Optional grouping vector with one entry per row of ``data``. When
            ``None``, all rows are summarised together and no ``group`` column is
            returned.
        group_lv: Group levels to report, in output order. When ``None``, the
            levels present in ``group`` are used: the categories if ``group`` is
            a :class:`pandas.Categorical`, the sorted unique values otherwise.
            Rows belonging to any other level are dropped.

    Returns:
        One row per feature, or per feature and group level when ``group`` is
        supplied. The leading columns are ``features`` and, when grouped,
        ``group``; the rest are :func:`describe_columns`. Features vary slowest,
        so the levels of one feature stay together.

    Raises:
        SaValueError: If ``data`` is neither a frame nor a 2-D array, if a
            feature is missing or not numeric, or if ``feats`` is empty.

    References:
        Joanes, D. N. and Gill, C. A. (1998). Comparing measures of sample
        skewness and kurtosis. *JRSS: Series D*, 47(1), 183-189.
    """
    grouped = group is not None

    # validate_wide_input always works in terms of levels, so an ungrouped call
    # is served by a single synthetic level that is dropped again on the way out.
    if not grouped:
        frame = _as_frame(data)
        group = ["all"] * len(frame.index)
        group_lv = ["all"]
        data = frame
    elif group_lv is None:
        group_lv = _levels_present(group)

    validated = validate_wide_input(data, feats, group, group_lv, min_levels=1)
    if validated.group is None:  # pragma: no cover - a level was always supplied
        raise SaValueError("`group` and `group_lv` must both be supplied or both be `None`.")
    frame = validated.data
    names = validated.feats
    membership = np.asarray(validated.group)
    levels = [str(level) for level in validated.group.categories]

    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    rows = [
        describe_vector(frame[name].to_numpy()[membership == level])
        for name in names
        for level in levels
    ]

    leading: dict[str, list[str]] = {
        "features": [name for name in names for _ in levels],
    }
    if grouped:
        leading["group"] = [level for _ in names for level in levels]

    return pd.concat(
        [
            pd.DataFrame(leading),
            pd.DataFrame(rows, columns=describe_columns(), dtype=float),
        ],
        axis=1,
    )


def _as_frame(data: Any) -> pd.DataFrame:
    """Read an ungrouped ``data`` before a synthetic level is attached to it."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return pd.DataFrame(data)
    raise SaValueError("`data` must be a data.frame or a matrix.")


def _levels_present(group: Any) -> list[str]:
    """The levels of a grouping vector, the way R reads them off a factor.

    A :class:`pandas.Categorical` keeps its own order, minus any category no row
    uses, which is ``levels(droplevels(group))``. Anything else is sorted, which
    is ``sort(unique(as.character(group)))``.
    """
    if isinstance(group, pd.Categorical | pd.Series) and isinstance(
        getattr(group, "dtype", None), pd.CategoricalDtype
    ):
        categorical = group if isinstance(group, pd.Categorical) else group.cat
        used = set(pd.Series(np.asarray(group)).dropna().astype(str))
        return [str(level) for level in categorical.categories if str(level) in used]
    values = pd.Series(np.asarray(group, dtype=object)).dropna().astype(str)
    return sorted(values.unique())
