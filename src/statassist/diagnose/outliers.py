"""Flag candidate outliers without removing them.

Port of ``R/screen_outliers.R``. Nothing is deleted and no analysis changes as a
result. Which observations belong in a data set is a decision about the
experiment rather than about the arithmetic, and the package does not make it on
the user's behalf.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.validate import check_feat_names, check_scalar_num, validate_wide_input
from ..kernel.diagnostic import OUTLIER_CRITERIA, flag_outliers

__all__ = ["SCREEN_COLUMNS", "Screening", "screen_outliers", "split_for_screening"]

#: Columns of the table one flagged observation becomes.
SCREEN_COLUMNS = ("features", "group", "row", "value", "score")

#: What an ungrouped screen calls its single sample, internally. R names the list
#: element ``all`` and writes ``NA`` into the ``group`` column, and so does this.
UNGROUPED = "all"


class Screening(NamedTuple):
    """Which rows to screen, one entry per group level or one for everything.

    Attributes:
        data: The validated frame, with rows outside ``group_lv`` already gone.
        rows: Per level, the positions into ``data`` that belong to it.
        row_id: For each position in ``data``, the row of the frame the caller
            passed in. **Zero-based**, where R's is one-based.
        grouped: Whether a grouping was supplied.
    """

    data: pd.DataFrame
    rows: dict[str, np.ndarray]
    row_id: np.ndarray
    grouped: bool


def split_for_screening(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
) -> Screening:
    """Row indices to screen, one entry per group level or one for everything.

    Port of ``sa_split_for_screening()``. Shared by :func:`screen_outliers` and
    :func:`~statassist.diagnose_distribution` so that an ungrouped call and a
    grouped one differ in exactly one place.

    Args:
        data: Wide frame (or 2-D array), one row per observation.
        feats: Names of the numeric columns to screen.
        group: Optional grouping vector, one entry per row of ``data``.
        group_lv: Levels to keep, in display order. Defaults to the sorted unique
            values of ``group``.

    Raises:
        SaValueError: If ``data`` is neither a frame nor a 2-D array, if it has
            no rows, or if a feature is missing or not numeric.
    """
    if group is None:
        frame = _as_frame(data)
        if len(frame.index) == 0:
            raise SaValueError("`data` has zero rows.")
        names = check_feat_names(feats)
        unknown = [name for name in names if name not in frame.columns]
        if unknown:
            raise SaValueError("`feats` not found in `data`: " + ", ".join(unknown))
        non_numeric = [
            name
            for name in names
            if not pd.api.types.is_numeric_dtype(frame[name]) or frame[name].dtype == bool
        ]
        if non_numeric:
            raise SaValueError(
                "`feats` must refer to numeric columns. Not numeric: " + ", ".join(non_numeric)
            )
        positions = np.arange(len(frame.index))
        return Screening(
            data=frame.reset_index(drop=True),
            rows={UNGROUPED: positions},
            row_id=positions,
            grouped=False,
        )

    if group_lv is None:
        group_lv = sorted({str(value) for value in pd.Series(group).dropna()})

    # min_levels = 1 because a single level is a legitimate thing to screen; it is
    # only the comparison functions that need two or more.
    validated = validate_wide_input(data, feats, group, group_lv, min_levels=1)
    if validated.group is None:  # pragma: no cover - a group was supplied
        raise SaValueError("`group` and `group_lv` must both be supplied or both be `None`.")
    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    membership = np.asarray(validated.group)
    levels = [str(level) for level in validated.group.categories]
    rows = {level: np.flatnonzero(membership == level) for level in levels}

    # Rows outside `group_lv` were dropped, so a position in the filtered data no
    # longer matches the row the caller can look up. `row_id` translates back.
    as_character = pd.Series(group).astype(object).map(lambda value: _as_level(value))
    kept = np.flatnonzero(as_character.isin(levels).to_numpy())
    return Screening(data=validated.data, rows=rows, row_id=kept, grouped=True)


def screen_outliers(
    data: Any,
    feats: Any,
    group: Any = None,
    group_lv: Any = None,
    criterion: str = OUTLIER_CRITERIA[0],
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Flag candidate outliers without removing them.

    Screens each feature, or each feature within each group level, and returns
    one row per flagged observation.

    Three rules are available and they do not agree with each other, which is the
    point of naming the one used. ``"iqr"`` flags anything past
    ``q1 - k * iqr`` or ``q3 + k * iqr``, the rule behind the whiskers of a box
    plot, and makes no distributional assumption. ``"robust_z"`` measures the
    distance from the median in units of the median absolute deviation, using
    those rather than the mean and standard deviation because one extreme value
    inflates the standard deviation enough to hide itself. ``"grubbs"`` tests
    only the single most extreme observation, and is the only rule that produces
    a p-value and the only one that assumes a distribution.

    Args:
        data: Wide frame (or 2-D array), one row per observation.
        feats: Names of the numeric columns to screen.
        group: Optional grouping vector with one entry per row of ``data``. When
            supplied, each feature is screened within each level separately,
            which is what keeps a genuine group difference from being read as a
            set of outliers.
        group_lv: Levels to keep, in display order. Defaults to the sorted unique
            values of ``group``.
        criterion: One of :data:`~statassist.kernel.diagnostic.OUTLIER_CRITERIA`.
        iqr_multiplier: Fence width for ``criterion="iqr"``.
        z_threshold: Cut-off for ``criterion="robust_z"``. The 3.5 default is the
            Iglewicz and Hoaglin recommendation.
        alpha: Significance level for ``criterion="grubbs"``.

    Returns:
        One row per flagged observation, with the columns
        :data:`SCREEN_COLUMNS`. ``row`` is the row of the ``data`` that was
        passed in, **zero-based** where R's is one-based, and ``group`` is
        missing when no grouping was supplied. ``score`` is the quantity the rule
        thresholded. Zero rows means nothing was flagged.

        The criterion and its three thresholds are attached to
        :attr:`~pandas.DataFrame.attrs`, which is where R attaches them as
        attributes of the data.frame.

    Raises:
        SaValueError: If ``criterion`` is unknown or a threshold is out of range.

    References:
        Iglewicz, B. and Hoaglin, D. C. (1993). *How to Detect and Handle
        Outliers*. ASQC Quality Press.

        Grubbs, F. E. (1969). Procedures for detecting outlying observations in
        samples. *Technometrics*, 11(1), 1-21.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 2.5, 1.5, 40.0]})
        >>> screen_outliers(data, "a")[["features", "row", "value"]]
          features  row  value
        0        a    5   40.0

        Within a group, so a genuine group difference is not read as a set of
        outliers. Nothing is flagged once the two levels are screened apart.

        >>> data["g"] = ["x", "x", "x", "y", "y", "y"]
        >>> len(screen_outliers(data, "a", data["g"]).index)
        0

        The three rules disagree, which is the point of naming the one used.

        >>> len(screen_outliers(data, "a", criterion="grubbs").index)
        1
    """
    if criterion not in OUTLIER_CRITERIA:
        raise SaValueError("`criterion` must be one of: " + ", ".join(OUTLIER_CRITERIA) + ".")
    check_scalar_num(iqr_multiplier, "iqr_multiplier", 0)
    check_scalar_num(z_threshold, "z_threshold", 0, lower_open=True)
    check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)

    split = split_for_screening(data, feats, group, group_lv)
    names = [str(name) for name in check_feat_names(feats)]

    found: list[dict[str, Any]] = []
    for name in names:
        column = split.data[name].to_numpy(dtype=float)
        for level, rows in split.rows.items():
            values = column[rows]
            result = flag_outliers(values, criterion, iqr_multiplier, z_threshold, alpha)
            for at in np.flatnonzero(result["flag"]):
                found.append(
                    {
                        "features": name,
                        "group": level if split.grouped else None,
                        "row": int(split.row_id[rows[at]]),
                        "value": float(values[at]),
                        "score": float(result["score"][at]),
                    }
                )

    out = pd.DataFrame(found, columns=list(SCREEN_COLUMNS))
    out = out.astype({"row": "int64", "value": float, "score": float})
    out.attrs.update(
        criterion=criterion,
        iqr_multiplier=iqr_multiplier,
        z_threshold=z_threshold,
        alpha=alpha,
    )
    return out


def _as_level(value: Any) -> Any:
    """``as.character()`` on one grouping value, keeping missing missing."""
    return None if value is None or value != value else str(value)


def _as_frame(data: Any) -> pd.DataFrame:
    """Read an ungrouped ``data`` the way R's ``as.data.frame()`` accepts it."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return pd.DataFrame(data)
    raise SaValueError("`data` must be a data.frame or a matrix.")
