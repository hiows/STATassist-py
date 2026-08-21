"""Centre every feature on the control group.

Port of ``R/center_by_control.R``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify, warn
from ..core.validate import UNSET, control_first, validate_wide_input
from ._foldchange import INPUT_SCALES, fc_center, resolve_fc_mean

__all__ = ["center_by_control", "control_baseline"]


def control_baseline(
    v: Any,
    control: str,
    mean_type: str,
    input_scale: str,
) -> float:
    """The control centre, on the scale the data arrived on.

    Port of ``sa_control_baseline()``. :func:`~statassist.transform._foldchange.fc_center`
    always reports on the raw scale, since that is the only scale on which a ratio
    is a ratio, so a log2 input needs its centre brought back before it can be
    subtracted from log2 values.

    The non-positive raw centre is rejected rather than reported. The fold change
    table only messages about one, because a ratio against a zero or negative
    centre is a strange number in a table the reader can see. Here it would divide
    the feature itself: a zero centre sends every value to infinity, and a
    negative one reverses the order of all of them, which silently flips the
    direction of every rank-based test run on the result afterwards.

    Args:
        v: Control group values for one feature, missing values included.
        control: The control level name, used in the error messages.
        mean_type: One of :data:`~statassist.transform._foldchange.FC_MEANS`.
        input_scale: One of :data:`~statassist.transform._foldchange.INPUT_SCALES`.

    Returns:
        The quantity to divide out (raw) or to subtract (log2).

    Raises:
        SaValueError: If the centre cannot be taken, or can be taken but cannot
            be applied.
    """
    values = np.asarray(v, dtype=float).reshape(-1)
    # `fc_center` takes the missing values as already gone, the way the
    # comparisons hand it their samples.
    centre = fc_center(values[~np.isnan(values)], control, mean_type, input_scale)

    if input_scale == "log2":
        baseline = math.log2(centre) if centre > 0 else -math.inf if centre == 0 else math.nan
        if not math.isfinite(baseline):
            raise SaValueError(
                f"the {control} centre is {_as_r_number(centre)} on the raw scale, "
                "which has no log2 to subtract."
            )
        return baseline

    if not math.isfinite(centre) or centre <= 0:
        if not math.isfinite(centre):
            consequence = "send every value to zero"
        elif centre == 0:
            consequence = "send every value to infinity"
        else:
            consequence = (
                "reverse the order of every value, which would flip the direction "
                "of the rank-based tests run on the result"
            )
        raise SaValueError(
            f"the {control} centre is {_as_r_number(centre)}, and dividing by it "
            f"would {consequence}. Pass logged values with "
            '`input_scale = "log2"` instead, or restrict `feats` to features '
            "whose control group is positive."
        )
    return centre


def center_by_control(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    control_label: Any = None,
    fc_mean: Any = UNSET,
    input_scale: str = INPUT_SCALES[0],
) -> pd.DataFrame:
    """Centre every feature on the control group.

    Divides, or on the log2 scale subtracts, the centre of the control group out
    of each feature, so that the control lands on 1 (raw) or 0 (log2) and every
    other observation reads as its distance from the control rather than as a
    measurement of its own. The ``data`` that comes back is the ``data`` that went
    in with the ``feats`` columns replaced, which is what lets the same call be
    handed straight to a comparison.

    The arguments are the ones the comparisons already take, in the same order and
    with the same defaults, because the point of the function is that one set of
    arguments describes both steps::

        centred = center_by_control(data, feats, group, group_lv, input_scale="log2")
        compare_two_groups(centred, feats, group, group_lv, input_scale="log2")

    ``fc_mean`` is the comparison's own argument rather than a centring method of
    this function's, and it is resolved the same way: the geometric mean by
    default on the log2 scale, where it is the convention, and the arithmetic mean
    otherwise. Passing the same value to both is what keeps the baseline removed
    here identical to the centre the comparison divides in its ``effect`` table.

    Two or more levels are accepted, so the same call serves a two-group and a
    multi-group design. Only the control level takes part in the baseline; the
    rest are centred on it.

    What a comparison reports afterwards mostly does not move, because the
    transformation is one constant per feature applied to every row of it.
    ``fold_change`` and ``log2fc`` survive it, since both group centres are
    divided by the same baseline. So do the p-values in every test family: on the
    log2 scale the baseline is subtracted and the tests are shift invariant, and
    on the raw scale it is divided, which leaves the t statistic and every rank
    untouched. What does change is the reference centre, which becomes 1, and with
    it the other centre in the row reads as the fold change itself.

    Args:
        data: Wide frame (or 2-D array), one row per observation.
        feats: Names of the numeric columns to centre. Columns not named here are
            returned untouched.
        group: Grouping vector with one entry per row of ``data``.
        group_lv: At least two group levels. Unlike in a comparison, rows
            belonging to another level are **kept**: they are centred on the same
            baseline and take no part in computing it, which leaves the result the
            same length as the ``group`` the comparison still has to be given. The
            comparison drops them itself.
        control_label: The level whose centre is divided out. Defaults to the
            first element of ``group_lv``, the reference the order already asked
            for, so naming it is only needed to point the baseline at a level that
            is not first.
        fc_mean: Which centre of the control group is removed, ``"arith"`` for the
            arithmetic mean or ``"geom"`` for the geometric mean. The geometric
            mean requires strictly positive values. Left unset it is ``"geom"``
            when ``input_scale="log2"`` and ``"arith"`` otherwise, exactly as in
            the comparisons.
        input_scale: The scale ``data`` arrives on, ``"raw"`` or ``"log2"``. This
            decides the operation: a ratio on the raw scale, a difference on the
            log2 scale. The centred data stays on the scale it arrived on, so the
            same ``input_scale`` is what the comparison should be given afterwards.

    Returns:
        The ``data`` that was passed in, as a frame, with the ``feats`` columns
        replaced by their control-relative values. Row count, row order, the index
        and every column that is not a feature are left as they arrived. A feature
        whose baseline could not be taken comes back as an all-missing column and
        is named in a warning.

    Raises:
        SaValueError: If an argument is unusable, in the same order and with the
            same messages a comparison would use.

    Warns:
        SaWarning: If the baseline could not be taken for at least one feature.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame(
        ...     {"a": [2.0, 4.0, 6.0, 12.0], "g": ["ctrl", "ctrl", "case", "case"]}
        ... )
        >>> centred = center_by_control(data, "a", data["g"], ["ctrl", "case"])
        >>> list(centred["a"])
        [0.6666666666666666, 1.3333333333333333, 2.0, 4.0]

        The control lands on 1, so the case centre reads as the fold change
        itself.

        >>> float(centred.loc[data["g"] == "ctrl", "a"].mean())
        1.0
        >>> float(centred.loc[data["g"] == "case", "a"].mean())
        3.0

        A column that is not a feature comes back untouched, which is what lets
        the result be handed straight to a comparison.

        >>> list(centred.columns)
        ['a', 'g']
    """
    if input_scale not in INPUT_SCALES:
        raise SaValueError("`input_scale` must be one of: " + ", ".join(INPUT_SCALES) + ".")
    mean_type = resolve_fc_mean(fc_mean, input_scale)

    # Every check a comparison makes on these arguments, made here in the same
    # order and with the same messages. Only the errors and `n_dropped` are taken
    # from the result: its `data` has the rows outside `group_lv` removed, and
    # returning that would leave the output shorter than the `group` vector the
    # caller still has to hand to the comparison.
    validated = validate_wide_input(data, feats, group, group_lv)
    if validated.group is None:  # pragma: no cover - a grouping is required here
        raise SaValueError("`group` must be supplied.")
    levels = [str(level) for level in validated.group.categories]
    control = control_first(levels, control_label)[0]

    out = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data))
    if validated.n_dropped > 0:
        notify(
            f"Kept {validated.n_dropped} row(s) belonging to a level outside "
            "`group_lv`. They are centred on the same baseline but take no part "
            "in it, and the comparison drops them itself."
        )

    # Positions rather than a boolean mask. A row outside `group_lv` compares a
    # missing value against the control name, and a missing index would put a
    # missing value into the baseline sample and take the whole feature down with
    # it.
    membership = (
        pd.Series(group)
        .astype(object)
        .map(lambda value: None if value is None or value != value else str(value))
    )
    control_rows = np.flatnonzero((membership == control).to_numpy())

    failures: dict[str, str] = {}
    for name in validated.feats:
        column = out[name].to_numpy(dtype=float)
        try:
            baseline = control_baseline(column[control_rows], control, mean_type, input_scale)
        except SaValueError as error:
            failures[name] = str(error)
            baseline = float("nan")
        out[name] = column - baseline if input_scale == "log2" else column / baseline

    # One warning naming every feature, rather than one per feature: a scan over
    # hundreds of columns must not be abandoned because one of them has no usable
    # control group.
    if failures:
        detail = "\n".join(f"  {name}: {reason}" for name, reason in failures.items())
        warn(
            f"The control baseline could not be taken for {len(failures)} of "
            f"{len(validated.feats)} feature(s); those columns are all NA:\n{detail}"
        )
    return out


def _as_r_number(value: float) -> str:
    """A number the way R's ``paste()`` writes it into a message."""
    if not math.isfinite(value):
        return {math.inf: "Inf", -math.inf: "-Inf"}.get(value, "NaN")
    if value == int(value):
        return str(int(value))
    return f"{value:.15g}"
