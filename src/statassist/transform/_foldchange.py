"""Fold change between the two group levels of a comparison.

Port of ``R/utils_foldchange.R``. Private, because the real consumer of
:func:`fold_change` is the two-group comparison of Phase 3 and the only caller
here is :func:`~statassist.center_by_control`.

This was an exported ``calculate_fold_change()`` taking ``data`` and deriving its
own samples. It now works from the samples the tests were run on, which is what
keeps the two axes of a volcano plot describing the same observations: under
``paired=True`` the tests keep complete pairs only, and a separately computed fold
change would quietly average a different set of rows.

The direction is not an argument either. The caller hands over the two levels
already in ``x``, ``y`` order, the same order the tests read as ``x - y``, so the
x and y axes of a volcano plot cannot end up pointing opposite ways.

``input_scale`` is the one thing here the tests do not share. They run on the
values as supplied, which is the point of logging them, while these centres are
always brought back to the original scale so that a ratio is a ratio. The two
therefore disagree numerically under ``input_scale="log2"``, and that is intended
rather than a leak.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify
from ..core.tables import feature_table, stat_row
from ..core.validate import UNSET

__all__ = ["FC_MEANS", "INPUT_SCALES", "fc_center", "fold_change", "resolve_fc_mean"]

#: The two centres a fold change can be taken between.
FC_MEANS: tuple[str, ...] = ("arith", "geom")

#: The two scales data can arrive on.
INPUT_SCALES: tuple[str, ...] = ("raw", "log2")


def resolve_fc_mean(fc_mean: Any, input_scale: str) -> str:
    """Resolve ``fc_mean`` against the scale the data arrived on.

    Port of ``sa_resolve_fc_mean()``. The default depends on another argument,
    which a formal default cannot express. R detects that with ``missing()``;
    here the caller leaves its own default as :data:`~statassist.core.UNSET` and
    passes it through.

    On the log2 scale the geometric mean is the convention and the only choice
    that reduces to a difference of means, so it wins by default there.

    Args:
        fc_mean: The argument as received, or ``UNSET`` if the user said nothing.
        input_scale: One of :data:`INPUT_SCALES`, already matched.
    """
    if fc_mean is UNSET:
        return "geom" if input_scale == "log2" else "arith"
    if fc_mean not in FC_MEANS:
        raise SaValueError("`fc_mean` must be one of: " + ", ".join(FC_MEANS) + ".")
    return str(fc_mean)


def fc_center(
    v: Any,
    side: str,
    mean_type: str,
    input_scale: str = INPUT_SCALES[0],
) -> float:
    """Central tendency of one group for the fold change ratio.

    Port of ``sa_fc_center()``.

    Args:
        v: Numeric vector with missing values already removed.
        side: Group label, used in the error message.
        mean_type: One of :data:`FC_MEANS`.
        input_scale: One of :data:`INPUT_SCALES`.

    Raises:
        SaValueError: If the sample is empty, if undoing a log2 transformation
            overflows, or if the geometric mean is asked for on a sample that is
            not strictly positive.
    """
    values = np.asarray(v, dtype=float).reshape(-1)
    if values.size == 0:
        raise SaValueError(f"no usable observation left in the {side} group.")

    # A ratio is only a ratio on the original scale. Undoing the transformation
    # here rather than dividing the log2 values keeps every centre downstream on
    # one scale, so `fold_change == x_center / y_center` still holds.
    if input_scale == "log2":
        # R's `2^v` overflows to Inf silently and the count is reported below;
        # NumPy would raise a RuntimeWarning saying the same thing less usefully.
        with np.errstate(over="ignore"):
            values = np.exp2(values)
        n_over = int(np.sum(~np.isfinite(values)))
        if n_over > 0:
            raise SaValueError(
                f"2^x overflows to infinity for {n_over} value(s) in the {side} "
                "group, so these observations are not on the log2 scale; use "
                '`input_scale = "raw"` instead.'
            )

    if mean_type == "arith":
        return float(np.mean(values))

    # Dropping the non-positive values instead would silently return the
    # geometric mean of the positive subset, which is a different quantity.
    n_low = int(np.sum(values <= 0))
    if n_low > 0:
        raise SaValueError(
            f"the geometric mean is undefined for the {n_low} value(s) at or "
            f'below zero in the {side} group; use `fc_mean = "arith"` instead.'
        )
    return float(np.exp(np.mean(np.log(values))))


def fold_change(
    samples: Any,
    feats: Sequence[str],
    group_lv: Sequence[str],
    mean_type: str,
    input_scale: str = INPUT_SCALES[0],
) -> pd.DataFrame:
    """Fold change table for a two-group comparison.

    Port of ``sa_fold_change()``.

    Args:
        samples: Per feature, a mapping with ``x`` and ``y``, already reduced to
            the observations the tests used.
        feats: Feature names, one output row per entry.
        group_lv: The two group levels in ``x``, ``y`` order, the first going in
            the numerator. Note that this is the reverse of the display order the
            user supplies, where the reference comes first.
        mean_type: One of :data:`FC_MEANS`.
        input_scale: One of :data:`INPUT_SCALES`.

    Returns:
        ``features``, ``x_center``, ``y_center``, ``fold_change`` and ``log2fc``.
    """
    names = [str(name) for name in feats]
    levels = [str(level) for level in group_lv]
    label = ("Arithmetic" if mean_type == "arith" else "Geometric") + " mean fold change"

    def one(index: int) -> dict[str, float]:
        held = samples[names[index]]
        x_center = fc_center(held["x"], levels[0], mean_type, input_scale)
        y_center = fc_center(held["y"], levels[1], mean_type, input_scale)
        ratio = x_center / y_center if y_center != 0 else math.copysign(math.inf, x_center)
        return stat_row(
            x_center=x_center,
            y_center=y_center,
            fold_change=ratio,
            log2fc=_log2(ratio),
        )

    out = feature_table(
        names,
        ["x_center", "y_center", "fold_change", "log2fc"],
        label,
        fun=one,
        p_adjust_method=None,
    )

    def report(mask: Any, what: str) -> None:
        hit = list(out["features"][mask.fillna(False)])
        if hit:
            notify(f"Fold change: {what} for {len(hit)} feature(s): " + ", ".join(hit) + ".")

    # A ratio only reads as "n times higher" when both centres are positive, so
    # each way of leaving that domain is called out separately.
    report(
        out["y_center"] == 0,
        f"the {levels[1]} centre is zero, so `fold_change` is infinite",
    )
    report(
        (out["x_center"] == 0) & (out["y_center"] != 0),
        f"the {levels[0]} centre is zero, so `log2fc` is -Inf and clears any cutoff",
    )
    report(
        out["x_center"] * out["y_center"] < 0,
        "the two centres have opposite signs, so `log2fc` is NaN",
    )
    return out


def _log2(ratio: float) -> float:
    """``log2()`` extended over the whole line, as R's is.

    R warns once per call on a negative argument, which says nothing about which
    feature caused it; the features are reported in the aggregated messages
    instead. Here the domain is handled arithmetically so no warning is raised to
    muffle.
    """
    if ratio > 0:
        return math.log2(ratio)
    if ratio == 0:
        return -math.inf
    return math.nan
