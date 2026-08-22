"""Run every applicable two-group test at once.

Port of ``R/compare_two_groups.R``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, notify, warn
from ..core.result import SaComparison, new_comparison
from ..core.tables import feature_table, stat_row
from ..core.validate import (
    UNSET,
    check_flag,
    check_p_adjust,
    check_scalar_num,
    control_first,
    pair_by_id,
    pair_by_order,
    validate_wide_input,
)
from ..diagnose.distribution import diagnose_samples
from ..kernel._shared import ALTERNATIVES, check_alternative
from ..kernel.robust import brunner_munzel, yuen_paired
from ..kernel.wilcox import rank_sum, signed_rank
from ..transform._foldchange import INPUT_SCALES, fold_change, resolve_fc_mean
from ._shared import t_independent, t_paired

__all__ = ["compare_two_groups"]

#: Per-feature samples in the ``x``, ``y`` order every reported quantity reads.
Samples = dict[str, dict[str, np.ndarray]]


def compare_two_groups(
    data: Any,
    feats: Any,
    group: Any,
    group_lv: Any,
    control_label: Any = None,
    id: Any = None,  # noqa: A002 - matches the R argument name
    alternative: str = ALTERNATIVES[0],
    paired: bool = False,
    conf_level: float = 0.95,
    tr: float = 0.2,
    fc_mean: Any = UNSET,
    input_scale: str = INPUT_SCALES[0],
    p_adjust: str = "BH",
    diagnose: bool = True,
) -> SaComparison:
    """Run every applicable two-group test at once.

    Compares exactly two group levels across any number of numeric features and
    returns a parametric, a rank-based and a robust test side by side, together
    with the fold change between the two groups. Nothing is chosen on the
    caller's behalf: reporting all of them makes disagreement between them
    visible, which is the situation where the choice of test actually matters.

    Which member of each family is used depends on ``paired``:

    ==========  =========================  ===============================
    family      ``paired=False``           ``paired=True``
    ==========  =========================  ===============================
    t           Welch's t-test             Paired t-test
    Wilcoxon    Rank-sum (Mann-Whitney U)  Signed-rank
    robust      Brunner-Munzel             Yuen's trimmed mean (dependent)
    ==========  =========================  ===============================

    Direction is set once, by the order of ``group_lv``, and every quantity in
    the result follows it. The first level is the reference, so
    ``alternative="greater"`` tests whether ``group_lv[1]`` exceeds
    ``group_lv[0]`` in all three families, and ``mean_diff``, ``hl_shift``,
    ``trim_diff``, ``fold_change`` and ``relative_effect`` are all above their
    null value when ``group_lv[1]`` is the larger group.

    Args:
        data: Wide frame (or 2-D array), one row per observation and one column
            per feature.
        feats: Names of the numeric columns to test. One output row per entry.
        group: Grouping vector with one entry per row of ``data``.
        group_lv: Exactly two group levels. The first is the reference, treated
            as ``y``, and the second is the level compared against it, treated
            as ``x``, so every difference reads as ``group_lv[1] - group_lv[0]``
            and every ratio as ``group_lv[1] / group_lv[0]``. Rows belonging to
            any other level are dropped.
        control_label: The level to hold as the reference. Naming it moves that
            level to the front of ``group_lv`` and leaves the other where it is,
            so ``control_label=group_lv[1]`` reverses every difference and ratio
            without the pair having to be rewritten. ``None`` keeps the order
            given.
        id: Optional pairing key with one entry per row of ``data``, used only
            when ``paired=True``. Supplying it matches observations by key
            instead of by row order, which is the safer choice whenever the rows
            may have been reordered or a subject may be missing from one group.
            Ids present in only one group are dropped.
        alternative: One of :data:`~statassist.kernel._shared.ALTERNATIVES`.
        paired: Whether observations are matched. See ``id`` for how the pairs
            are formed; without it the pairs are formed by row order, which
            cannot detect rows that have been reordered.
        conf_level: Confidence level for all reported intervals.
        tr: Trimming proportion for Yuen's test, in ``[0, 0.5)``. Ignored when
            ``paired=False``.
        fc_mean: Which centre the fold change divides, ``"arith"`` or
            ``"geom"``. Left unset it is ``"geom"`` when
            ``input_scale="log2"``, where it is the convention, and ``"arith"``
            otherwise.
        input_scale: The scale ``data`` arrives on, ``"raw"`` or ``"log2"``. This
            changes the ``effect`` table only, never the tests: they run on the
            values as supplied, which is the point of logging them, while the
            centres are always brought back to the original scale so that a
            ratio is a ratio.
        p_adjust: Multiplicity adjustment applied across ``feats`` within each
            test table. ``"none"`` disables it.
        diagnose: Whether to attach the normality and homogeneity of variance
            checks the tests rest on, computed on the same observations the
            tests used.

    Returns:
        A :class:`~statassist.core.result.SaComparison`. ``effect`` holds
        ``x_center``, ``y_center``, ``fold_change`` and ``log2fc``; ``tests``
        holds ``t_test``, ``wilcox_test`` and ``robust_test``, each one row per
        feature. There is no ``posthoc`` or ``pairwise`` slot: with two groups
        the omnibus comparison is already the only contrast there is.

    Raises:
        SaValueError: If an argument is unusable, or if the two levels cannot be
            paired.

    Warns:
        SaWarning: If a test could not be run for at least one feature. Those
            rows come back missing and are named together, rather than the scan
            being abandoned.

    Examples:
        A simulator's ``args`` is named after this function's arguments, so the
        analysis is one call away.

        >>> from statassist import simulate_two_groups
        >>> sim = simulate_two_groups(
        ...     n_feats=4, n_case=20, n_control=20, n_up=1, n_down=1, seed=1
        ... )
        >>> res = compare_two_groups(**sim.args)
        >>> list(res.tests)
        ['t_test', 'wilcox_test', 'robust_test']

        Every table has one row per feature, in one order.

        >>> list(res.effect["features"]) == res.features
        True
        >>> list(res.effect.columns)
        ['features', 'x_center', 'y_center', 'fold_change', 'log2fc']

        The planted direction comes back in the sign of ``log2fc``.

        >>> up = sim.truth.loc[sim.truth["direction"] == "up", "features"].iloc[0]
        >>> bool(res.effect.set_index("features").loc[up, "log2fc"] > 0)
        True
    """
    check_alternative(alternative)
    if input_scale not in INPUT_SCALES:
        raise SaValueError("`input_scale` must be one of: " + ", ".join(INPUT_SCALES) + ".")
    mean_type = resolve_fc_mean(fc_mean, input_scale)
    paired = check_flag(paired, "paired")
    diagnose = check_flag(diagnose, "diagnose")
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    tr = check_scalar_num(tr, "tr", 0, 0.5, upper_open=True)
    p_adjust = check_p_adjust(p_adjust, "p_adjust")

    if id is not None and not paired:
        warn(
            "`id` is only used to form pairs and is ignored when "
            "`paired = False`. Set `paired = True` if the observations are "
            "matched."
        )

    validated = validate_wide_input(data, feats, group, group_lv, id=id, n_levels=2)
    frame = validated.data
    names = validated.feats
    if validated.group is None:  # pragma: no cover - n_levels forces a grouping
        raise SaValueError("`group` must be supplied.")
    levels = control_first([str(level) for level in validated.group.categories], control_label)

    if validated.n_dropped > 0:
        notify(f"Dropped {validated.n_dropped} row(s) belonging to a level outside `group_lv`.")

    # `group_lv` is in display order, reference first, while `x` and `y` are the
    # two sides of every difference the tests report. The reference is the one
    # subtracted, so the two orders are reverses of each other.
    lv_xy = list(reversed(levels))
    membership = np.asarray(validated.group.astype(str), dtype=object)

    if not paired:
        idx_x = np.flatnonzero(membership == lv_xy[0])
        idx_y = np.flatnonzero(membership == lv_xy[1])
        unmatched: list[str] = []
    else:
        pairing = (
            pair_by_order(membership, lv_xy)
            if validated.id is None
            else pair_by_id(validated.id, membership, lv_xy)
        )
        idx_x, idx_y, unmatched = pairing
        if unmatched:
            notify(
                f"Dropped {len(unmatched)} id(s) present in only one group: "
                + ", ".join(unmatched)
                + "."
            )

    samples = _samples(frame, names, idx_x, idx_y, paired)
    effect = fold_change(samples, names, lv_xy, mean_type, input_scale)

    t_label = "Paired t-test" if paired else "Welch's t-test"
    t_result = feature_table(
        names,
        [
            "n_x",
            "n_y",
            "n_used",
            "x_mean",
            "y_mean",
            "mean_diff",
            "stderr",
            "t_stat",
            "df",
            "pval",
            "lower_conf",
            "upper_conf",
        ],
        t_label,
        fun=lambda index: _t_row(samples[names[index]], paired, alternative, conf_level),
        p_adjust_method=p_adjust,
    )

    w_label = (
        "Wilcoxon signed-rank test" if paired else "Wilcoxon rank sum test (Mann-Whitney U test)"
    )
    w_result = feature_table(
        names,
        ["n_x", "n_y", "n_used", "hl_shift", "w_stat", "pval", "lower_conf", "upper_conf"],
        w_label,
        fun=lambda index: _wilcox_row(samples[names[index]], paired, alternative, conf_level),
        p_adjust_method=p_adjust,
    )

    robust_label, robust_columns, robust_row = (
        _yuen_stage(alternative, conf_level, tr) if paired else _bm_stage(alternative, conf_level)
    )
    robust_result = feature_table(
        names,
        robust_columns,
        robust_label,
        fun=lambda index: robust_row(samples[names[index]]),
        p_adjust_method=p_adjust,
    )

    return new_comparison(
        analysis="two_group_comparison",
        features=names,
        design={
            "group_lv": levels,
            "paired": paired,
            "pairing": None if not paired else ("order" if validated.id is None else "id"),
            "n_dropped": validated.n_dropped,
            "unmatched_ids": unmatched,
        },
        parameters={
            "alternative": alternative,
            "conf_level": conf_level,
            "tr": tr if paired else float("nan"),
            "fc_mean": mean_type,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
        },
        effect=effect,
        tests={"t_test": t_result, "wilcox_test": w_result, "robust_test": robust_result},
        test_info={
            "t_test": {
                "id": "paired_t_test" if paired else "welch_t_test",
                "label": t_label,
                "paired": paired,
            },
            "wilcox_test": {
                "id": "wilcoxon_signed_rank" if paired else "mann_whitney_u",
                "label": w_label,
                "paired": paired,
            },
            "robust_test": {
                "id": "yuen_paired" if paired else "brunner_munzel",
                "label": robust_label,
                "paired": paired,
            },
        },
        diagnostics=_diagnostics(samples, names, levels) if diagnose else None,
        subclass="sa_two_group",
    )


def _samples(
    frame: pd.DataFrame,
    feats: Sequence[str],
    idx_x: np.ndarray,
    idx_y: np.ndarray,
    paired: bool,
) -> Samples:
    """The observations each test is run on, per feature.

    Taken straight out of the wide columns rather than by reshaping to long
    format and subsetting, which is what let the group column be picked up from
    somewhere other than ``data``.

    Missing values are handled per feature and per design: independent samples
    drop them within each group, a paired design keeps complete pairs only. The
    fold change is computed from these same arrays, so it never rests on a
    different subset of the data than the p-value beside it.
    """
    out: Samples = {}
    for name in feats:
        column = frame[name].to_numpy(dtype=float)
        x, y = column[idx_x], column[idx_y]
        if paired:
            keep = ~np.isnan(x) & ~np.isnan(y)
            out[str(name)] = {"x": x[keep], "y": y[keep]}
        else:
            out[str(name)] = {"x": x[~np.isnan(x)], "y": y[~np.isnan(y)]}
    return out


def _t_row(
    held: dict[str, np.ndarray],
    paired: bool,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """One row of the t-test table.

    The group means are computed here rather than read off the engine: R's
    ``t.test()`` reports two means when independent and a single difference when
    paired, and the table has one column layout for both designs.
    """
    x, y = held["x"], held["y"]
    n_x, n_y = x.size, y.size
    if n_x < 2 or n_y < 2:
        raise SaValueError(f"needs at least 2 usable observations per group, got {n_x} and {n_y}.")
    engine = t_paired if paired else t_independent
    return {
        **stat_row(
            n_x=n_x,
            n_y=n_y,
            n_used=n_x if paired else n_x + n_y,
            x_mean=float(np.mean(x)),
            y_mean=float(np.mean(y)),
            mean_diff=float(np.mean(x)) - float(np.mean(y)),
        ),
        **engine(x, y, alternative=alternative, conf_level=conf_level),
    }


#: The label, the numeric columns and the per-feature body of a robust stage.
RobustStage = tuple[str, list[str], Callable[[dict[str, np.ndarray]], dict[str, float]]]


def _bm_stage(alternative: str, conf_level: float) -> RobustStage:
    """The robust stage of an independent design: Brunner-Munzel."""
    columns = [
        "n_x",
        "n_y",
        "n_used",
        "relative_effect",
        "bm_stat",
        "df",
        "pval",
        "lower_conf",
        "upper_conf",
    ]

    def row(held: dict[str, np.ndarray]) -> dict[str, float]:
        x, y = held["x"], held["y"]
        if x.size < 2 or y.size < 2:
            raise SaValueError(
                f"needs at least 2 usable observations per group, got {x.size} and {y.size}."
            )
        return {
            **stat_row(n_x=x.size, n_y=y.size, n_used=x.size + y.size),
            **brunner_munzel(x, y, alternative=alternative, conf_level=conf_level),
        }

    return "Brunner-Munzel test", columns, row


def _yuen_stage(alternative: str, conf_level: float, tr: float) -> RobustStage:
    """The robust stage of a paired design: Yuen's trimmed mean test."""
    columns = [
        "n_x",
        "n_y",
        "n_used",
        "x_trim_mean",
        "y_trim_mean",
        "trim_diff",
        "stderr",
        "yuen_stat",
        "df",
        "pval",
        "lower_conf",
        "upper_conf",
        "robust_dz",
    ]

    def row(held: dict[str, np.ndarray]) -> dict[str, float]:
        n_pairs = held["x"].size
        # Checked here rather than left to the kernel: how many pairs survive is
        # a fact about this feature, and the message names the proportion the
        # caller passed, which the kernel does not know it was given.
        surviving = n_pairs - 2 * int(tr * n_pairs)
        if surviving < 2:
            raise SaValueError(
                f"only {surviving} observation(s) survive trimming {tr} from each "
                f"tail of {n_pairs} pair(s); 2 are needed."
            )
        return {
            **stat_row(n_x=n_pairs, n_y=n_pairs, n_used=n_pairs),
            **yuen_paired(
                held["x"],
                held["y"],
                tr=tr,
                alternative=alternative,
                conf_level=conf_level,
            ),
        }

    return "Yuen's trimmed mean test for dependent samples", columns, row


def _wilcox_row(
    held: dict[str, np.ndarray],
    paired: bool,
    alternative: str,
    conf_level: float,
) -> dict[str, float]:
    """One row of the Wilcoxon table.

    A paired design tests the within-pair differences against zero, which is
    what ``wilcox.test(x, y, paired = TRUE)`` does, so the signed-rank ``V``
    lands in the same ``w_stat`` column the rank-sum ``W`` does.
    """
    x, y = held["x"], held["y"]
    n_x, n_y = x.size, y.size
    if n_x < 1 or n_y < 1:
        raise SaValueError(f"needs at least 1 usable observation per group, got {n_x} and {n_y}.")
    if paired:
        produced = signed_rank(x - y, alternative=alternative, conf_level=conf_level)
        statistic = produced["v_stat"]
    else:
        produced = rank_sum(x, y, alternative=alternative, conf_level=conf_level)
        statistic = produced["w_stat"]

    return stat_row(
        n_x=n_x,
        n_y=n_y,
        n_used=n_x if paired else n_x + n_y,
        hl_shift=produced["hl_shift"],
        w_stat=statistic,
        pval=produced["pval"],
        lower_conf=produced["lower_conf"],
        upper_conf=produced["upper_conf"],
    )


def _diagnostics(samples: Samples, feats: Sequence[str], group_lv: Sequence[str]) -> Any:
    """The assumption checks, on the observations the tests actually used.

    Built from ``samples`` rather than from the original columns: a paired
    design keeps complete pairs only, and a diagnosis run on the full column
    would describe a different set of observations than the p-value beside it.

    The two samples go back into display order here, so the diagnosis reads
    reference first the way every other per-level table in the package does.
    Independent is passed for a paired design too: the pairing is not what the
    per-level normality of the two samples is about.
    """
    reference, compared = str(group_lv[0]), str(group_lv[1])
    per_feature = {
        str(name): {reference: samples[str(name)]["y"], compared: samples[str(name)]["x"]}
        for name in feats
    }
    names = [str(name) for name in feats]
    return diagnose_samples(per_feature, names, [reference, compared], False)
