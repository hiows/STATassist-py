"""Reduce a contingency table to significance verdicts.

Port of ``R/estimate_categorical_significance.R``.

The categorical counterpart of :func:`~statassist.estimate_significance`, and the
reason it is a second function rather than a branch of that one is that the two
axes a verdict is made of sit at a different granularity here. A comparison asks
its question once per feature; a contingency table is asked about as a whole, and
the place where a signed effect and a p-value exist side by side is one level
down, at the cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.contingency import finite_or_na
from ..core.contracts import assoc_scale
from ..core.errors import SaValueError, warn
from ..core.padjust import p_adjust as adjust
from ..core.result import (
    SaCategorical,
    SaCategoricalSignificance,
    SaComparison,
    new_categorical_significance,
)
from ..core.validate import UNSET, check_p_adjust, check_pvalues, check_scalar_num

# The rule that combines a magnitude with a p-value is one rule, so the three
# helpers the numeric verdict is built from are shared rather than written again:
# a cell of a table and a feature of a scan are called significant the same way,
# and undecided the same way.
from .significance import _at_least, _at_most, _kleene

__all__ = ["CATEGORICAL_READINGS", "estimate_categorical_significance"]

#: The two axes a categorical verdict can be read along, in the order R lists them.
#:
#: ``"cell"`` is the reading that generalises - ``observed / expected`` is defined
#: on a table of any shape - so it is the default.
CATEGORICAL_READINGS: tuple[str, ...] = ("cell", "table")

#: Smallest ``abs(log2_lift)`` a cell needs to be called significant.
#:
#: A cell holding twice, or half, what its null expected, and deliberately the
#: same number :func:`~statassist.estimate_significance` uses. It is a strict
#: demand of a contingency table, where departures well under a doubling are
#: ordinary.
DEFAULT_LOG2_LIFT_CUTOFF = 1.0

#: Largest p-value allowed for a verdict of significant.
DEFAULT_PVAL_CUTOFF = 0.05

#: Multiplicity adjustment across the cells of one table.
DEFAULT_ADJ_TYPE = "BH"

#: Read the measure a table reading reports off the design rather than naming one.
AUTO_MEASURE = "auto"

#: The measure each design defines, for ``measure="auto"``.
#:
#: Which measures exist at all depends on the design and on the size of the
#: table, so the null the result was tested against is what decides, with the
#: shape breaking the tie for an independent one.
_AUTO_BY_NULL = {
    "symmetry": "odds_ratio_paired",
    "marginal_homogeneity": "kendalls_w",
}

#: The measure an independent design defines, by whether the table is 2 x 2.
_AUTO_INDEPENDENT = ("odds_ratio", "cramers_v")

#: Rows and columns a table must have for an odds ratio to exist.
_TWO_BY_TWO = (2, 2)


def estimate_categorical_significance(
    categorical_comparison_result: Any,
    by: str = CATEGORICAL_READINGS[0],
    test: Any = UNSET,
    log2_lift_cutoff: Any = UNSET,
    pval_cutoff: float = DEFAULT_PVAL_CUTOFF,
    adj_type: Any = UNSET,
    measure: Any = UNSET,
    effect_cutoff: Any = UNSET,
) -> SaCategoricalSignificance:
    """Reduce a contingency table to significance verdicts.

    :func:`~statassist.compare_categorical_groups` says that an association is
    not signed, which is true of the **table**: Cramer's V reports how far the
    table sits from its null and not in which direction, because past a 2 x 2
    there is no single direction to name. A **cell** is different. It was expected
    at some count and observed at another, and ``observed / expected`` says both
    how far it moved and which way.

    That ratio is ``lift``, the same quantity
    :func:`~statassist.simulate_categorical_groups` plants and reports in
    ``truth_cell["lift"]``, and ``log2(lift)`` is the effect axis of this
    function. It is defined on every table, 2 x 2, 2 x 3 or larger, so one cutoff
    means the same thing whatever shape the table is. The p-value axis is
    ``std_residual``, which is built to be referred to a standard normal.

    The two axes are not the same information. ``lift`` is a ratio and does not
    change when the table is observed on twice as many rows; ``std_residual``
    grows with the square root of the count. So a cell can be far from what was
    expected and poorly evidenced, or close to it and firmly established, exactly
    as ``log2fc`` and a p-value come apart in the numeric scenarios.

    A cell axis also restores a multiplicity axis: the table has as many cells as
    it has, they are one family, and ``adj_type`` adjusts across them. That is
    why this function computes an adjustment rather than reusing one - a
    categorical result carries no adjusted column, there being nothing to adjust
    across at the level its tests are reported at.

    Args:
        categorical_comparison_result: A categorical comparison result, as
            :func:`~statassist.compare_categorical_groups` returns. A numeric
            comparison is refused and pointed at
            :func:`~statassist.estimate_significance`.
        by: ``"cell"`` for one verdict per cell of the table, ``"table"`` for one
            verdict for the table as a whole. Each reading ignores the arguments
            the other one reads, and says so rather than letting a setting that
            changes nothing pass unremarked.
        test: Which test supplies the p-value of a table reading, one of
            ``categorical_comparison_result.tests``. Left unset it is the first
            test the design ran. Read only by ``by="table"``: a cell reading takes
            its p-value from the cell's own standardized residual, which no test
            reports.
        log2_lift_cutoff: Minimum ``abs(log2_lift)`` required to call a cell
            significant. Read only by ``by="cell"``, and
            :data:`DEFAULT_LOG2_LIFT_CUTOFF` when unset.
        pval_cutoff: Largest p-value allowed for a verdict of significant. Read
            by both readings, against ``adj_pvalue`` under ``by="cell"`` and
            against the test's own ``pvalue`` under ``by="table"``.
        adj_type: Multiplicity adjustment across the cells, one of
            :data:`~statassist.core.padjust.P_ADJUST_METHODS`. Read only by
            ``by="cell"``, and :data:`DEFAULT_ADJ_TYPE` when unset. Unlike the
            ``adj_type`` of :func:`~statassist.estimate_significance` this is the
            first adjustment rather than a replacement for one. ``"none"`` tests
            the raw p-values.
        measure: Which row of ``association`` a table reading puts on its effect
            axis, or :data:`AUTO_MEASURE` to take the one the design defines.
            Read only by ``by="table"``.
        effect_cutoff: Magnitude ``measure`` has to reach for a table reading to
            call the table significant. Left unset the verdict is the p-value
            alone: the conventional thresholds for Cramer's V are conventions
            rather than facts about the measure, and a default is the one place a
            convention is hardest to notice. Naming a number reads it on the
            measure's own scale, so an odds ratio cutoff is a fold either way and
            has to be at least 1.

    Returns:
        A :class:`~statassist.core.result.SaCategoricalSignificance`. Under
        ``by="cell"`` its ``significance`` holds one row per cell with
        ``row_level``, ``col_level``, ``observed``, ``expected``, ``lift``,
        ``log2_lift``, ``std_residual``, ``pvalue``, ``adj_pvalue`` and
        ``is_signif``; under ``by="table"``, one row with ``measure``,
        ``estimate``, ``lower_conf``, ``upper_conf``, ``pvalue`` and
        ``is_signif``. The cutoffs, the reading, the null hypothesis and
        whichever of the test name and the adjustment applies are in the table's
        ``attrs``, so a table that has been passed around still says which rule
        produced it.

        A cell table keys on ``row_level`` and ``col_level``, the key
        ``categorical_comparison_result.cells`` and
        ``simulate_categorical_groups().truth_cell`` also use, so a verdict
        merges with either without renaming.

    Raises:
        SaValueError: If an argument is unusable, or if a cell reading is asked
            of a result tested for symmetry.

    Notes:
        ``is_signif`` is three-valued and follows the rule the numeric scenarios
        use: undecided rather than decided against, which is what a missing
        ``std_residual`` or an undefined ``lift`` leaves a cell as.

        A cell holding no observation at all has a ``lift`` of exactly zero and
        so a ``log2_lift`` of ``-inf``, an infinitely large shortfall, which
        clears any magnitude cutoff. A cell whose expected count is zero has no
        ratio to take and is missing in both columns. The two are different
        findings and are reported differently.

        A matched pair of conditions has no cell reading. It is tested for
        symmetry, and the variance correction the standardized residual divides
        by is derived for a table held against its own margins, so
        ``cells["std_residual"]`` is missing throughout such a result: there is no
        p-value axis to read and the request is refused rather than answered with
        a different quantity. Three or more matched conditions are a different
        case, their null being marginal homogeneity, so the cell reading works
        there.

    Examples:
        >>> import pandas as pd
        >>> from statassist import compare_categorical_groups
        >>> smoking = pd.DataFrame(
        ...     {
        ...         "smoker": ["y"] * 60 + ["n"] * 60,
        ...         "grade": (
        ...             ["high"] * 10 + ["mid"] * 20 + ["low"] * 30
        ...             + ["high"] * 30 + ["mid"] * 20 + ["low"] * 10
        ...         ),
        ...     }
        ... )
        >>> res = compare_categorical_groups(smoking)
        >>> sig = estimate_categorical_significance(res)
        >>> list(sig.significance.columns)
        ['row_level', 'col_level', 'observed', 'expected', 'lift', 'log2_lift',
         'std_residual', 'pvalue', 'adj_pvalue', 'is_signif']

        ``lift`` is a ratio, so the cell holding half of what independence
        expected sits at 0.5.

        >>> float(sig.significance["lift"].min())
        0.5

        The whole-table verdict instead. A 2 x 3 table has no odds ratio, so the
        measure ``"auto"`` reports is Cramer's V.

        >>> table = estimate_categorical_significance(res, by="table")
        >>> str(table.significance["measure"].iloc[0])
        'cramers_v'
    """
    if by not in CATEGORICAL_READINGS:
        raise SaValueError("`by` must be one of: " + ", ".join(CATEGORICAL_READINGS) + ".")

    # Before anything reaches for a slot, so that the wrong object is told what
    # it is rather than failing somewhere inside on a slot it does not have.
    res = categorical_comparison_result
    if not isinstance(res, SaCategorical):
        if isinstance(res, SaComparison):
            raise SaValueError(
                "`categorical_comparison_result` is a numeric comparison result. "
                "estimate_significance() is what reads one; this function reads a "
                "contingency table."
            )
        raise SaValueError(
            "`categorical_comparison_result` must be a categorical comparison "
            "result, as returned by compare_categorical_groups()."
        )

    lift_cutoff = check_scalar_num(
        DEFAULT_LOG2_LIFT_CUTOFF if log2_lift_cutoff is UNSET else log2_lift_cutoff,
        "log2_lift_cutoff",
        0,
    )
    pval_cutoff = check_scalar_num(pval_cutoff, "pval_cutoff", 0, 1, lower_open=True)
    adjustment = check_p_adjust(DEFAULT_ADJ_TYPE if adj_type is UNSET else adj_type, "adj_type")

    _warn_unread_args(by, test, measure, effect_cutoff, log2_lift_cutoff, adj_type)

    if by == "cell":
        if res["design"]["null"] == "symmetry":
            raise SaValueError(
                '`by = "cell"` needs a p-value per cell, and this result was '
                "tested for symmetry, where `cells['std_residual']` is missing "
                "throughout: the variance correction it divides by is derived for "
                "a table held against its own margins and has no counterpart "
                'there. Use `by = "table"`, whose verdict reads McNemar\'s '
                "p-value and the paired odds ratio."
            )
        out = _cell_table(res, adjustment, lift_cutoff, pval_cutoff)
        out.attrs.update(_verdict_attrs(res, by, pval_cutoff))
        out.attrs.update({"log2_lift_cutoff": lift_cutoff, "adj_type": adjustment})
        return new_categorical_significance(res["analysis"], out)

    named_test = next(iter(res["tests"])) if test is UNSET else test
    table = _pick_test(res, named_test)
    named_measure = _resolve_measure(res, AUTO_MEASURE if measure is UNSET else measure)
    cutoff = None if effect_cutoff is UNSET else _check_effect_cutoff(effect_cutoff, named_measure)

    out = _table_row(res, table, named_measure, cutoff, pval_cutoff)
    out.attrs.update(_verdict_attrs(res, by, pval_cutoff))
    out.attrs.update(
        {
            "test": named_test,
            "test_label": res["test_info"][named_test]["label"],
            "measure": named_measure,
            "effect_cutoff": cutoff,
        }
    )
    return new_categorical_significance(res["analysis"], out)


def _cell_table(
    res: SaCategorical,
    adj_type: str,
    log2_lift_cutoff: float,
    pval_cutoff: float,
) -> pd.DataFrame:
    """One verdict per cell of the table.

    Port of ``sa_categorical_significance_cells()``. ``lift`` and the
    standardized residual are two readings of the same departure, and they are
    kept apart because they answer different questions: the first is a ratio and
    does not move with the sample size, the second is the departure divided by
    what its own sampling variability would have been.
    """
    cells = res["cells"]
    observed = np.asarray(cells["observed"], dtype=float)
    expected = np.asarray(cells["expected"], dtype=float)
    std_residual = np.asarray(cells["std_residual"], dtype=float)

    # An expected count of zero leaves no ratio to take, which is not a lift of
    # zero: `finite_or_na` is what keeps the two apart. A lift of exactly zero
    # survives it, and its log2 is -inf, an infinitely large shortfall.
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = finite_or_na(observed / expected)
        log2_lift = np.log2(lift)

    # The standardized residual is built to be referred to a standard normal, so
    # this is the cell's own two-sided p-value and not an approximation of one.
    pvalue = 2 * np.asarray(stats.norm.cdf(-np.abs(std_residual)), dtype=float)
    check_pvalues(pvalue)

    # The cells of one table are one family. This is the first adjustment rather
    # than a replacement for one, since a categorical result carries no adjusted
    # column: at the level its tests are reported at there is nothing to adjust
    # across.
    adj_pvalue = adjust(pvalue, adj_type)

    out = pd.DataFrame(
        {
            "row_level": [str(level) for level in cells["row_level"]],
            "col_level": [str(level) for level in cells["col_level"]],
            "observed": observed,
            "expected": expected,
            "lift": lift,
            "log2_lift": log2_lift,
            "std_residual": std_residual,
            "pvalue": pvalue,
            "adj_pvalue": adj_pvalue,
        }
    )
    out["is_signif"] = _at_least(np.abs(log2_lift), log2_lift_cutoff) & _at_most(
        adj_pvalue, pval_cutoff
    )
    return out


def _table_row(
    res: SaCategorical,
    table: pd.DataFrame,
    measure: str,
    effect_cutoff: float | None,
    pval_cutoff: float,
) -> pd.DataFrame:
    """The one-row verdict a table reading produces.

    Port of ``sa_categorical_significance_row()``.
    """
    association = res["association"]
    row = association.loc[association["measure"].astype(str) == measure]
    pvalue = np.asarray([float(table["pval"].iloc[0])], dtype=float)
    check_pvalues(pvalue)

    out = pd.DataFrame(
        {
            "measure": [measure],
            "estimate": [float(row["estimate"].iloc[0])],
            "lower_conf": [float(row["lower_conf"].iloc[0])],
            "upper_conf": [float(row["upper_conf"].iloc[0])],
            "pvalue": pvalue,
        }
    )
    out["is_signif"] = _assoc_clears(measure, out["estimate"].iloc[0], effect_cutoff) & _at_most(
        pvalue, pval_cutoff
    )
    return out


def _resolve_measure(res: SaCategorical, measure: Any) -> str:
    """The measure a design defines, or the one that was named.

    Port of ``sa_resolve_measure()``.
    """
    if not isinstance(measure, str) or measure == "":
        raise SaValueError('`measure` must be a single measure name, or "auto".')

    if measure == AUTO_MEASURE:
        null = res["design"]["null"]
        if null in _AUTO_BY_NULL:
            return _AUTO_BY_NULL[null]
        dims = tuple(int(size) for size in res["design"]["dim"])
        return _AUTO_INDEPENDENT[0] if dims == _TWO_BY_TWO else _AUTO_INDEPENDENT[1]

    known = [str(name) for name in res["association"]["measure"]]
    if measure not in known:
        raise SaValueError(
            "`measure` must name one of the measures this design defines: "
            + ", ".join(known)
            + f". Got {measure}. Which measures exist depends on the design and "
            "on the size of the table, so a measure absent here is one this "
            "result has no value for rather than one that was left out."
        )
    return measure


def _assoc_clears(measure: str, estimate: float, cutoff: float | None) -> pd.Series:
    """Whether an estimate reaches the cutoff on its own scale.

    Port of ``sa_assoc_clears()``. Three-valued, as R's comparison is: an
    estimate the design could not form leaves the table undecided rather than
    decided against.
    """
    if cutoff is None:
        return pd.Series([True], dtype="boolean")

    values = np.asarray([estimate], dtype=float)
    if assoc_scale(measure) == "ratio":
        decided = (values >= cutoff) | (values <= 1 / cutoff)
    else:
        decided = np.abs(values) >= cutoff
    return _kleene(decided, values)


def _check_effect_cutoff(cutoff: Any, measure: str) -> float:
    """Refuse a cutoff that cannot mean what it says on this measure.

    Port of ``sa_check_effect_cutoff()``. A ratio cutoff below 1 admits every
    table rather than a stricter set of them, since ``estimate >= c`` or
    ``estimate <= 1 / c`` covers the whole line once ``c`` drops under 1. That is
    a silently empty demand, so it is an error instead.
    """
    value = check_scalar_num(cutoff, "effect_cutoff", 0, lower_open=True)
    if assoc_scale(measure) == "ratio" and value < 1:
        raise SaValueError(
            f"`effect_cutoff` is read on the scale of `{measure}`, a ratio centred "
            "at 1, so it is a fold either way and has to be at least 1. A cutoff "
            f"of {value:g} asks for an estimate above {value:g} or below "
            f"{1 / value:g}, which every value meets."
        )
    return value


def _pick_test(res: SaCategorical, test: Any) -> pd.DataFrame:
    """The test table a table reading names.

    Port of ``sa_pick_categorical_test()``.
    :func:`~statassist.core.result.pick_test` is the counterpart for the
    comparison scenarios and requires a comparison, which this result
    deliberately is not.
    """
    if not isinstance(test, str):
        raise SaValueError("`test` must be a single test name.")
    if test not in res["tests"]:
        raise SaValueError(
            "`test` must name one of the tests in `categorical_comparison_result`: "
            + ", ".join(res["tests"])
            + f". Got {test}."
        )
    table: pd.DataFrame = res["tests"][test]
    return table


def _warn_unread_args(
    by: str,
    test: Any,
    measure: Any,
    effect_cutoff: Any,
    log2_lift_cutoff: Any,
    adj_type: Any,
) -> None:
    """Say so when a setting the chosen reading does not read was supplied.

    Port of ``sa_warn_unread_args()``. The two readings take their p-value from
    different places, so about half of the arguments do nothing under either one.
    A setting that changes nothing is worth a sentence rather than silence, which
    is the choice :func:`~statassist.compare_categorical_groups` makes about
    ``exact`` and ``simulate_p_value``.
    """
    candidates: tuple[tuple[str, Any], ...]
    if by == "cell":
        candidates = (("test", test), ("measure", measure), ("effect_cutoff", effect_cutoff))
        detail = (
            "A cell reading takes its p-value from the cell's own standardized "
            "residual, which no test and no association measure reports."
        )
    else:
        candidates = (("log2_lift_cutoff", log2_lift_cutoff), ("adj_type", adj_type))
        detail = (
            "A table reading has one p-value and no family to adjust across, and "
            "its effect axis is `measure` rather than a lift."
        )

    supplied = [name for name, value in candidates if value is not UNSET]
    if not supplied:
        return

    plural = len(supplied) > 1
    warn(
        " and ".join(f"`{name}`" for name in supplied)
        + (" are" if plural else " is")
        + f' not read by `by = "{by}"` and '
        + ("were" if plural else "was")
        + f" ignored. {detail}"
    )


def _verdict_attrs(res: SaCategorical, by: str, pval_cutoff: float) -> dict[str, Any]:
    """The attributes both readings describe themselves with.

    Port of ``sa_categorical_significance_attrs()``. ``table_dim`` rather than
    ``dim``, which on a frame is the number of rows and columns of the verdict
    itself and cannot be borrowed for anything else.
    """
    return {
        "analysis": res["analysis"],
        "null": res["design"]["null"],
        "by": by,
        "table_dim": list(res["design"]["dim"]),
        "pval_cutoff": pval_cutoff,
    }
