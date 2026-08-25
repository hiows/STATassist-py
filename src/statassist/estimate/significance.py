"""Reduce a comparison to one significance verdict per feature.

Port of ``R/estimate_significance.R``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError
from ..core.padjust import p_adjust as adjust
from ..core.result import SaSignificance, new_significance, pick_test
from ..core.validate import check_p_adjust, check_pvalues, check_scalar_num

__all__ = ["READINGS", "estimate_significance"]

#: The three axes a verdict can be read along, in the order R lists them.
READINGS: tuple[str, ...] = ("omnibus", "contrast", "term")


def estimate_significance(
    comparison_result: Any,
    test: str | None = None,
    log2fc_cutoff: float = 1,
    pval_cutoff: float = 0.05,
    adj_type: str | None = None,
    by: str = READINGS[0],
) -> SaSignificance:
    """Reduce a comparison to one significance verdict per feature.

    Puts the two axes of a volcano plot side by side and flags the features that
    clear both cutoffs. The effect size comes from the comparison's ``effect``
    table and the p-value from whichever test is named, so a single comparison
    can be read out through the parametric, the rank-based or the robust lens
    without recomputing anything.

    Args:
        comparison_result: A comparison result, as
            :func:`~statassist.compare_two_groups`,
            :func:`~statassist.compare_multiple_groups` or
            :func:`~statassist.compare_one_sample` returns.
        test: Which test supplies the p-values, one of
            ``comparison_result.tests``. ``None`` takes the first test the
            scenario ran, which is the parametric one in every scenario.
        log2fc_cutoff: Smallest absolute effect size required to call a feature
            significant. Under ``by="omnibus"`` and ``by="contrast"`` that is
            ``abs(log2fc)``; under ``by="term"`` it is ``abs(log2_effect)``. The
            default of 1 is a two-fold change on a fold-change axis.
        pval_cutoff: Largest ``adj_pvalue`` allowed for a feature to be called
            significant.
        adj_type: Multiplicity adjustment. ``None`` reuses the ``pval_adj``
            column the comparison already computed. Naming a method re-adjusts
            the raw p-values instead, which is the only way to get a different
            adjustment without rerunning the comparison. Passing a method here
            does not adjust twice: the input is always the unadjusted ``pval``.
        by: Which p-value the verdict is read from. ``"omnibus"`` uses the named
            test's own table. ``"contrast"`` uses the pairwise stage of a
            multi-group comparison and returns one verdict table per contrast.
            ``"term"`` uses the ``terms`` table of a factorial comparison and
            returns one verdict table per model term, each carrying that term's
            own effect size in ``log2_effect`` rather than ``log2fc``.

    Returns:
        A :class:`~statassist.core.result.SaSignificance`. Its ``significance``
        is one table with ``features``, ``log2fc``, ``pvalue``, ``adj_pvalue``
        and ``is_signif`` - a multi-group omnibus table also carries
        ``extreme_level``, a factorial one ``extreme_cell`` - or, under
        ``by="contrast"``, a mapping of those tables keyed by contrast. Under
        ``by="term"``, a mapping of tables keyed by term with ``log2_effect`` in
        place of ``log2fc``. The cutoffs, the test and the adjustment actually
        used are in each table's ``attrs``, which is where ``draw_volcano_plot()``
        picks them up so that a plotted guide cannot disagree with the verdict.

    Raises:
        SaValueError: If an argument is unusable, or if the reading asked for
            has no axis in this comparison.

    Notes:
        ``is_signif`` combines the absolute effect size against
        ``log2fc_cutoff`` with ``adj_pvalue <= pval_cutoff`` and is therefore
        judged on the adjusted p-values. Pass ``adj_type="none"`` to test the
        raw ones. The effect column is ``log2fc`` except under ``by="term"``,
        where it is ``log2_effect``.

        The verdict is three-valued, as R's is. A feature whose effect is
        ``NaN`` - which is what two centres of opposite sign produce for a fold
        change - or whose p-value is missing is undecided rather than decided
        against, unless the magnitude cutoff already rules it out, which makes
        it ``False`` whatever the p-value would have been. A fold change of
        exactly zero is a different matter: ``log2fc`` is ``-inf``, an
        infinitely large decrease, and it clears any magnitude cutoff.

        The two ways of reading a multi-group comparison answer different
        questions. ``by="omnibus"`` asks whether a feature differs across the
        levels at all and pairs that with the one ``log2fc`` the ``effect``
        table carries, the most extreme level against the reference.
        ``by="contrast"`` asks the same question of one pair of levels at a time
        and carries the ``log2fc`` of that pair. Both divide by the reference,
        so the two readings agree on which way a feature moved.

        The adjustment axis differs too. Under ``by="contrast"`` with
        ``adj_type=None`` the ``pval_adj`` of the pairwise stage is reused, which
        was adjusted across the contrasts within each feature - or not at all for
        Tukey's HSD and Games-Howell, whose p-values are already family-wise.
        Naming a method instead adjusts across the features within each
        contrast, which is the axis ``by="omnibus"`` always works on.

        Under ``by="term"``, the verdict carries ``log2_effect``, the same ANOVA
        component ``terms["log2_effect"]`` holds, rather than a ratio of two
        centres renamed to ``log2fc``. It measures a deviation from what the
        rest of the model predicts, so a two-level factor whose levels differ by
        one log2 unit contributes -0.5 and +0.5 rather than 1. The default
        ``log2fc_cutoff=1`` is therefore a stricter demand here than the same
        number is elsewhere. ``draw_volcano_plot()`` plots ``|log2_effect|`` on
        that axis, so a term panel has no up/down colouring.

    Examples:
        >>> from statassist import compare_two_groups, simulate_two_groups
        >>> sim = simulate_two_groups(
        ...     n_feats=4, n_case=20, n_control=20, n_up=1, n_down=1, seed=1
        ... )
        >>> res = compare_two_groups(**sim.args, diagnose=False)
        >>> verdict = estimate_significance(res)
        >>> list(verdict.significance.columns)
        ['features', 'log2fc', 'pvalue', 'adj_pvalue', 'is_signif']

        The same comparison read through the rank-based test, without anything
        being recomputed.

        >>> ranked = estimate_significance(res, test="wilcox_test")
        >>> ranked.significance["pvalue"].equals(res.tests["wilcox_test"]["pval"])
        True

        The cutoffs travel with the table, so a plot cannot draw a guide the
        verdict was not judged against.

        >>> verdict.significance.attrs["log2fc_cutoff"]
        1.0
    """
    if by not in READINGS:
        raise SaValueError("`by` must be one of: " + ", ".join(READINGS) + ".")
    log2fc_cutoff = check_scalar_num(log2fc_cutoff, "log2fc_cutoff", 0)
    pval_cutoff = check_scalar_num(pval_cutoff, "pval_cutoff", 0, 1, lower_open=True)
    if adj_type is not None:
        adj_type = check_p_adjust(adj_type, "adj_type")

    if test is None:
        if not isinstance(comparison_result, Mapping) or "tests" not in comparison_result:
            raise SaValueError(
                "`comparison_result` must be a comparison result, as returned by "
                "compare_two_groups()."
            )
        test = next(iter(comparison_result["tests"]))
    table = pick_test(comparison_result, test, arg="comparison_result")

    if by == "term":
        return new_significance(
            comparison_result["analysis"],
            _by_term(comparison_result, test, adj_type, log2fc_cutoff, pval_cutoff),
        )

    if by == "contrast":
        return new_significance(
            comparison_result["analysis"],
            _by_contrast(comparison_result, test, adj_type, log2fc_cutoff, pval_cutoff),
        )

    pvalue = check_pvalues(table["pval"].to_numpy(dtype=float))

    # Re-adjusting `pval_adj` would apply the correction twice. The raw column is
    # always the input, so `adj_type` replaces the comparison's choice rather
    # than compounding it.
    if adj_type is None:
        adj_pvalue = table["pval_adj"].to_numpy(dtype=float)
        adj_used = comparison_result["parameters"].get("p_adjust")
    else:
        adj_pvalue = adjust(pvalue, adj_type)
        adj_used = adj_type

    effect = comparison_result["effect"]
    out = _verdict_table(
        comparison_result["features"],
        effect["log2fc"].to_numpy(dtype=float),
        pvalue,
        adj_pvalue,
        log2fc_cutoff,
        pval_cutoff,
    )
    # Which level or cell produced `log2fc` is not the same question in every
    # scenario, so the column is added where there is one to name.
    if comparison_result["analysis"] == "factorial_comparison":
        out["extreme_cell"] = list(effect["extreme_cell"])
    elif comparison_result["analysis"] == "multi_group_comparison":
        out["extreme_level"] = list(effect["extreme_level"])

    out.attrs.update(_verdict_attrs(comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff))
    return new_significance(comparison_result["analysis"], out)


def _verdict_attrs(
    comparison_result: Any,
    test: str,
    adj_used: Any,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, Any]:
    """The attributes a verdict table describes itself with.

    Port of ``sa_significance_attrs()``. ``draw_volcano_plot()`` reads the
    cutoffs back off here, so a table that has been passed around still knows
    which comparison and which thresholds produced it.
    """
    return {
        "analysis": comparison_result["analysis"],
        "group_lv": comparison_result["design"].get("group_lv"),
        "test": test,
        "test_label": comparison_result["test_info"][test]["label"],
        "adj_type": adj_used,
        "log2fc_cutoff": log2fc_cutoff,
        "pval_cutoff": pval_cutoff,
    }


def _verdict_table(
    features: Sequence[str],
    effect: np.ndarray,
    pvalue: np.ndarray,
    adj_pvalue: np.ndarray,
    log2fc_cutoff: float,
    pval_cutoff: float,
    effect_col: str = "log2fc",
) -> pd.DataFrame:
    """The verdict table both readings of a comparison produce.

    Port of ``sa_significance_table()``. Shared so that the two axes and the rule
    combining them cannot drift apart between the omnibus and the contrast paths.
    ``effect_col`` is ``"log2fc"`` for those paths and ``"log2_effect"`` for a
    term reading.

    ``is_signif`` is a nullable boolean rather than a plain one, which is what
    reproduces R's three-valued ``&``: a missing magnitude and a ``False``
    p-value verdict is still ``False``, while a missing magnitude and a ``True``
    one is undecided.
    """
    out = pd.DataFrame(
        {
            "features": [str(name) for name in features],
            "pvalue": pvalue,
            "adj_pvalue": adj_pvalue,
        }
    )
    out[effect_col] = effect
    # Keep the effect column between features and pvalue, matching the historical
    # column order of the fold-change reading.
    out = out[["features", effect_col, "pvalue", "adj_pvalue"]]
    out["is_signif"] = _at_least(np.abs(effect), log2fc_cutoff) & _at_most(
        adj_pvalue, pval_cutoff
    )
    return out


def _at_least(values: np.ndarray, bound: float) -> pd.Series:
    """``values >= bound``, missing where the value is."""
    return _kleene(values >= bound, values)


def _at_most(values: np.ndarray, bound: float) -> pd.Series:
    """``values <= bound``, missing where the value is."""
    return _kleene(values <= bound, values)


def _kleene(decided: np.ndarray, values: np.ndarray) -> pd.Series:
    """A comparison as a nullable boolean, undecided where the input was.

    NumPy calls every comparison against ``NaN`` ``False``, which would report a
    feature as decided against when it was never decided at all.
    """
    return pd.Series(decided, dtype="boolean").mask(np.isnan(values), other=pd.NA)


def _by_contrast(
    comparison_result: Any,
    test: str,
    adj_type: str | None,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, pd.DataFrame]:
    """One verdict table per pairwise contrast.

    Port of ``sa_significance_by_contrast()``.
    """
    pairwise = (
        comparison_result.get("pairwise", {}).get(test)
        if "pairwise" in (comparison_result)
        else None
    )
    if not pairwise:
        raise SaValueError(
            '`by = "contrast"` needs a pairwise stage, and '
            f"`comparison_result['pairwise'][{test!r}]` is absent. "
            "compare_multiple_groups() is the one scenario that builds it, and "
            "only when `posthoc = True`; a factorial comparison keeps its "
            "contrasts in `posthoc` alone."
        )

    out: dict[str, pd.DataFrame] = {}
    for contrast, table in pairwise.items():
        pvalue = check_pvalues(table["pval"].to_numpy(dtype=float))
        if adj_type is None:
            adj_pvalue = table["pval_adj"].to_numpy(dtype=float)
            adj_used = comparison_result["parameters"].get("posthoc_p_adjust")
        else:
            # Across the features of this one contrast, which is a different
            # family from the one the pairwise stage corrected over.
            adj_pvalue = adjust(pvalue, adj_type)
            adj_used = adj_type

        verdict = _verdict_table(
            table["features"],
            table["log2fc"].to_numpy(dtype=float),
            pvalue,
            adj_pvalue,
            log2fc_cutoff,
            pval_cutoff,
        )
        verdict.attrs.update(
            _verdict_attrs(comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff)
        )
        verdict.attrs.update(
            {
                "contrast": str(contrast),
                "group1": str(table["group1"].iloc[0]),
                "group2": str(table["group2"].iloc[0]),
            }
        )
        out[str(contrast)] = verdict
    return out


def _by_term(
    comparison_result: Any,
    test: str,
    adj_type: str | None,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, pd.DataFrame]:
    """One verdict table per model term.

    Port of ``sa_significance_by_term()``. The factorial counterpart of
    :func:`_by_contrast`, and the same shape: a named mapping of verdict tables,
    in the order ``terms`` lists the terms (main effects first, then
    interactions).
    """
    terms_tbl = comparison_result.get("terms") if "terms" in comparison_result else None
    if terms_tbl is None or len(terms_tbl.index) == 0:
        raise SaValueError(
            '`by = "term"` needs a term axis, and `comparison_result["terms"]` is '
            "absent. compare_factorial_groups() is the one scenario that builds "
            "it; a design with a single factor has one term and reads out through "
            '`by = "omnibus"`.'
        )

    labels = list(dict.fromkeys(str(name) for name in terms_tbl["terms"]))
    out: dict[str, pd.DataFrame] = {}
    for label in labels:
        table = terms_tbl.loc[terms_tbl["terms"].astype(str) == label]
        pvalue = check_pvalues(table["pval"].to_numpy(dtype=float))
        # Both branches adjust along the feature axis within one term, so unlike
        # the contrast reading there is no second family to choose between:
        # naming a method changes the method and nothing else.
        if adj_type is None:
            adj_pvalue = table["pval_adj"].to_numpy(dtype=float)
            adj_used = comparison_result["parameters"].get("p_adjust")
        else:
            adj_pvalue = adjust(pvalue, adj_type)
            adj_used = adj_type

        verdict = _verdict_table(
            table["features"],
            table["log2_effect"].to_numpy(dtype=float),
            pvalue,
            adj_pvalue,
            log2fc_cutoff,
            pval_cutoff,
            effect_col="log2_effect",
        )
        verdict.attrs.update(
            _verdict_attrs(comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff)
        )
        verdict.attrs.update(
            {
                "term": label,
                "term_order": int(table["term_order"].iloc[0]),
            }
        )
        out[label] = verdict
    return out
