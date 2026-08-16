"""Reduce a comparison to one significance verdict per feature."""

from __future__ import annotations

from typing import Any

import numpy as np

from statassist.contracts.comparison import ComparisonResult, sa_pick_test
from statassist.contracts.significance import (
    SignificanceResult,
    attach_significance_attrs,
    sa_new_significance,
    sa_significance_attrs,
    sa_significance_by_contrast,
    sa_significance_by_term,
    sa_significance_table,
    validate_significance_inputs,
)
from statassist.utils.validate import p_adjust, sa_check_pvalues


def estimate_significance(
    comparison_result: ComparisonResult,
    test: str | None = None,
    log2fc_cutoff: float = 1.0,
    pval_cutoff: float = 0.05,
    adj_type: str | None = None,
    by: str = "omnibus",
) -> SignificanceResult:
    if by not in ("omnibus", "contrast", "term"):
        raise ValueError('`by` must be one of "omnibus", "contrast", or "term".')

    validate_significance_inputs(log2fc_cutoff, pval_cutoff, adj_type)

    if comparison_result.analysis == "categorical_comparison":
        raise ValueError(
            "`comparison_result` is a categorical comparison. This scenario has "
            "no feature axis, so it cannot be read as a volcano plot. "
            "estimate_categorical_significance() is what reads it, one verdict "
            "per cell of the table, and draw_mosaic_plot() is what draws it."
        )

    if test is None:
        test = next(iter(comparison_result.tests))

    if by == "contrast":
        blocks = sa_significance_by_contrast(
            comparison_result,
            test,
            adj_type,
            log2fc_cutoff,
            pval_cutoff,
        )
        return sa_new_significance(comparison_result.analysis, blocks)

    if by == "term":
        blocks = sa_significance_by_term(
            comparison_result,
            test,
            adj_type,
            log2fc_cutoff,
            pval_cutoff,
        )
        return sa_new_significance(comparison_result.analysis, blocks)

    tbl = sa_pick_test(comparison_result, test)
    pvalue = tbl["pval"].to_numpy()
    sa_check_pvalues(pvalue)

    if adj_type is None:
        adj_pvalue = tbl["pval_adj"].to_numpy()
        adj_used = comparison_result.parameters["p_adjust"]
    else:
        adj_pvalue = p_adjust(pvalue, adj_type)
        adj_used = adj_type

    out = sa_significance_table(
        comparison_result.features,
        comparison_result.effect["log2fc"].to_numpy(),
        pvalue,
        adj_pvalue,
        log2fc_cutoff,
        pval_cutoff,
    )

    if comparison_result.analysis == "factorial_comparison":
        out["extreme_cell"] = comparison_result.effect["extreme_cell"].to_numpy()
    elif comparison_result.analysis == "multi_group_comparison":
        out["extreme_level"] = comparison_result.effect["extreme_level"].to_numpy()

    attrs = sa_significance_attrs(
        comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff
    )
    out = attach_significance_attrs(out, attrs)
    return sa_new_significance(comparison_result.analysis, out)
