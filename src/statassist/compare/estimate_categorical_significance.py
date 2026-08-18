"""Reduce a categorical comparison to one significance verdict per cell or table."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.categorical import (
    CategoricalResult,
    sa_categorical,
    sa_new_categorical_significance,
)
from statassist.utils.validate import p_adjust, sa_check_scalar_num


def sa_finite_or_na(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out if arr.ndim > 0 else float(out.flat[0])


def _sa_assoc_scale(measure: str) -> str:
    if measure in ("odds_ratio", "odds_ratio_paired"):
        return "ratio"
    return "magnitude"


def _sa_assoc_clears(measure: str, estimate: float, cutoff: float | None) -> bool:
    if cutoff is None:
        return True
    est = float(estimate)
    if _sa_assoc_scale(measure) == "ratio":
        return est >= cutoff or est <= 1 / cutoff
    return abs(est) >= cutoff


def _sa_check_effect_cutoff(cutoff: float | None, measure: str) -> None:
    if cutoff is None:
        return
    sa_check_scalar_num(cutoff, "effect_cutoff", lower=0, lower_open=True)
    if _sa_assoc_scale(measure) == "ratio" and cutoff < 1:
        raise ValueError(
            f"`effect_cutoff` is read on the scale of `{measure}`, a ratio centred "
            f"at 1, so it is a fold either way and has to be at least 1."
        )


def _sa_resolve_measure(res: CategoricalResult, measure: str) -> str:
    if measure == "auto":
        null = res.design.get("null")
        dims = res.design.get("dim", [])
        if null == "symmetry":
            return "odds_ratio_paired"
        if null == "marginal_homogeneity":
            return "kendalls_w"
        if len(dims) == 2 and all(d == 2 for d in dims):
            return "odds_ratio"
        return "cramers_v"
    avail = res.association["measure"].tolist()
    if measure not in avail:
        raise ValueError(
            f"`measure` must name one of the measures this design defines: "
            f"{', '.join(avail)}. Got {measure}."
        )
    return measure


def _sa_pick_categorical_test(res: CategoricalResult, test: str) -> pd.DataFrame:
    if test not in res.tests:
        raise ValueError(
            f"`test` must name one of the tests this result ran: "
            f"{', '.join(res.tests)}. Got {test}."
        )
    return res.tests[test]


def _sa_warn_unread_args(
    by: str,
    *,
    test_supplied: bool,
    measure_supplied: bool,
    effect_cutoff_supplied: bool,
    log2_lift_supplied: bool,
    adj_supplied: bool,
) -> None:
    if by == "cell":
        supplied = [
            name
            for name, flag in (
                ("test", test_supplied),
                ("measure", measure_supplied),
                ("effect_cutoff", effect_cutoff_supplied),
            )
            if flag
        ]
        suffix = (
            "A cell reading takes its p-value from the cell's own standardized "
            "residual, which no test and no association measure reports."
        )
    else:
        supplied = [
            name
            for name, flag in (
                ("log2_lift_cutoff", log2_lift_supplied),
                ("adj_type", adj_supplied),
            )
            if flag
        ]
        suffix = (
            "A table reading has one p-value and no family to adjust across, "
            "and its effect axis is `measure` rather than a lift."
        )
    if not supplied:
        return
    joined = " and ".join(f"`{s}`" for s in supplied)
    verb = "are" if len(supplied) > 1 else "is"
    warnings.warn(
        f"{joined} {verb} not read by `by = \"{by}\"` and were ignored. {suffix}",
        stacklevel=3,
    )


def _attrs_base(res: CategoricalResult, by: str, pval_cutoff: float) -> dict[str, Any]:
    return {
        "analysis": res.analysis,
        "null": res.design.get("null"),
        "by": by,
        "table_dim": res.design.get("dim"),
        "pval_cutoff": pval_cutoff,
    }


def _cell_significance(
    res: CategoricalResult,
    adj_type: str,
    log2_lift_cutoff: float,
    pval_cutoff: float,
) -> pd.DataFrame:
    cells = res.cells
    lift = sa_finite_or_na(cells["observed"].to_numpy() / cells["expected"].to_numpy())
    log2_lift = np.log2(lift)
    z = cells["std_residual"].to_numpy(dtype=float)
    pvalue = 2 * stats.norm.sf(np.abs(z))
    adj_pvalue = p_adjust(pvalue, adj_type)

    out = pd.DataFrame(
        {
            "row_level": cells["row_level"],
            "col_level": cells["col_level"],
            "observed": cells["observed"],
            "expected": cells["expected"],
            "lift": lift,
            "log2_lift": log2_lift,
            "std_residual": cells["std_residual"],
            "pvalue": pvalue,
            "adj_pvalue": adj_pvalue,
        }
    )
    out["is_signif"] = (np.abs(out["log2_lift"]) >= log2_lift_cutoff) & (
        out["adj_pvalue"] <= pval_cutoff
    )
    out.attrs.update(
        _attrs_base(res, "cell", pval_cutoff)
        | {
            "log2_lift_cutoff": log2_lift_cutoff,
            "adj_type": adj_type,
        }
    )
    return out


def _table_significance(
    res: CategoricalResult,
    tbl: pd.DataFrame,
    measure: str,
    effect_cutoff: float | None,
    pval_cutoff: float,
    test: str,
) -> pd.DataFrame:
    row = res.association.loc[res.association["measure"] == measure].iloc[0]
    pvalue = float(tbl["pval"].iloc[0])
    out = pd.DataFrame(
        {
            "measure": [measure],
            "estimate": [row["estimate"]],
            "lower_conf": [row["lower_conf"]],
            "upper_conf": [row["upper_conf"]],
            "pvalue": [pvalue],
        }
    )
    out["is_signif"] = _sa_assoc_clears(measure, row["estimate"], effect_cutoff) & (
        pvalue <= pval_cutoff
    )
    out.attrs.update(
        _attrs_base(res, "table", pval_cutoff)
        | {
            "test": test,
            "test_label": res.test_info[test]["label"],
            "measure": measure,
            "effect_cutoff": effect_cutoff,
        }
    )
    return out


def estimate_categorical_significance(
    categorical_result: CategoricalResult,
    by: str = "cell",
    test: str | None = None,
    log2_lift_cutoff: float = 1.0,
    pval_cutoff: float = 0.05,
    adj_type: str = "BH",
    measure: str = "auto",
    effect_cutoff: float | None = None,
    ratio_cutoff: float | None = None,
) -> CategoricalSignificanceResult:
    if not isinstance(categorical_result, CategoricalResult) and not isinstance(
        categorical_result, sa_categorical
    ):
        raise ValueError(
            "`categorical_result` must be a categorical comparison result."
        )

    if by not in ("cell", "table"):
        raise ValueError('`by` must be "cell" or "table".')

    if ratio_cutoff is not None:
        log2_lift_cutoff = float(np.log2(ratio_cutoff))

    sa_check_scalar_num(log2_lift_cutoff, "log2_lift_cutoff", lower=0)
    sa_check_scalar_num(pval_cutoff, "pval_cutoff", lower=0, upper=1, lower_open=True)

    res = categorical_result
    test_was_missing = test is None
    if test is None:
        test = next(iter(res.tests))

    _sa_warn_unread_args(
        by,
        test_supplied=not test_was_missing and by == "cell",
        measure_supplied=measure != "auto" and by == "cell",
        effect_cutoff_supplied=effect_cutoff is not None and by == "cell",
        log2_lift_supplied=by == "table" and log2_lift_cutoff != 1.0,
        adj_supplied=by == "table" and adj_type != "BH",
    )

    if by == "cell":
        if res.design.get("null") == "symmetry":
            raise ValueError(
                '`by = "cell"` needs a p-value per cell, and this result was '
                "tested for symmetry, where `$cells$std_residual` is NA throughout. "
                'Use `by = "table"`.'
            )
        significance = _cell_significance(res, adj_type, log2_lift_cutoff, pval_cutoff)
    else:
        tbl = _sa_pick_categorical_test(res, test)
        resolved = _sa_resolve_measure(res, measure)
        _sa_check_effect_cutoff(effect_cutoff, resolved)
        significance = _table_significance(
            res, tbl, resolved, effect_cutoff, pval_cutoff, test
        )

    return sa_new_categorical_significance(res.analysis, significance)
