"""Significance verdict contract (sa_significance)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from statassist.contracts.repr import repr_sa_significance
from statassist.utils.validate import p_adjust, sa_check_pvalues, sa_check_scalar_num


class sa_result:
    """Marker base class matching the R S3 sa_result type."""


class sa_significance(sa_result):
    """Marker class for significance verdict results."""


@dataclass(repr=False)
class SignificanceResult:
    """Structured significance verdict matching the R sa_significance contract."""

    analysis_type: str
    significance: pd.DataFrame | dict[str, pd.DataFrame]

    def __repr__(self) -> str:
        return repr_sa_significance(self)

    def to_dict(self) -> dict[str, Any]:
        sig = self.significance
        if isinstance(sig, pd.DataFrame):
            payload: Any = sig.copy()
        else:
            payload = {k: v.copy() for k, v in sig.items()}
        return {
            "analysis_type": self.analysis_type,
            "significance": payload,
        }


def sa_new_significance(
    analysis_type: str,
    significance: pd.DataFrame | dict[str, pd.DataFrame],
) -> SignificanceResult:
    obj = SignificanceResult(
        analysis_type=analysis_type,
        significance=significance,
    )
    tagged_cls = type("SignificanceResult", (sa_significance, SignificanceResult), {})
    obj.__class__ = tagged_cls
    return obj


def sa_significance_table(
    features: list[str],
    log2fc: pd.Series | np.ndarray,
    pvalue: pd.Series | np.ndarray,
    adj_pvalue: pd.Series | np.ndarray,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "features": features,
            "log2fc": np.asarray(log2fc, dtype=float),
            "pvalue": np.asarray(pvalue, dtype=float),
            "adj_pvalue": np.asarray(adj_pvalue, dtype=float),
        }
    )
    out["is_signif"] = (np.abs(out["log2fc"]) >= log2fc_cutoff) & (
        out["adj_pvalue"] <= pval_cutoff
    )
    return out.reset_index(drop=True)


def sa_significance_attrs(
    comparison_result: Any,
    test: str,
    adj_used: str,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, Any]:
    return {
        "analysis": comparison_result.analysis,
        "group_lv": comparison_result.design.get("group_lv"),
        "test": test,
        "test_label": comparison_result.test_info[test]["label"],
        "adj_type": adj_used,
        "log2fc_cutoff": log2fc_cutoff,
        "pval_cutoff": pval_cutoff,
    }


def attach_significance_attrs(
    df: pd.DataFrame,
    attrs: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    merged = dict(attrs)
    if extra:
        merged.update(extra)
    out.attrs.update(merged)
    return out


def sa_significance_by_contrast(
    comparison_result: Any,
    test: str,
    adj_type: str | None,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, pd.DataFrame]:
    pairwise = comparison_result.pairwise
    if not pairwise or test not in pairwise or not pairwise[test]:
        raise ValueError(
            f'`by = "contrast"` needs a pairwise stage, and '
            f'`comparison_result$pairwise${test}` is absent. '
            "compare_multiple_groups() is the one scenario that builds it, and "
            "only when `posthoc = TRUE`; a factorial comparison keeps its "
            "contrasts in `$posthoc` alone."
        )

    blocks: dict[str, pd.DataFrame] = {}
    for contrast, tbl in pairwise[test].items():
        pvalue = tbl["pval"].to_numpy()
        sa_check_pvalues(pvalue)
        if adj_type is None:
            adj_pvalue = tbl["pval_adj"].to_numpy()
            adj_used = comparison_result.parameters.get("posthoc_p_adjust", "none")
        else:
            adj_pvalue = p_adjust(pvalue, adj_type)
            adj_used = adj_type

        out = sa_significance_table(
            tbl["features"].tolist(),
            tbl["log2fc"].to_numpy(),
            pvalue,
            adj_pvalue,
            log2fc_cutoff,
            pval_cutoff,
        )
        attrs = sa_significance_attrs(
            comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff
        )
        extra = {
            "contrast": tbl["contrast"].iloc[0],
            "group1": tbl["group1"].iloc[0],
            "group2": tbl["group2"].iloc[0],
        }
        blocks[contrast] = attach_significance_attrs(out, attrs, extra=extra)
    return blocks


def sa_significance_by_term(
    comparison_result: Any,
    test: str,
    adj_type: str | None,
    log2fc_cutoff: float,
    pval_cutoff: float,
) -> dict[str, pd.DataFrame]:
    terms_tbl = comparison_result.terms
    if terms_tbl is None or len(terms_tbl) == 0:
        raise ValueError(
            '`by = "term"` needs a term axis, and `comparison_result$terms` is '
            "absent. compare_factorial_groups() is the one scenario that builds "
            "it; a design with a single factor has one term and reads out through "
            '`by = "omnibus"`.'
        )

    labels = terms_tbl["terms"].drop_duplicates().tolist()
    blocks: dict[str, pd.DataFrame] = {}
    for label in labels:
        tbl = terms_tbl.loc[terms_tbl["terms"] == label]
        pvalue = tbl["pval"].to_numpy()
        sa_check_pvalues(pvalue)
        if adj_type is None:
            adj_pvalue = tbl["pval_adj"].to_numpy()
            adj_used = comparison_result.parameters["p_adjust"]
        else:
            adj_pvalue = p_adjust(pvalue, adj_type)
            adj_used = adj_type

        log2_effect = tbl["log2_effect"].to_numpy()
        if np.all(np.isnan(log2_effect)):
            log2_effect = np.zeros(len(tbl))

        out = sa_significance_table(
            tbl["features"].tolist(),
            log2_effect,
            pvalue,
            adj_pvalue,
            log2fc_cutoff,
            pval_cutoff,
        )
        attrs = sa_significance_attrs(
            comparison_result, test, adj_used, log2fc_cutoff, pval_cutoff
        )
        extra = {
            "term": tbl["terms"].iloc[0],
            "term_order": int(tbl["term_order"].iloc[0]),
        }
        blocks[label] = attach_significance_attrs(out, attrs, extra=extra)
    return blocks


def validate_significance_inputs(
    log2fc_cutoff: float,
    pval_cutoff: float,
    adj_type: str | None,
) -> None:
    sa_check_scalar_num(log2fc_cutoff, "log2fc_cutoff", lower=0)
    sa_check_scalar_num(
        pval_cutoff, "pval_cutoff", lower=0, upper=1, lower_open=True
    )
    if adj_type is not None:
        from statassist.utils.validate import sa_check_p_adjust

        sa_check_p_adjust(adj_type, "adj_type")
