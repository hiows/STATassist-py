"""Formatting helpers for R-style result summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def sa_fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x) or math.isinf(x):
        return "NA"
    text = f"{x:.{digits}g}"
    if "." not in text and "e" not in text.lower():
        text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return text


def sa_fmt_est(value: Any) -> str:
    return sa_fmt_num(value, 3)


def sa_fmt_pval(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    x = float(value)
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.3g}"


def sa_cat_field(name: str, value: str, width: int = 10) -> str:
    return f"  {name:<{width}}: {value}\n"


def sa_left(name: str, width: int) -> str:
    return f"{name:<{width}}"


def sa_signif_count(tbl: pd.DataFrame, alpha: float = 0.05) -> tuple[int, int, int]:
    if "pval_adj" in tbl.columns:
        p = tbl["pval_adj"]
    elif "adj_pvalue" in tbl.columns:
        p = tbl["adj_pvalue"]
    else:
        p = tbl.get("pval", pd.Series(dtype=float))
    n_signif = int((p.notna() & (p <= alpha)).sum())
    n_failed = int(tbl["pval"].isna().sum()) if "pval" in tbl.columns else 0
    return n_signif, len(tbl), n_failed


def sa_verdict_count(tbl: pd.DataFrame) -> str:
    if "is_signif" not in tbl.columns:
        n_signif, n_total, _ = sa_signif_count(tbl)
        return f"{n_signif} of {n_total} significant"
    n_signif = int((tbl["is_signif"] == True).sum())  # noqa: E712
    n_undecided = int(tbl["is_signif"].isna().sum())
    text = f"{n_signif} of {len(tbl)} significant"
    if n_undecided:
        text += f"  ({n_undecided} undecided)"
    return text


def sa_class_tag(obj: Any) -> str:
    markers = (
        "sa_two_group",
        "sa_multi_group",
        "sa_one_sample",
        "sa_factorial",
        "sa_categorical",
        "sa_significance",
        "sa_categorical_significance",
        "sa_diagnosis",
        "sa_model",
        "sa_selection",
        "sa_performance",
        "sa_reduction",
        "sa_cluster",
        "sa_split",
    )
    if isinstance(obj, dict):
        cls_tuple = obj.get("__class__", ())
        if cls_tuple:
            return str(cls_tuple[0])
    for cls in type(obj).__mro__:
        if cls.__name__ in markers:
            return cls.__name__
    return type(obj).__name__
