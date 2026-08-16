"""Reduce a categorical comparison to one significance verdict per cell."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.categorical import CategoricalResult, sa_categorical
from statassist.utils.validate import p_adjust, sa_check_scalar_num


def estimate_categorical_significance(
    categorical_result: CategoricalResult,
    ratio_cutoff: float = 1.5,
    pval_cutoff: float = 0.05,
    adj_type: str = "BH",
) -> pd.DataFrame:
    if not isinstance(categorical_result, CategoricalResult) and not isinstance(
        categorical_result, sa_categorical
    ):
        raise ValueError(
            "`categorical_result` must be a categorical comparison result."
        )
    sa_check_scalar_num(ratio_cutoff, "ratio_cutoff", lower=0)
    sa_check_scalar_num(pval_cutoff, "pval_cutoff", lower=0, upper=1, lower_open=True)

    cells = categorical_result.cells.copy()
    ratio = cells["observed"] / cells["expected"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    z = cells["std_residual"].to_numpy()
    pvalue = 2 * stats.norm.sf(np.abs(z))
    adj = p_adjust(pvalue, adj_type)

    out = pd.DataFrame(
        {
            "row_level": cells["row_level"],
            "col_level": cells["col_level"],
            "ratio": ratio.to_numpy(),
            "pvalue": pvalue,
            "adj_pvalue": adj,
            "is_signif": (np.abs(np.log2(ratio.to_numpy())) >= np.log2(ratio_cutoff))
            & (adj <= pval_cutoff),
        }
    )
    out.attrs.update(
        {
            "analysis": categorical_result.analysis,
            "null": categorical_result.design.get("null"),
            "adj_type": adj_type,
            "ratio_cutoff": ratio_cutoff,
            "pval_cutoff": pval_cutoff,
        }
    )
    return out
