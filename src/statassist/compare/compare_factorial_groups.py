"""Analyse a crossed-factor design as one model (simplified Type III two-way ANOVA)."""

from __future__ import annotations

import warnings
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

from statassist.contracts.comparison import sa_factorial, sa_new_comparison
from statassist.kernels.posthoc import sa_tukey
from statassist.utils.foldchange import sa_group_centers, sa_multi_fold_change, sa_resolve_fc_mean
from statassist.utils.validate import (
    p_adjust as adjust_pvalues,
    sa_check_flag,
    sa_check_p_adjust,
    sa_check_scalar_num,
    sa_control_first,
    sa_feature_table,
    sa_row,
    sa_validate_wide_input,
)


def _fact_grid(factor_lv: dict[str, list[str]]) -> pd.DataFrame:
    names = list(factor_lv.keys())
    levels = [factor_lv[n] for n in names]
    rows = list(product(*[range(len(lv)) for lv in levels]))
    out = pd.DataFrame(rows, columns=names)
    for nm, lv in factor_lv.items():
        out[nm] = [factor_lv[nm][i] for i in out[nm]]
    return out


def _cell_labels(factor_lv: dict[str, list[str]], grid: pd.DataFrame) -> list[str]:
    return [".".join(row) for _, row in grid.iterrows()]


def compare_factorial_groups(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    factors: dict[str, Any],
    factor_lv: dict[str, list[str]] | None = None,
    control_label: dict[str, str] | None = None,
    *,
    within: list[str] | None = None,
    id: Any = None,
    conf_level: float = 0.95,
    ss_type: str = "III",
    posthoc: bool = True,
    posthoc_alpha: float = 0.05,
    posthoc_scope: str = "marginal",
    fc_mean: str | None = None,
    input_scale: str = "raw",
    p_adjust: str = "BH",
    diagnose: bool = True,
):
    if within:
        raise ValueError(
            "`within` is not implemented in this version; repeated-measures factorial "
            "designs cannot be analysed here."
        )
    if id is not None:
        warnings.warn("`id` is ignored for independent factorial designs.", UserWarning)

    if ss_type not in ("I", "II", "III"):
        raise ValueError('`ss_type` must be one of "I", "II", or "III".')
    if posthoc_scope not in ("marginal", "simple", "both"):
        raise ValueError('`posthoc_scope` must be one of "marginal", "simple", or "both".')
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)
    sa_check_flag(posthoc, "posthoc")
    sa_check_flag(diagnose, "diagnose")
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    sa_check_scalar_num(posthoc_alpha, "posthoc_alpha", 0, 1, lower_open=True)
    sa_check_p_adjust(p_adjust, "p_adjust")

    if len(factors) < 2:
        raise ValueError(
            "`factors` must be a named list of at least two crossed factors."
        )

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    values: dict[str, np.ndarray] = {}
    for nm, src in factors.items():
        if isinstance(src, str):
            values[nm] = data[src].astype(str).to_numpy()
        else:
            values[nm] = np.asarray(src).astype(str)

    if factor_lv is None:
        factor_lv = {nm: sorted(np.unique(v[~pd.isna(v)]).tolist()) for nm, v in values.items()}
    else:
        factor_lv = {k: list(v) for k, v in factor_lv.items()}

    if control_label:
        for nm, ref in control_label.items():
            if nm in factor_lv:
                factor_lv[nm] = sa_control_first(factor_lv[nm], ref)

    # Keep complete rows for all factor levels
    mask = np.ones(len(data), dtype=bool)
    for nm, lv in factor_lv.items():
        mask &= np.isin(values[nm], lv)
    n_dropped = int((~mask).sum())
    data = data.loc[mask].reset_index(drop=True)
    for nm in values:
        values[nm] = values[nm][mask]

    grid = _fact_grid(factor_lv)
    cell_label = _cell_labels(factor_lv, grid)
    n_cells = len(cell_label)

    per_feature: list[dict[str, np.ndarray]] = []
    for f in feats:
        y = data[f].to_numpy(dtype=float)
        cell_samples: dict[str, np.ndarray] = {}
        for ci, lbl in enumerate(cell_label):
            m = np.ones(len(y), dtype=bool)
            for nm in factor_lv:
                m &= values[nm] == grid.iloc[ci][nm]
            cell_samples[lbl] = y[m & np.isfinite(y)]
        per_feature.append(cell_samples)

    centers = sa_group_centers(
        per_feature, feats, cell_label, fc_mean, paired=False, input_scale=input_scale
    )
    effect = sa_multi_fold_change(centers, feats, cell_label, fc_mean)
    effect = effect.rename(columns={"n_groups": "n_cells", "extreme_level": "extreme_cell"})
    effect = effect[
        [
            "features", "n_used", "n_cells", "ref_center", "extreme_cell",
            "extreme_center", "fold_change", "log2fc",
        ]
    ]

    factor_names = list(factor_lv.keys())
    formula_rhs = " * ".join(f"C({nm})" for nm in factor_names)
    typ = {"I": 1, "II": 2, "III": 3}[ss_type]

    term_rows: list[dict[str, Any]] = []
    whole_rows: list[pd.Series] = []
    ms_errors: list[float] = []
    fits: list[Any] = []

    for i, f in enumerate(feats):
        df_fit = pd.DataFrame({nm: values[nm] for nm in factor_names})
        df_fit["y"] = data[f].to_numpy()
        df_fit = df_fit.dropna()
        if df_fit.empty or df_fit["y"].size < n_cells + 1:
            whole_rows.append(sa_row(n_used=np.nan, n_groups=n_cells, f_stat=np.nan, df1=np.nan, df2=np.nan, pval=np.nan, lower_conf=np.nan, upper_conf=np.nan))
            ms_errors.append(np.nan)
            fits.append(None)
            continue
        try:
            model = ols(f"y ~ {formula_rhs}", data=df_fit).fit()
            aov = sm.stats.anova_lm(model, typ=typ)
            ms_errors.append(float(model.mse_resid))
            fits.append(model)
            ss_reg = float(aov["sum_sq"].sum() - aov.loc["Residual", "sum_sq"])
            ss_res = float(aov.loc["Residual", "sum_sq"])
            df1 = len(aov) - 2
            df2 = float(aov.loc["Residual", "df"])
            f_stat = (ss_reg / df1) / (ss_res / df2) if df2 > 0 else np.nan
            pval = float(stats.f.sf(f_stat, df1, df2)) if np.isfinite(f_stat) else np.nan
            whole_rows.append(
                sa_row(
                    n_used=df_fit.shape[0],
                    n_groups=n_cells,
                    f_stat=f_stat,
                    df1=df1,
                    df2=df2,
                    pval=pval,
                    lower_conf=np.nan,
                    upper_conf=np.nan,
                )
            )
            for term in aov.index[:-1]:
                term_rows.append(
                    {
                        "features": f,
                        "terms": term,
                        "term_order": term.count(":") + 1,
                        "n_used": df_fit.shape[0],
                        "df": float(aov.loc[term, "df"]),
                        "ss": float(aov.loc[term, "sum_sq"]),
                        "ms": float(aov.loc[term, "sum_sq"] / aov.loc[term, "df"]),
                        "f_stat": float(aov.loc[term, "F"]),
                        "df_error": df2,
                        "eta_sq": float(aov.loc[term, "sum_sq"] / (ss_reg + ss_res)),
                        "partial_eta_sq": float(
                            aov.loc[term, "sum_sq"]
                            / (aov.loc[term, "sum_sq"] + ss_res)
                        ),
                        "log2_effect": np.nan,
                        "pval": float(aov.loc[term, "PR(>F)"]),
                    }
                )
        except Exception:
            whole_rows.append(sa_row(n_used=np.nan, n_groups=n_cells, f_stat=np.nan, df1=np.nan, df2=np.nan, pval=np.nan, lower_conf=np.nan, upper_conf=np.nan))
            ms_errors.append(np.nan)
            fits.append(None)

    tests = {
        "anova_test": sa_feature_table(
            feats,
            ["n_used", "n_groups", "f_stat", "df1", "df2", "pval", "lower_conf", "upper_conf"],
            "Factorial ANOVA",
            lambda i: whole_rows[i],
            p_adjust_method=p_adjust,
        )
    }

    terms_tbl = pd.DataFrame(term_rows)
    if len(terms_tbl):
        for term in terms_tbl["terms"].unique():
            m = terms_tbl["terms"] == term
            terms_tbl.loc[m, "pval_adj"] = adjust_pvalues(
                terms_tbl.loc[m, "pval"].to_numpy(), p_adjust
            )

    cell_blocks = []
    for i, f in enumerate(feats):
        for ci, lbl in enumerate(cell_label):
            s = per_feature[i][lbl]
            cell_blocks.append(
                {
                    "features": f,
                    **{nm: grid.iloc[ci][nm] for nm in factor_names},
                    "cell": lbl,
                    "n": s.size,
                    "mean": float(np.mean(s)) if s.size else np.nan,
                    "sd": float(np.std(s, ddof=1)) if s.size > 1 else np.nan,
                    "se": (
                        float(np.sqrt(ms_errors[i] / s.size))
                        if s.size and np.isfinite(ms_errors[i])
                        else np.nan
                    ),
                }
            )
    cells_tbl = pd.DataFrame(cell_blocks)

    posthoc_tbl = pd.DataFrame()
    if posthoc and len(terms_tbl):
        ph_rows = []
        for i, f in enumerate(feats):
            if fits[i] is None:
                continue
            padj = terms_tbl.loc[terms_tbl["features"] == f].set_index("terms")["pval_adj"]
            for fac, lv in factor_lv.items():
                main = f"C({fac}, levels={factor_lv[fac]})"
                if main not in padj or padj[main] > posthoc_alpha:
                    continue
                try:
                    mat = sa_tukey(per_feature[i], conf_level)
                    pairs = list(combinations(range(len(cell_label)), 2))
                    for pi, (a, b) in enumerate(pairs):
                        if grid.iloc[a][fac] == grid.iloc[b][fac]:
                            continue
                        ph_rows.append(
                            {
                                "features": f,
                                "factor": fac,
                                "stratum": np.nan,
                                "contrast": f"{cell_label[a]} - {cell_label[b]}",
                                "group1": cell_label[a],
                                "group2": cell_label[b],
                                "n1": mat[pi, 0],
                                "n2": mat[pi, 1],
                                "estimate": mat[pi, 2],
                                "stderr": mat[pi, 3],
                                "statistic": mat[pi, 4],
                                "df": mat[pi, 5],
                                "pval": mat[pi, 6],
                                "pval_adj": mat[pi, 6],
                                "lower_conf": mat[pi, 7],
                                "upper_conf": mat[pi, 8],
                            }
                        )
                except Exception:
                    pass
        if ph_rows:
            posthoc_tbl = pd.DataFrame(ph_rows)

    n_factors = len(factor_lv)
    anova_type = {2: "two_way", 3: "three_way"}.get(n_factors, "factorial")
    label = f"{anova_type.replace('_', '-').title()} ANOVA (Type {ss_type} sums of squares)"

    diagnostics = None
    if diagnose:
        from statassist.compare.diagnose_distribution import sa_diagnose_samples

        diagnostics = sa_diagnose_samples(per_feature, feats, cell_label, paired=False)

    return sa_new_comparison(
        analysis="factorial_comparison",
        features=feats,
        design={
            "factor_lv": factor_lv,
            "anova_type": anova_type,
            "n_factors": n_factors,
            "group_lv": cell_label,
            "cell_n": [len(per_feature[0][c]) for c in cell_label],
            "n_empty_cells": sum(1 for c in cell_label if len(per_feature[0][c]) == 0),
            "paired": False,
            "pairing": None,
            "within": [],
            "n_dropped": n_dropped,
            "unmatched_ids": [],
        },
        parameters={
            "alternative": "two.sided",
            "conf_level": conf_level,
            "ss_type": ss_type,
            "fc_mean": fc_mean,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
            "posthoc": posthoc,
            "posthoc_alpha": posthoc_alpha,
            "posthoc_p_adjust": None,
            "posthoc_scope": posthoc_scope,
            "n_posthoc": { "anova_test": posthoc_tbl["features"].nunique() if len(posthoc_tbl) else 0 },
        },
        effect=effect,
        tests=tests,
        terms=terms_tbl if len(terms_tbl) else None,
        cells=cells_tbl,
        posthoc={"anova_test": posthoc_tbl} if len(posthoc_tbl) else None,
        test_info={
            "anova_test": {
                "id": "factorial_anova",
                "label": label,
                "paired": False,
                "posthoc_id": "factorial_tukey",
                "posthoc_label": "Tukey HSD on marginal means",
            }
        },
        diagnostics=diagnostics,
        subclass=sa_factorial,
    )
