"""Test a contingency table with every applicable test at once."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.categorical import (
    sa_association_columns,
    sa_categorical_cell_columns,
    sa_categorical_test_columns,
    sa_new_categorical,
)
from statassist.utils.validate import (
    sa_check_flag,
    sa_check_scalar_num,
    sa_preserve_seed,
)


def _build_cell_table(
    counts: np.ndarray,
    row_lv: list[str],
    col_lv: list[str],
    null: str,
) -> pd.DataFrame:
    n = counts.sum()
    row_m = counts.sum(axis=1, keepdims=True)
    col_m = counts.sum(axis=0, keepdims=True)

    if null == "independence":
        expected = row_m @ col_m / n
    elif null == "symmetry":
        expected = (counts + counts.T) / 2
    else:
        expected = np.outer(row_m.ravel(), col_m.ravel()) / n

    rows = []
    for i, rl in enumerate(row_lv):
        for j, cl in enumerate(col_lv):
            obs = float(counts[i, j])
            exp = float(expected[i, j])
            resid = (obs - exp) / np.sqrt(exp) if exp > 0 else np.nan
            if null == "symmetry":
                std_res = np.nan
            else:
                denom = exp * (1 - row_m[i, 0] / n) * (1 - col_m[0, j] / n)
                std_res = (obs - exp) / np.sqrt(denom) if denom > 0 else np.nan
            rows.append(
                {
                    "row_level": rl,
                    "col_level": cl,
                    "observed": obs,
                    "expected": exp,
                    "residual": resid,
                    "std_residual": std_res,
                    "prop_total": obs / n if n else np.nan,
                    "prop_row": obs / row_m[i, 0] if row_m[i, 0] else np.nan,
                    "prop_col": obs / col_m[0, j] if col_m[0, j] else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _association_measures(counts: np.ndarray, null: str) -> pd.DataFrame:
    n = counts.sum()
    chi2, _, _, _ = stats.chi2_contingency(counts)
    r, c = counts.shape
    k = min(r - 1, c - 1)
    cramers = np.sqrt(chi2 / (n * k)) if n and k else np.nan
    cont_coef = np.sqrt(chi2 / (chi2 + n)) if n else np.nan
    rows = [
        {"measure": "cramers_v", "estimate": cramers, "lower_conf": np.nan, "upper_conf": np.nan},
        {
            "measure": "contingency_coefficient",
            "estimate": cont_coef,
            "lower_conf": np.nan,
            "upper_conf": np.nan,
        },
    ]
    if r == 2 and c == 2:
        phi = np.sqrt(chi2 / n) if n else np.nan
        a, b, c_, d = counts.ravel()
        if min(a, b, c_, d) == 0:
            or_est = np.nan
        else:
            or_est = (a * d) / (b * c_)
        rows.extend(
            [
                {"measure": "phi_coefficient", "estimate": phi, "lower_conf": np.nan, "upper_conf": np.nan},
                {"measure": "odds_ratio", "estimate": or_est, "lower_conf": np.nan, "upper_conf": np.nan},
            ]
        )
    return pd.DataFrame(rows)[sa_association_columns()]


def compare_categorical_groups(
    data: pd.DataFrame | np.ndarray,
    category_lv: dict[str, list[str]] | None = None,
    control_label: dict[str, str] | str | None = None,
    *,
    paired: bool = False,
    conf_level: float = 0.95,
    correct: bool = True,
    exact: bool | None = None,
    simulate_p_value: bool = False,
    n_resamples: int = 9999,
    max_levels: int = 20,
    seed: float | None = None,
    diagnose: bool = True,
):
    sa_check_flag(paired, "paired")
    sa_check_flag(correct, "correct")
    sa_check_flag(simulate_p_value, "simulate_p_value")
    sa_check_flag(diagnose, "diagnose")
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    with sa_preserve_seed(seed):
        if category_lv is None:
            category_lv = {
                col: sorted(data[col].astype(str).dropna().unique().tolist())
                for col in data.columns
            }
        variables = list(category_lv.keys())

        if control_label:
            if isinstance(control_label, str):
                cl = {variables[0]: control_label}
            else:
                cl = dict(control_label)
            for var, ref in cl.items():
                if var in category_lv and ref in category_lv[var]:
                    rest = [x for x in category_lv[var] if x != ref]
                    category_lv[var] = [ref] + rest

        clean = data[list(category_lv.keys())].astype(str)
        for var, lv in category_lv.items():
            clean = clean[clean[var].isin(lv)]
        clean = clean.dropna(how="any")
        n_dropped = len(data) - len(clean.loc[data.index.isin(clean.index)])
        n_incomplete = len(data) - len(data.dropna(subset=list(category_lv.keys())))

        if paired:
            if len(variables) < 2:
                raise ValueError("a matched design needs at least two condition columns.")
            if any(len(category_lv[v]) != 2 for v in variables):
                raise ValueError(
                    "a matched design requires binary levels for every condition column."
                )
            null = "symmetry" if len(variables) == 2 else "marginal_homogeneity"
            row_var, col_var = variables[0], variables[1]
            row_lv, col_lv = category_lv[row_var], category_lv[col_var]
            counts = np.zeros((len(row_lv), len(col_lv)))
            for _, row in clean.iterrows():
                counts[row_lv.index(row[row_var]), col_lv.index(row[col_var])] += 1

            if len(variables) == 2:
                b, c_ = counts[0, 1], counts[1, 0]
                use_exact = (exact if exact is not None else (b + c_) < 25)
                if use_exact:
                    pval = float(stats.binomtest(b, b + c_, 0.5).pvalue) if b + c_ else 1.0
                    stat = np.nan
                    df = np.nan
                    exact_used = True
                else:
                    stat = (abs(b - c_) - 1) ** 2 / (b + c_) if b + c_ else np.nan
                    df = 1.0
                    pval = float(stats.chi2.sf(stat, 1)) if np.isfinite(stat) else np.nan
                    exact_used = False
                tests = {
                    "mcnemar_test": pd.DataFrame(
                        [{
                            "n_used": int(counts.sum()),
                            "statistic": stat,
                            "df": df,
                            "pval": pval,
                            "lower_conf": np.nan,
                            "upper_conf": np.nan,
                        }]
                    )
                }
                test_info = {
                    "mcnemar_test": {
                        "id": "mcnemar",
                        "label": "McNemar's test",
                        "paired": True,
                    }
                }
            else:
                stat, pval, df, _ = stats.chi2_contingency(counts)
                tests = {
                    "cochran_q": pd.DataFrame(
                        [{
                            "n_used": int(counts.sum()),
                            "statistic": float(stat),
                            "df": float(df),
                            "pval": float(pval),
                            "lower_conf": np.nan,
                            "upper_conf": np.nan,
                        }]
                    )
                }
                test_info = {
                    "cochran_q": {
                        "id": "cochran_q",
                        "label": "Cochran's Q test",
                        "paired": True,
                    }
                }
                exact_used = None
        else:
            if len(variables) != 2:
                raise ValueError(
                    "an independent categorical comparison crosses exactly two variables."
                )
            null = "independence"
            row_var, col_var = variables[0], variables[1]
            row_lv, col_lv = category_lv[row_var], category_lv[col_var]
            counts = np.zeros((len(row_lv), len(col_lv)))
            for _, row in clean.iterrows():
                counts[row_lv.index(row[row_var]), col_lv.index(row[col_var])] += 1

            if simulate_p_value:
                chi_res = stats.chi2_contingency(counts)
                stat, pval, df = float(chi_res[0]), float(chi_res[1]), float(chi_res[2])
            else:
                chi_res = stats.chi2_contingency(counts, correction=correct)
                stat, pval, df = float(chi_res[0]), float(chi_res[1]), float(chi_res[2])

            tests = {
                "chisq_test": pd.DataFrame(
                    [{
                        "n_used": int(counts.sum()),
                        "statistic": stat,
                        "df": df,
                        "pval": pval,
                        "lower_conf": np.nan,
                        "upper_conf": np.nan,
                    }]
                )
            }
            test_info = {
                "chisq_test": {
                    "id": "chisq_independence",
                    "label": "Chi-square test of independence",
                    "paired": False,
                }
            }

            if counts.shape == (2, 2):
                try:
                    _, fisher_p = stats.fisher_exact(counts)
                    tests["fisher_test"] = pd.DataFrame(
                        [{
                            "n_used": int(counts.sum()),
                            "statistic": np.nan,
                            "df": np.nan,
                            "pval": float(fisher_p),
                            "lower_conf": np.nan,
                            "upper_conf": np.nan,
                        }]
                    )
                    test_info["fisher_test"] = {
                        "id": "fisher_exact",
                        "label": "Fisher's exact test",
                        "paired": False,
                    }
                except Exception:
                    pass
            exact_used = None

        cells = _build_cell_table(counts, row_lv, col_lv, null)
        association = _association_measures(counts, null)
        diagnostics = None
        if diagnose:
            exp = cells["expected"].to_numpy().reshape(counts.shape)
            small = int(np.sum(exp < 5))
            diagnostics = {
                "approx_ok": small == 0,
                "n_small_expected": small,
                "note": f"{small} cell(s) have expected count below 5.",
            }

        return sa_new_categorical(
            analysis="categorical_comparison",
            variables=variables,
            design={
                "category_lv": category_lv,
                "null": null,
                "paired": paired,
                "pairing": "row" if paired else None,
                "dim": list(counts.shape),
                "row_var": row_var,
                "col_var": col_var,
                "n_used": int(counts.sum()),
                "n_dropped": n_dropped,
                "n_incomplete": n_incomplete,
            },
            parameters={
                "conf_level": conf_level,
                "correct": correct,
                "exact": exact_used,
                "simulate_p_value": simulate_p_value,
                "n_resamples": n_resamples if simulate_p_value else None,
                "max_levels": max_levels,
                "seed": seed,
            },
            cells=cells,
            tests=tests,
            test_info=test_info,
            association=association,
            diagnostics=diagnostics,
        )
