"""Compare one sample against a hypothesised value."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.comparison import sa_new_comparison, sa_one_sample
from statassist.utils.foldchange import sa_fc_center, sa_resolve_fc_mean
from statassist.utils.validate import (
    sa_check_flag,
    sa_check_p_adjust,
    sa_check_scalar_num,
    sa_feature_table,
    sa_row,
    sa_split_for_screening,
)


def _one_sample_prop(
    v: np.ndarray,
    p: float,
    success: float,
    alternative: str,
    conf_level: float,
) -> pd.Series:
    observed = np.unique(v)
    if observed.size > 2:
        raise ValueError(
            f"the proportion test needs a binary feature, but this one takes "
            f"{observed.size} distinct values."
        )
    if success not in observed:
        raise ValueError(
            f"the value counted as a success, {success}, does not occur in this feature."
        )
    n = v.size
    n_success = int(np.sum(v == success))
    prop_res = stats.binomtest(n_success, n, p=p, alternative=alternative)
    proportion = n_success / n
    ci = prop_res.proportion_ci(confidence_level=conf_level)
    return sa_row(
        n_used=n,
        n_success=n_success,
        proportion=proportion,
        p=p,
        diff=proportion - p,
        chi_sq=np.nan,
        df=1,
        cohens_h=2 * np.arcsin(np.sqrt(proportion)) - 2 * np.arcsin(np.sqrt(p)),
        pval=float(prop_res.pvalue),
        lower_conf=float(ci.low),
        upper_conf=float(ci.high),
    )


def compare_one_sample(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    mu: float = 0,
    p: float = 0.5,
    success: float = 1,
    *,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    fc_mean: str | None = None,
    input_scale: str = "raw",
    p_adjust: str = "BH",
    diagnose: bool = True,
):
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(
            "`alternative` must be one of 'two.sided', 'less', or 'greater'."
        )
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)
    sa_check_scalar_num(mu, "mu")
    sa_check_scalar_num(p, "p", 0, 1, lower_open=True, upper_open=True)
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    sa_check_p_adjust(p_adjust, "p_adjust")
    sa_check_flag(diagnose, "diagnose")

    if isinstance(feats, str):
        feats = [feats]
    else:
        feats = list(feats)

    split = sa_split_for_screening(data, feats, group=None, group_lv=None)
    data = split["data"]

    samples = {
        f: data[f].to_numpy()[np.isfinite(data[f].to_numpy())] for f in feats
    }

    mu_ref = 2.0**mu if input_scale == "log2" else mu
    if not np.isfinite(mu_ref):
        raise ValueError(
            f"2^`mu` overflows to infinity, so `mu` = {mu} is not on the log2 scale."
        )

    effect = sa_feature_table(
        feats,
        ["n_used", "center", "mu", "diff", "fold_change", "log2fc"],
        "Fold change against mu",
        lambda i: _effect_row(samples[feats[i]], mu_ref, fc_mean, input_scale),
        p_adjust_method=None,
    )
    if mu_ref == 0:
        print(
            "`mu` is 0, so `fold_change` and `log2fc` are undefined and the "
            "`effect` table reports them as NA."
        )

    alt_map = {"two.sided": "two-sided", "less": "less", "greater": "greater"}

    t_result = sa_feature_table(
        feats,
        [
            "n_used", "center", "mu", "diff", "stderr", "t_stat", "df",
            "cohens_d", "pval", "lower_conf", "upper_conf",
        ],
        "One-sample t-test",
        lambda i: _t_row(samples[feats[i]], mu, alt_map[alternative], conf_level),
        p_adjust_method=p_adjust,
    )

    w_result = sa_feature_table(
        feats,
        ["n_used", "hl_shift", "v_stat", "pval", "lower_conf", "upper_conf"],
        "One-sample Wilcoxon signed-rank test",
        lambda i: _w_row(samples[feats[i]], mu, alt_map[alternative]),
        p_adjust_method=p_adjust,
    )

    prop_result = sa_feature_table(
        feats,
        [
            "n_used", "n_success", "proportion", "p", "diff", "chi_sq", "df",
            "cohens_h", "pval", "lower_conf", "upper_conf",
        ],
        "One-sample proportion test",
        lambda i: _one_sample_prop(
            samples[feats[i]], p, success, alt_map[alternative], conf_level
        ),
        p_adjust_method=p_adjust,
    )

    per_feature = [{ "sample": samples[f]} for f in feats]
    diagnostics = None
    if diagnose:
        from statassist.compare.diagnose_distribution import sa_diagnose_samples

        diagnostics = sa_diagnose_samples(
            per_feature, feats, ["sample"], paired=False
        )

    return sa_new_comparison(
        analysis="one_sample_comparison",
        features=feats,
        design={"mu": mu, "p": p, "success": success, "paired": False, "n_dropped": 0},
        parameters={
            "alternative": alternative,
            "conf_level": conf_level,
            "fc_mean": fc_mean,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
        },
        effect=effect,
        tests={
            "t_test": t_result,
            "wilcox_test": w_result,
            "prop_test": prop_result,
        },
        test_info={
            "t_test": {"id": "one_sample_t_test", "label": "One-sample t-test", "paired": False},
            "wilcox_test": {
                "id": "one_sample_wilcoxon",
                "label": "One-sample Wilcoxon signed-rank test",
                "paired": False,
            },
            "prop_test": {
                "id": "one_sample_proportion",
                "label": "One-sample proportion test",
                "paired": False,
            },
        },
        diagnostics=diagnostics,
        subclass=sa_one_sample,
    )


def _effect_row(
    v: np.ndarray, mu_ref: float, fc_mean: str, input_scale: str
) -> pd.Series:
    if v.size == 0:
        raise ValueError("no usable observation left.")
    center = sa_fc_center(v, "sample", fc_mean, input_scale)
    ratio = np.nan if mu_ref == 0 else center / mu_ref
    with np.errstate(invalid="ignore", divide="ignore"):
        log2fc = np.log2(ratio)
    return sa_row(
        n_used=v.size,
        center=center,
        mu=mu_ref,
        diff=center - mu_ref,
        fold_change=ratio,
        log2fc=log2fc,
    )


def _t_row(v: np.ndarray, mu: float, alternative: str, conf_level: float) -> pd.Series:
    if v.size < 2:
        raise ValueError(f"needs at least 2 usable observations, got {v.size}.")
    res = stats.ttest_1samp(v, popmean=mu, alternative=alternative)
    spread = float(np.std(v, ddof=1))
    se = spread / np.sqrt(v.size)
    alpha = 1 - conf_level
    df = v.size - 1
    diff = float(np.mean(v) - mu)
    q = stats.t.ppf(1 - alpha / 2, df) if alternative == "two-sided" else stats.t.ppf(1 - alpha, df)
    if alternative == "two-sided":
        ci = (diff - q * se, diff + q * se)
    elif alternative == "greater":
        ci = (diff - q * se, np.inf)
    else:
        ci = (-np.inf, diff + q * se)
    return sa_row(
        n_used=v.size,
        center=float(np.mean(v)),
        mu=mu,
        diff=diff,
        stderr=se,
        t_stat=float(res.statistic),
        df=float(df),
        cohens_d=(diff / spread) if spread > 0 else np.nan,
        pval=float(res.pvalue),
        lower_conf=ci[0],
        upper_conf=ci[1],
    )


def _w_row(v: np.ndarray, mu: float, alternative: str) -> pd.Series:
    if v.size < 1:
        raise ValueError("needs at least 1 usable observation.")
    res = stats.wilcoxon(v - mu, alternative=alternative, method="auto")
    return sa_row(
        n_used=v.size,
        hl_shift=float(np.median(v - mu)),
        v_stat=float(res.statistic),
        pval=float(res.pvalue),
        lower_conf=np.nan,
        upper_conf=np.nan,
    )
