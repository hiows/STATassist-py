"""Run every applicable two-group test at once."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from statassist.contracts.comparison import ComparisonResult, sa_new_comparison, sa_two_group
from statassist.kernels.diagnostic import sa_bartlett, sa_levene
from statassist.kernels.robust import sa_brunner_munzel, sa_yuen_paired
from statassist.utils.foldchange import sa_fold_change, sa_resolve_fc_mean
from statassist.utils.validate import (
    sa_check_flag,
    sa_check_p_adjust,
    sa_check_scalar_num,
    sa_control_first,
    sa_feature_table,
    sa_pair_by_id,
    sa_pair_by_order,
    sa_row,
    sa_validate_wide_input,
)


def _map_alternative(alternative: str) -> Literal["two-sided", "less", "greater"]:
    mapping = {
        "two.sided": "two-sided",
        "less": "less",
        "greater": "greater",
    }
    return mapping[alternative]  # type: ignore[return-value]


def _walsh_averages(d: np.ndarray) -> np.ndarray:
    n = d.size
    vals = [(d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return np.sort(vals)


def _hl_ci_paired(d: np.ndarray, conf_level: float) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    wa = _walsh_averages(d)
    m = wa.size
    alpha = 1.0 - conf_level
    n = d.size
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.round((m - z * np.sqrt(m * (n + 1) / 6.0)) / 2))
    k = max(1, min(k, m // 2))
    return float(wa[k - 1]), float(wa[m - k])


def _hl_ci_independent(
    x: np.ndarray, y: np.ndarray, conf_level: float
) -> tuple[float, float]:
    diffs = np.sort(np.subtract.outer(x, y).ravel())
    m = diffs.size
    alpha = 1.0 - conf_level
    n1, n2 = x.size, y.size
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.round((m - z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)) / 2))
    k = max(1, min(k, m // 2))
    return float(diffs[k - 1]), float(diffs[m - k])


def _wilcox_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    paired: bool,
    alternative: str,
    conf_level: float,
) -> tuple[float, float, float, float, float]:
    alt = _map_alternative(alternative)
    if paired:
        stat, pval = stats.wilcoxon(x, y, alternative=alt, method="auto")
        hl = float(np.median(x - y))
        lower, upper = _hl_ci_paired(x - y, conf_level)
        if alternative == "greater":
            lower, upper = lower, np.inf
        elif alternative == "less":
            lower, upper = -np.inf, upper
        return float(stat), float(pval), hl, float(lower), float(upper)

    stat, pval = stats.mannwhitneyu(x, y, alternative=alt, method="auto")
    hl = float(np.median(np.subtract.outer(x, y)))
    lower, upper = _hl_ci_independent(x, y, conf_level)
    if alternative == "greater":
        lower, upper = lower, np.inf
    elif alternative == "less":
        lower, upper = -np.inf, upper
    return float(stat), float(pval), hl, lower, upper


def _diagnose_samples(
    per_feature: list[dict[str, np.ndarray]],
    feats: list[str],
    group_lv: list[str],
) -> dict[str, Any]:
    """Attach assumption checks a comparison rests on."""
    norm_rows: list[dict[str, Any]] = []
    var_rows: list[dict[str, Any]] = []

    for f, samples in zip(feats, per_feature):
        group_samples = [samples[lv] for lv in group_lv]
        for lv, v in zip(group_lv, group_samples):
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            row: dict[str, Any] = {
                "features": f,
                "group": lv,
                "n_used": v.size,
            }
            if v.size >= 3:
                sh = stats.shapiro(v)
                row["shapiro_stat"] = float(sh.statistic)
                row["shapiro_pval"] = float(sh.pvalue)
            else:
                row["shapiro_stat"] = np.nan
                row["shapiro_pval"] = np.nan
            if v.size >= 2:
                mu, sig = float(v.mean()), float(v.std(ddof=1))
                if sig > 0:
                    ks = stats.kstest(v, stats.norm.cdf, args=(mu, sig))
                    row["ks_stat"] = float(ks.statistic)
                    row["ks_pval"] = float(ks.pvalue)
                else:
                    row["ks_stat"] = np.nan
                    row["ks_pval"] = np.nan
                row["skewness"] = float(stats.skew(v, bias=False))
                row["excess_kurtosis"] = float(stats.kurtosis(v, bias=False))
            else:
                row["ks_stat"] = np.nan
                row["ks_pval"] = np.nan
                row["skewness"] = np.nan
                row["excess_kurtosis"] = np.nan
            norm_rows.append(row)

        vals = [np.asarray(s, dtype=float) for s in group_samples]
        vals = [v[np.isfinite(v)] for v in vals]
        n_used = sum(v.size for v in vals)
        var_row: dict[str, Any] = {
            "features": f,
            "n_used": n_used,
            "n_groups": len(group_lv),
        }
        if all(v.size >= 2 for v in vals):
            lev = sa_levene(vals, center="median")
            var_row.update(lev)
            bart = sa_bartlett(vals)
            var_row.update(bart)
        else:
            for col in (
                "levene_stat",
                "levene_df1",
                "levene_df2",
                "levene_pval",
                "bartlett_stat",
                "bartlett_df",
                "bartlett_pval",
            ):
                var_row[col] = np.nan
        var_rows.append(var_row)

    normality = pd.DataFrame(norm_rows)
    variance = pd.DataFrame(var_rows)
    summary = pd.DataFrame(
        {
            "features": feats,
            "normal_all_groups": [
                bool(
                    np.all(
                        normality.loc[
                            (normality["features"] == f) & normality["shapiro_pval"].notna(),
                            "shapiro_pval",
                        ]
                        > 0.05
                    )
                )
                if normality.loc[normality["features"] == f, "shapiro_pval"].notna().any()
                else False
                for f in feats
            ],
            "variance_homogeneous": [
                bool(
                    variance.loc[variance["features"] == f, "levene_pval"].iloc[0] > 0.05
                )
                if variance.loc[variance["features"] == f, "levene_pval"].notna().any()
                else False
                for f in feats
            ],
        }
    )
    return {
        "normality": normality,
        "variance": variance,
        "summary": summary,
    }


def compare_two_groups(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: pd.Series | np.ndarray | list[Any],
    group_lv: list[str],
    *,
    control_label: str | None = None,
    id: list[str] | np.ndarray | None = None,
    alternative: str = "two.sided",
    paired: bool = False,
    conf_level: float = 0.95,
    tr: float = 0.2,
    fc_mean: str | None = None,
    input_scale: str = "raw",
    p_adjust: str = "BH",
    diagnose: bool = True,
) -> ComparisonResult:
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(
            "`alternative` must be one of 'two.sided', 'less', or 'greater'."
        )
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)
    sa_check_flag(paired, "paired")
    sa_check_flag(diagnose, "diagnose")
    sa_check_scalar_num(
        conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True
    )
    sa_check_scalar_num(tr, "tr", 0, 0.5, upper_open=True)
    sa_check_p_adjust(p_adjust, "p_adjust")

    if id is not None and not paired:
        warnings.warn(
            "`id` is only used to form pairs and is ignored when "
            "`paired = FALSE`. Set `paired = TRUE` if the observations are "
            "matched.",
            UserWarning,
            stacklevel=2,
        )

    if control_label is None:
        control_label = group_lv[0]

    inp = sa_validate_wide_input(
        data, feats, group, group_lv, id=id, n_levels=2
    )
    data = inp["data"]
    feats = inp["feats"]
    group = inp["group"]
    id_arr = inp["id"]
    group_lv = sa_control_first(list(group.categories), control_label)
    group = pd.Categorical(group.astype(str), categories=group_lv, ordered=True)

    if inp["n_dropped"] > 0:
        print(
            f"Dropped {inp['n_dropped']} row(s) belonging to a level outside "
            "`group_lv`."
        )

    lv_xy = list(reversed(group_lv))

    if not paired:
        group_arr = np.asarray(group)
        idx_x = np.where(group_arr == lv_xy[0])[0]
        idx_y = np.where(group_arr == lv_xy[1])[0]
        unmatched: list[str] = []
    else:
        pairing = (
            sa_pair_by_order(group, lv_xy)
            if id_arr is None
            else sa_pair_by_id(id_arr, group, lv_xy)
        )
        idx_x = pairing["idx_x"]
        idx_y = pairing["idx_y"]
        unmatched = list(pairing["unmatched"])
        if unmatched:
            print(
                f"Dropped {len(unmatched)} id(s) present in only one group: "
                f"{', '.join(unmatched)}."
            )

    samples: list[dict[str, np.ndarray]] = []
    for f in feats:
        x = data[f].to_numpy()[idx_x]
        y = data[f].to_numpy()[idx_y]
        if paired:
            keep = ~np.isnan(x) & ~np.isnan(y)
            samples.append({"x": x[keep], "y": y[keep]})
        else:
            samples.append(
                {"x": x[~np.isnan(x)], "y": y[~np.isnan(y)]}
            )

    effect = sa_fold_change(samples, feats, lv_xy, fc_mean, input_scale)

    alt = alternative
    t_label = "Paired t-test" if paired else "Welch's t-test"
    t_columns = [
        "n_x",
        "n_y",
        "n_used",
        "x_mean",
        "y_mean",
        "mean_diff",
        "stderr",
        "t_stat",
        "df",
        "pval",
        "lower_conf",
        "upper_conf",
    ]

    def _t_row(i: int) -> pd.Series:
        s = samples[i]
        n_x = s["x"].size
        n_y = s["y"].size
        if n_x < 2 or n_y < 2:
            raise ValueError(
                f"needs at least 2 usable observations per group, got {n_x} and {n_y}."
            )
        if paired:
            res = stats.ttest_rel(s["x"], s["y"], alternative=_map_alternative(alt))
            se = float(
                np.std(s["x"] - s["y"], ddof=1) / np.sqrt(n_x)
            )
            ci = _t_conf_int_paired(s["x"], s["y"], alt, conf_level)
        else:
            res = stats.ttest_ind(
                s["x"], s["y"], equal_var=False, alternative=_map_alternative(alt)
            )
            se = float(res.stderr) if hasattr(res, "stderr") else np.nan
            ci = _t_conf_int_welch(s["x"], s["y"], alt, conf_level)
        return sa_row(
            n_x=n_x,
            n_y=n_y,
            n_used=n_x if paired else n_x + n_y,
            x_mean=float(np.mean(s["x"])),
            y_mean=float(np.mean(s["y"])),
            mean_diff=float(np.mean(s["x"]) - np.mean(s["y"])),
            stderr=se,
            t_stat=float(res.statistic),
            df=float(res.df),
            pval=float(res.pvalue),
            lower_conf=ci[0],
            upper_conf=ci[1],
        )

    t_result = sa_feature_table(feats, t_columns, t_label, _t_row, p_adjust)

    w_label = (
        "Wilcoxon signed-rank test"
        if paired
        else "Wilcoxon rank sum test (Mann-Whitney U test)"
    )
    w_columns = [
        "n_x",
        "n_y",
        "n_used",
        "hl_shift",
        "w_stat",
        "pval",
        "lower_conf",
        "upper_conf",
    ]

    def _w_row(i: int) -> pd.Series:
        s = samples[i]
        n_x = s["x"].size
        n_y = s["y"].size
        if n_x < 1 or n_y < 1:
            raise ValueError(
                f"needs at least 1 usable observation per group, got {n_x} and {n_y}."
            )
        w_stat, pval, hl, lower, upper = _wilcox_test(
            s["x"], s["y"], paired=paired, alternative=alt, conf_level=conf_level
        )
        return sa_row(
            n_x=n_x,
            n_y=n_y,
            n_used=n_x if paired else n_x + n_y,
            hl_shift=hl,
            w_stat=w_stat,
            pval=pval,
            lower_conf=lower,
            upper_conf=upper,
        )

    w_result = sa_feature_table(feats, w_columns, w_label, _w_row, p_adjust)

    if not paired:
        robust_label = "Brunner-Munzel test"
        robust_columns = [
            "n_x",
            "n_y",
            "n_used",
            "relative_effect",
            "bm_stat",
            "df",
            "pval",
            "lower_conf",
            "upper_conf",
        ]

        def _robust_row(i: int) -> pd.Series:
            s = samples[i]
            n_x = s["x"].size
            n_y = s["y"].size
            if n_x < 2 or n_y < 2:
                raise ValueError(
                    f"needs at least 2 usable observations per group, got {n_x} and {n_y}."
                )
            row = sa_row(n_x=n_x, n_y=n_y, n_used=n_x + n_y)
            bm = sa_brunner_munzel(
                s["x"], s["y"], alternative=alt, conf_level=conf_level
            )
            return pd.concat([row, pd.Series(bm)])

        robust_result = sa_feature_table(
            feats, robust_columns, robust_label, _robust_row, p_adjust
        )
        robust_id = "brunner_munzel"
    else:
        robust_label = "Yuen's trimmed mean test for dependent samples"
        robust_columns = [
            "n_x",
            "n_y",
            "n_used",
            "x_trim_mean",
            "y_trim_mean",
            "trim_diff",
            "stderr",
            "yuen_stat",
            "df",
            "pval",
            "lower_conf",
            "upper_conf",
            "robust_dz",
        ]

        def _robust_row(i: int) -> pd.Series:
            s = samples[i]
            n_pairs = s["x"].size
            h = n_pairs - 2 * int(np.floor(tr * n_pairs))
            if h < 2:
                raise ValueError(
                    f"only {h} observation(s) survive trimming {tr} from each "
                    f"tail of {n_pairs} pair(s); 2 are needed."
                )
            row = sa_row(n_x=n_pairs, n_y=n_pairs, n_used=n_pairs)
            yuen = sa_yuen_paired(
                s["x"], s["y"], tr=tr, alternative=alt, conf_level=conf_level
            )
            return pd.concat([row, pd.Series(yuen)])

        robust_result = sa_feature_table(
            feats, robust_columns, robust_label, _robust_row, p_adjust
        )
        robust_id = "yuen_paired"

    per_feature_diag = [
        {group_lv[1]: s["x"], group_lv[0]: s["y"]} for s in samples
    ]
    diagnostics = (
        _diagnose_samples(per_feature_diag, feats, group_lv) if diagnose else None
    )

    return sa_new_comparison(
        analysis="two_group_comparison",
        features=feats,
        design={
            "group_lv": group_lv,
            "paired": paired,
            "pairing": (
                None
                if not paired
                else ("order" if id_arr is None else "id")
            ),
            "n_dropped": inp["n_dropped"],
            "unmatched_ids": unmatched,
        },
        parameters={
            "alternative": alternative,
            "conf_level": conf_level,
            "tr": tr if paired else np.nan,
            "fc_mean": fc_mean,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
        },
        effect=effect,
        tests={
            "t_test": t_result,
            "wilcox_test": w_result,
            "robust_test": robust_result,
        },
        test_info={
            "t_test": {
                "id": "paired_t_test" if paired else "welch_t_test",
                "label": t_label,
                "paired": paired,
            },
            "wilcox_test": {
                "id": "wilcoxon_signed_rank" if paired else "mann_whitney_u",
                "label": w_label,
                "paired": paired,
            },
            "robust_test": {
                "id": robust_id,
                "label": robust_label,
                "paired": paired,
            },
        },
        diagnostics=diagnostics,
        subclass=sa_two_group,
    )


def _t_conf_int_welch(
    x: np.ndarray, y: np.ndarray, alternative: str, conf_level: float
) -> tuple[float, float]:
    res = stats.ttest_ind(x, y, equal_var=False)
    df = float(res.df)
    diff = float(np.mean(x) - np.mean(y))
    se = float(res.stderr) if hasattr(res, "stderr") else np.nan
    alpha = 1.0 - conf_level
    if alternative == "two.sided":
        q = stats.t.ppf(1 - alpha / 2, df=df)
        return diff - q * se, diff + q * se
    if alternative == "greater":
        q = stats.t.ppf(1 - alpha, df=df)
        return diff - q * se, np.inf
    q = stats.t.ppf(1 - alpha, df=df)
    return -np.inf, diff + q * se


def _t_conf_int_paired(
    x: np.ndarray, y: np.ndarray, alternative: str, conf_level: float
) -> tuple[float, float]:
    d = x - y
    n = d.size
    diff = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    df = n - 1
    alpha = 1.0 - conf_level
    if alternative == "two.sided":
        q = stats.t.ppf(1 - alpha / 2, df=df)
        return diff - q * se, diff + q * se
    if alternative == "greater":
        q = stats.t.ppf(1 - alpha, df=df)
        return diff - q * se, np.inf
    q = stats.t.ppf(1 - alpha, df=df)
    return -np.inf, diff + q * se
