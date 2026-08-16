"""Run every applicable multi-group test at once."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.contracts.comparison import (
    sa_multi_group,
    sa_new_comparison,
)
from statassist.kernels.anova import (
    sa_friedman,
    sa_kruskal,
    sa_oneway_anova,
    sa_rm_anova,
    sa_welch_anova,
    sa_yuen_anova,
)
from statassist.kernels.posthoc import (
    sa_conover,
    sa_dunn,
    sa_games_howell,
    sa_pairwise_paired_t,
    sa_pairwise_yuen,
    sa_posthoc_columns,
    sa_tukey,
)
from statassist.utils.foldchange import (
    sa_group_centers,
    sa_multi_fold_change,
    sa_pairwise_tables,
    sa_resolve_fc_mean,
)
from statassist.utils.validate import (
    sa_align_by_subject,
    sa_check_flag,
    sa_check_p_adjust,
    sa_check_scalar_num,
    sa_control_first,
    sa_feature_table,
    sa_posthoc_table,
    sa_row,
    sa_validate_wide_input,
)


def _require_groups(samples: dict[str, np.ndarray], n_min: int) -> dict[str, np.ndarray]:
    sizes = {k: v.size for k, v in samples.items()}
    short = [k for k, n in sizes.items() if n < n_min]
    if short:
        detail = ", ".join(f"{k} = {sizes[k]}" for k in short)
        raise ValueError(
            f"needs at least {n_min} usable observation(s) per group; {detail}."
        )
    return samples


def _diagnose_samples(
    per_feature: list[Any],
    feats: list[str],
    group_lv: list[str],
    paired: bool,
) -> dict[str, Any]:
    from statassist.compare.diagnose_distribution import sa_diagnose_samples

    return sa_diagnose_samples(per_feature, feats, group_lv, paired)


def compare_multiple_groups(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: pd.Series | np.ndarray | list[Any],
    group_lv: list[str],
    *,
    control_label: str | None = None,
    id: list[str] | np.ndarray | None = None,
    paired: bool = False,
    conf_level: float = 0.95,
    tr: float = 0.2,
    posthoc: bool = True,
    posthoc_alpha: float = 0.05,
    fc_mean: str | None = None,
    input_scale: str = "raw",
    p_adjust: str = "BH",
    posthoc_p_adjust: str = "holm",
    diagnose: bool = True,
):
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)
    sa_check_flag(paired, "paired")
    sa_check_flag(posthoc, "posthoc")
    sa_check_flag(diagnose, "diagnose")
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    sa_check_scalar_num(tr, "tr", 0, 0.5, upper_open=True)
    sa_check_scalar_num(posthoc_alpha, "posthoc_alpha", 0, 1, lower_open=True)
    sa_check_p_adjust(p_adjust, "p_adjust")
    sa_check_p_adjust(posthoc_p_adjust, "posthoc_p_adjust")

    if paired and id is None:
        raise ValueError(
            "`paired = TRUE` needs `id` to say which rows belong to the same "
            "subject. Three or more conditions cannot be matched by row order."
        )
    if id is not None and not paired:
        warnings.warn(
            "`id` is only used to match repeated conditions and is ignored "
            "when `paired = FALSE`.",
            UserWarning,
            stacklevel=2,
        )

    if control_label is None:
        control_label = group_lv[0]

    inp = sa_validate_wide_input(
        data, feats, group, group_lv, id=id, min_levels=3
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

    if paired:
        aligned = sa_align_by_subject(id_arr, group, group_lv)
        unmatched = aligned["unmatched"]
        if unmatched:
            print(
                f"Dropped {len(unmatched)} subject(s) missing at least one condition."
            )
        per_feature: list[Any] = []
        for f in feats:
            mat = data[f].to_numpy()[aligned["idx"]]
            mat = pd.DataFrame(mat, index=aligned["subjects"], columns=group_lv)
            per_feature.append(mat.dropna(how="any"))
    else:
        unmatched = []
        group_arr = np.asarray(group)
        per_feature = []
        for f in feats:
            samples = {
                lv: data[f].to_numpy()[group_arr == lv]
                for lv in group_lv
            }
            per_feature.append(
                {lv: v[np.isfinite(v)] for lv, v in samples.items()}
            )

    centers = sa_group_centers(
        per_feature, feats, group_lv, fc_mean, paired, input_scale
    )
    effect = sa_multi_fold_change(centers, feats, group_lv, fc_mean)

    if paired:
        specs = {
            "anova_test": {
                "id": "repeated_measures_anova",
                "label": "Repeated measures ANOVA",
                "columns": [
                    "n_used", "n_groups", "f_stat", "df1", "df2",
                    "partial_eta_sq", "gen_eta_sq", "mauchly_w", "mauchly_pval",
                    "gg_eps", "pval_gg", "hf_eps", "pval_hf", "pval",
                    "lower_conf", "upper_conf",
                ],
                "posthoc_id": "pairwise_paired_t",
                "posthoc_label": "Pairwise paired t-tests",
                "posthoc_familywise": False,
            },
            "kruskal_test": {
                "id": "friedman_test",
                "label": "Friedman test",
                "columns": [
                    "n_used", "n_groups", "chi_sq", "df", "kendalls_w",
                    "pval", "lower_conf", "upper_conf",
                ],
                "posthoc_id": "conover_posthoc",
                "posthoc_label": "Conover post-hoc test",
                "posthoc_familywise": False,
            },
        }
    else:
        specs = {
            "anova_test": {
                "id": "oneway_anova",
                "label": "One-way ANOVA",
                "columns": [
                    "n_used", "n_groups", "f_stat", "df1", "df2", "eta_sq",
                    "omega_sq", "pval", "lower_conf", "upper_conf",
                ],
                "posthoc_id": "tukey_hsd",
                "posthoc_label": "Tukey HSD",
                "posthoc_familywise": True,
            },
            "welch_test": {
                "id": "welch_anova",
                "label": "Welch's one-way ANOVA",
                "columns": [
                    "n_used", "n_groups", "f_stat", "df1", "df2", "eta_sq",
                    "omega_sq", "pval", "lower_conf", "upper_conf",
                ],
                "posthoc_id": "games_howell",
                "posthoc_label": "Games-Howell post-hoc test",
                "posthoc_familywise": True,
            },
            "robust_test": {
                "id": "yuen_anova",
                "label": "Yuen's trimmed mean one-way ANOVA",
                "columns": [
                    "n_used", "n_groups", "f_stat", "df1", "df2",
                    "robust_eta_sq", "pval", "lower_conf", "upper_conf",
                ],
                "posthoc_id": "pairwise_yuen",
                "posthoc_label": "Pairwise Yuen tests",
                "posthoc_familywise": False,
            },
            "kruskal_test": {
                "id": "kruskal_wallis",
                "label": "Kruskal-Wallis test",
                "columns": [
                    "n_used", "n_groups", "h_stat", "df", "epsilon_sq",
                    "eta_sq_rank", "pval", "lower_conf", "upper_conf",
                ],
                "posthoc_id": "dunn_test",
                "posthoc_label": "Dunn's post-hoc test",
                "posthoc_familywise": False,
            },
        }

    tests: dict[str, pd.DataFrame] = {}

    def _omnibus_anova(i: int) -> pd.Series:
        return sa_row(**sa_oneway_anova(_require_groups(per_feature[i], 2)))

    def _omnibus_welch(i: int) -> pd.Series:
        return sa_row(**sa_welch_anova(_require_groups(per_feature[i], 2)))

    def _omnibus_yuen(i: int) -> pd.Series:
        return sa_row(**sa_yuen_anova(_require_groups(per_feature[i], 2), tr))

    def _omnibus_kruskal(i: int) -> pd.Series:
        return sa_row(**sa_kruskal(_require_groups(per_feature[i], 1)))

    def _omnibus_rm(i: int) -> pd.Series:
        return sa_row(**sa_rm_anova(per_feature[i].to_numpy()))

    def _omnibus_friedman(i: int) -> pd.Series:
        return sa_row(**sa_friedman(per_feature[i].to_numpy()))

    omnibus_fns: dict[str, Any] = {}
    posthoc_fns: dict[str, Any] = {}

    if paired:
        omnibus_fns = {
            "anova_test": _omnibus_rm,
            "kruskal_test": _omnibus_friedman,
        }
        posthoc_fns = {
            "anova_test": lambda f: sa_pairwise_paired_t(per_feature[feats.index(f)], conf_level),
            "kruskal_test": lambda f: sa_conover(per_feature[feats.index(f)], conf_level),
        }
    else:
        omnibus_fns = {
            "anova_test": _omnibus_anova,
            "welch_test": _omnibus_welch,
            "robust_test": _omnibus_yuen,
            "kruskal_test": _omnibus_kruskal,
        }
        posthoc_fns = {
            "anova_test": lambda f: sa_tukey(per_feature[feats.index(f)], conf_level),
            "welch_test": lambda f: sa_games_howell(per_feature[feats.index(f)], conf_level),
            "robust_test": lambda f: sa_pairwise_yuen(per_feature[feats.index(f)], tr, conf_level),
            "kruskal_test": lambda f: sa_dunn(per_feature[feats.index(f)], conf_level),
        }

    for nm, spec in specs.items():
        tests[nm] = sa_feature_table(
            feats, spec["columns"], spec["label"], omnibus_fns[nm], p_adjust
        )

    posthoc_tables: dict[str, pd.DataFrame] = {}
    pairwise_tables: dict[str, dict[str, pd.DataFrame]] = {}
    n_posthoc: dict[str, int] = {nm: 0 for nm in specs}
    posthoc_cols = sa_posthoc_columns()

    if posthoc:
        for nm, spec in specs.items():
            padj = tests[nm]["pval_adj"]
            qualified = [
                f for f, p in zip(feats, padj) if pd.notna(p) and p <= posthoc_alpha
            ]
            n_posthoc[nm] = len(qualified)
            adj = "none" if spec["posthoc_familywise"] else posthoc_p_adjust
            posthoc_tables[nm] = sa_posthoc_table(
                qualified,
                group_lv,
                posthoc_cols,
                spec["posthoc_label"],
                posthoc_fns[nm],
                p_adjust_method=adj,
            )
            pairwise_tables[nm] = sa_pairwise_tables(
                posthoc_tables[nm], centers["centers"], feats, group_lv
            )

    diagnostics = (
        _diagnose_samples(per_feature, feats, group_lv, paired) if diagnose else None
    )

    return sa_new_comparison(
        analysis="multi_group_comparison",
        features=feats,
        design={
            "group_lv": group_lv,
            "paired": paired,
            "pairing": "id" if paired else None,
            "n_dropped": inp["n_dropped"],
            "unmatched_ids": unmatched,
        },
        parameters={
            "alternative": "two.sided",
            "conf_level": conf_level,
            "tr": np.nan if paired else tr,
            "fc_mean": fc_mean,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
            "posthoc": posthoc,
            "posthoc_alpha": posthoc_alpha,
            "posthoc_p_adjust": posthoc_p_adjust,
            "n_posthoc": n_posthoc,
        },
        effect=effect,
        tests=tests,
        posthoc=posthoc_tables if posthoc else None,
        pairwise=pairwise_tables if posthoc else None,
        test_info={
            nm: {
                "id": spec["id"],
                "label": spec["label"],
                "paired": paired,
                "posthoc_id": spec["posthoc_id"],
                "posthoc_label": spec["posthoc_label"],
            }
            for nm, spec in specs.items()
        },
        diagnostics=diagnostics,
        subclass=sa_multi_group,
    )
