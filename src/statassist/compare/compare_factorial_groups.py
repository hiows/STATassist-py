"""Analyse a crossed-factor design as one model (R compare_factorial_groups.R)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.contracts.comparison import (
    sa_cell_table_columns,
    sa_factorial,
    sa_new_comparison,
    sa_posthoc_table_columns,
)
from statassist.kernels.factorial import (
    sa_factorial_anova,
    sa_factorial_plan,
    sa_factorial_tukey,
)
from statassist.kernels.posthoc import sa_posthoc_columns
from statassist.utils.factorial_utils import (
    sa_fact_cell_index,
    sa_fact_cell_labels,
    sa_fact_contrast_skeleton,
    sa_fact_control_first,
    sa_fact_grid,
    sa_fact_term_effect,
)
from statassist.utils.foldchange import (
    sa_group_centers,
    sa_multi_fold_change,
    sa_resolve_fc_mean,
)
from statassist.utils.validate import (
    p_adjust as adjust_pvalues,
    sa_check_flag,
    sa_check_p_adjust,
    sa_check_scalar_num,
    sa_feature_table,
    sa_resolve_row_vector,
    sa_validate_wide_input,
)

_WHOLE_COLUMNS = [
    "n_used",
    "n_cells",
    "f_stat",
    "df1",
    "df2",
    "eta_sq",
    "omega_sq",
    "pval",
    "lower_conf",
    "upper_conf",
]

_TERM_STAT_COLUMNS = [
    "n_used",
    "df",
    "ss",
    "ms",
    "f_stat",
    "df_error",
    "eta_sq",
    "partial_eta_sq",
    "pval",
]


def sa_fact_anova_type(n_factors: int) -> str:
    return {2: "two_way", 3: "three_way"}.get(n_factors, "factorial")


def sa_fact_anova_label(anova_type: str, ss_type: str) -> str:
    name = {
        "two_way": "Two-way ANOVA",
        "three_way": "Three-way ANOVA",
    }.get(anova_type, "Factorial ANOVA")
    return f"{name} (Type {ss_type} sums of squares)"


def sa_fact_layout(
    data: pd.DataFrame,
    factors: dict[str, Any],
    factor_lv: dict[str, list[str]] | None,
    control_label: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the crossed factors and lay the observations out in cells.

    Everything the analysis needs to know about where an observation sits,
    settled in one pass so that the levels, the cells and the row selections
    cannot be derived from each other twice in different orders.
    """
    if (
        not isinstance(factors, dict)
        or len(factors) < 2
        or any(nm is None or nm == "" for nm in factors)
    ):
        raise ValueError(
            "`factors` must be a named list of at least two crossed factors, "
            "each entry a column name of `data` or one value per row of it. "
            "Use compare_multiple_groups() for a single factor."
        )

    # A factor sharing a name with one of the statistics columns would silently
    # overwrite it, so the collision is refused here rather than discovered in
    # the result.
    taken = [nm for nm in factors if nm in sa_cell_table_columns()]
    if taken:
        raise ValueError(
            "`factors` may not name a factor after a column of the cell table: "
            f"{', '.join(taken)}. Reserved: "
            f"{', '.join(sa_cell_table_columns())}. Rename the factor."
        )

    values: dict[str, np.ndarray] = {}
    for nm, src in factors.items():
        resolved = sa_resolve_row_vector(src, f"factors${nm}", data, allow_na=True)
        values[nm] = pd.Series(resolved["value"]).astype(object).to_numpy()

    named_lv = factor_lv is not None
    if not named_lv:
        factor_lv = {
            nm: sorted({str(x) for x in v if not pd.isna(x)})
            for nm, v in values.items()
        }
    else:
        if not isinstance(factor_lv, dict) or set(factor_lv) != set(factors):
            raise ValueError(
                "`factor_lv` must be a named list holding the levels of every "
                "factor `factors` names, or None to take them from the data. "
                f"`factors` has: {', '.join(factors)}."
            )
        # Declaration order is the term order and the Type I order, and it is
        # `factor_lv` that states it when both lists are given.
        values = {nm: values[nm] for nm in factor_lv}

    resolved_lv: dict[str, list[str]] = {}
    for nm in factor_lv:
        lv = [str(x) for x in factor_lv[nm]]
        if len(lv) < 2 or any(x == "" or x == "nan" for x in lv) or len(set(lv)) != len(lv):
            raise ValueError(
                f"`factor_lv${nm}` must be at least two distinct non-empty "
                "level names, the first being the reference."
            )
        present = {str(x) for x in values[nm] if not pd.isna(x)}
        absent = [x for x in lv if x not in present]
        if absent:
            raise ValueError(
                f"`factor_lv${nm}` level(s) absent from `factors${nm}`: "
                f"{', '.join(absent)}."
            )
        resolved_lv[nm] = lv

    # After the levels are settled and before anything is counted from them, so
    # that the grid, the cell labels and the row selections are all built from
    # the order the reference ended up in.
    resolved_lv = sa_fact_control_first(
        resolved_lv, control_label, "factor_lv" if named_lv else "factors"
    )

    n_rows = len(data)
    level_idx = np.zeros((n_rows, len(resolved_lv)), dtype=float)
    for j, nm in enumerate(resolved_lv):
        lookup = {lv: i + 1 for i, lv in enumerate(resolved_lv[nm])}
        level_idx[:, j] = [
            lookup.get(str(x), np.nan) if not pd.isna(x) else np.nan
            for x in values[nm]
        ]
    keep = np.all(np.isfinite(level_idx), axis=1)

    cells = sa_fact_grid(resolved_lv)
    n_cells = len(cells)
    dims = [len(resolved_lv[nm]) for nm in resolved_lv]
    cell_idx = sa_fact_cell_index(level_idx[keep].astype(int), dims)
    kept_rows = np.flatnonzero(keep)
    rows_of_cell = [kept_rows[cell_idx == c + 1] for c in range(n_cells)]

    return {
        "factor_lv": resolved_lv,
        "cells": cells,
        "n_cells": n_cells,
        "cell_label": sa_fact_cell_labels(resolved_lv, cells),
        "cell_n": [int(r.size) for r in rows_of_cell],
        "rows_of_cell": rows_of_cell,
        "n_empty_cells": int(sum(1 for r in rows_of_cell if r.size == 0)),
        "n_dropped": int((~keep).sum()),
        "anova_type": sa_fact_anova_type(len(resolved_lv)),
    }


def sa_fact_effect(
    centers: dict[str, Any],
    feats: list[str],
    cell_label: list[str],
    mean_type: str,
) -> pd.DataFrame:
    """Fold change of the most extreme cell against the reference cell."""
    out = sa_multi_fold_change(centers, feats, cell_label, mean_type)
    out = out.rename(
        columns={"n_groups": "n_cells", "extreme_level": "extreme_cell"}
    )
    return out[
        [
            "features",
            "n_used",
            "n_cells",
            "ref_center",
            "extreme_cell",
            "extreme_center",
            "fold_change",
            "log2fc",
        ]
    ]


def sa_fact_cell_table(
    feats: list[str],
    per_feature: list[dict[str, np.ndarray]],
    fits: list[dict[str, Any] | None],
    design: dict[str, Any],
) -> pd.DataFrame:
    """Stack the cell means into one feature by cell table.

    The means are recomputed from the samples rather than taken from the fits,
    so a feature whose model could not be fitted still reports what its cells
    held. `se` is the one column that needs the fit, because it is pooled over
    the whole model.
    """
    n_cells = design["n_cells"]
    cells = design["cells"]
    factor_lv = design["factor_lv"]

    blocks = []
    for i, f in enumerate(feats):
        block = {"features": [f] * n_cells}
        for nm in factor_lv:
            block[nm] = [
                factor_lv[nm][int(cells.iloc[c][nm]) - 1] for c in range(n_cells)
            ]
        block["cell"] = list(design["cell_label"])
        samples = [per_feature[i][lbl] for lbl in design["cell_label"]]
        block["n"] = [int(s.size) for s in samples]
        block["mean"] = [float(np.mean(s)) if s.size else np.nan for s in samples]
        block["sd"] = [
            float(np.std(s, ddof=1)) if s.size > 1 else np.nan for s in samples
        ]
        if fits[i] is None:
            block["se"] = [np.nan] * n_cells
        else:
            block["se"] = [
                float(np.sqrt(fits[i]["ms_error"] / s.size)) if s.size else np.nan
                for s in samples
            ]
        blocks.append(pd.DataFrame(block))

    return pd.concat(blocks, ignore_index=True)


def sa_fact_term_table(
    feats: list[str],
    plan: dict[str, Any],
    fits: list[dict[str, Any] | None],
    p_adjust: str,
    centers: dict[str, Any],
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Stack the per-feature term results into one feature by term table."""
    labels = plan["labels"]
    n_terms = len(labels)

    stat_blocks = []
    for fit in fits:
        if fit is None:
            stat_blocks.append(
                pd.DataFrame(
                    np.full((n_terms, len(_TERM_STAT_COLUMNS)), np.nan),
                    columns=_TERM_STAT_COLUMNS,
                )
            )
        else:
            stat_blocks.append(
                fit["terms"][_TERM_STAT_COLUMNS].reset_index(drop=True)
            )

    out = pd.DataFrame(
        {
            "features": np.repeat(feats, n_terms),
            "terms": labels * len(feats),
            "term_order": plan["orders"] * len(feats),
        }
    )
    out = pd.concat([out, pd.concat(stat_blocks, ignore_index=True)], axis=1)

    # The effect size of a term, on the log2 scale the centres put it on. Taken
    # from the same centre matrix `effect` is, so the two are the same numbers
    # read at two grains. Components are deviations from what the other terms
    # predict, so they do not depend on which cell is the reference.
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_center = np.log2(np.asarray(centers["centers"], dtype=float))
    out["log2_effect"] = np.concatenate(
        [sa_fact_term_effect(log2_center[i], cells, plan["terms"]) for i in range(len(feats))]
    )

    # One term is one family. The main effect and the interaction are two
    # different questions, and pooling them would correct each for the other's
    # multiplicity.
    out["pval_adj"] = np.nan
    for nm in labels:
        at = out["terms"] == nm
        out.loc[at, "pval_adj"] = adjust_pvalues(
            out.loc[at, "pval"].to_numpy(dtype=float), p_adjust
        )

    return out[
        [
            "features",
            "terms",
            "term_order",
            "n_used",
            "df",
            "ss",
            "ms",
            "f_stat",
            "df_error",
            "eta_sq",
            "partial_eta_sq",
            "log2_effect",
            "pval",
            "pval_adj",
        ]
    ]


def sa_fact_posthoc_stage(
    feats: list[str],
    fits: list[dict[str, Any] | None],
    terms_tbl: pd.DataFrame,
    design: dict[str, Any],
    scope: str,
    alpha: float,
    conf_level: float,
) -> dict[str, Any]:
    """Run the contrasts each feature's term tests earned it.

    The pairwise stage of a factorial model is ragged on both axes: a feature
    has contrasts only for the terms it cleared, so the marginal comparisons of
    one factor can be present while another factor's are absent for the same
    feature.
    """
    factor_lv = design["factor_lv"]
    skeleton = sa_fact_contrast_skeleton(design)
    rows = skeleton["table"]
    columns = ["features", "factor", "stratum"] + [
        c for c in sa_posthoc_table_columns() if c != "features"
    ]

    is_marginal = rows["stratum"].isna().to_numpy()
    if scope == "both":
        wanted = np.ones(len(rows), dtype=bool)
    elif scope == "marginal":
        wanted = is_marginal
    else:
        wanted = ~is_marginal
    candidates = np.flatnonzero(wanted)

    # The term that licenses a contrast: a factor's own main effect for a
    # marginal comparison, and its interaction with everything held fixed for a
    # simple one. In `factor_lv` order, which is the term label order.
    names = list(factor_lv)
    gate = []
    for k in candidates:
        used = [rows["factor"].iloc[k]] if is_marginal[k] else names
        gate.append(":".join(nm for nm in names if nm in used))
    nmeans = np.array(
        [len(factor_lv[f]) for f in rows["factor"]], dtype=int
    )

    blocks: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for i, f in enumerate(feats):
        if fits[i] is None:
            continue
        block = terms_tbl.loc[terms_tbl["features"] == f]
        padj = dict(zip(block["terms"], block["pval_adj"]))
        earned = np.array([padj.get(g, np.nan) for g in gate], dtype=float)
        take = candidates[np.isfinite(earned) & (earned <= alpha)]
        if take.size == 0:
            continue
        try:
            mat = sa_factorial_tukey(fits[i], skeleton, nmeans, take, conf_level)
        except Exception as exc:  # noqa: BLE001 - reported as a grouped warning
            failures[f] = str(exc)
            mat = np.full((take.size, len(sa_posthoc_columns())), np.nan)

        stats_df = pd.DataFrame(mat, columns=sa_posthoc_columns())
        # The studentised range controls the error rate over the block of
        # contrasts this one sits in, so adjusting it again would correct twice
        # for one comparison.
        stats_df["pval_adj"] = stats_df["pval"]
        label_df = rows.iloc[take][
            ["factor", "stratum", "contrast", "group1", "group2"]
        ].reset_index(drop=True)
        blocks.append(
            pd.concat(
                [
                    pd.DataFrame({"features": [f] * take.size}),
                    label_df,
                    stats_df,
                ],
                axis=1,
            )
        )

    if failures:
        detail = "\n".join(f"  {k}: {v}" for k, v in failures.items())
        warnings.warn(
            "Tukey HSD on marginal means and simple effects could not be "
            f"computed for {len(failures)} of {len(feats)} feature(s); those "
            f"rows are NA:\n{detail}",
            RuntimeWarning,
            stacklevel=3,
        )

    if not blocks:
        empty = pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
        return {"table": empty, "n_posthoc": 0}

    out = pd.concat(blocks, ignore_index=True)
    return {"table": out[columns], "n_posthoc": len(blocks)}


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
    posthoc_scope: str = "both",
    fc_mean: str | None = None,
    input_scale: str = "raw",
    p_adjust: str = "BH",
    diagnose: bool = True,
):
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)
    if ss_type not in ("III", "II", "I"):
        raise ValueError('`ss_type` must be one of "III", "II", or "I".')
    if posthoc_scope not in ("both", "marginal", "simple"):
        raise ValueError(
            '`posthoc_scope` must be one of "both", "marginal", or "simple".'
        )
    sa_check_flag(posthoc, "posthoc")
    sa_check_flag(diagnose, "diagnose")
    sa_check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    sa_check_scalar_num(posthoc_alpha, "posthoc_alpha", 0, 1, lower_open=True)
    sa_check_p_adjust(p_adjust, "p_adjust")

    if within:
        joined = ", ".join(within)
        raise ValueError(
            "`within` names factor(s) measured within subjects, which this "
            f"version cannot analyse: {joined}. A mixed factorial model needs a "
            "second error stratum, which is not implemented yet. Analyse the "
            "between-subject factors on one level of the within one, or use "
            "compare_multiple_groups(paired=True) on the repeated factor alone."
        )
    if id is not None:
        warnings.warn(
            "`id` is only used to match repeated measurements and is ignored by "
            "a between-subject factorial analysis.",
            UserWarning,
            stacklevel=2,
        )

    input_ = sa_validate_wide_input(data, feats, group=None, group_lv=None)
    data = input_["data"]
    feats = input_["feats"]

    design = sa_fact_layout(data, factors, factor_lv, control_label)
    if design["n_empty_cells"] > 0:
        empty = ", ".join(
            lbl
            for lbl, n in zip(design["cell_label"], design["cell_n"])
            if n == 0
        )
        warnings.warn(
            f"{design['n_empty_cells']} of {design['n_cells']} cell(s) hold no "
            "observation, so no crossed model can be fitted on them: "
            f"{empty}.",
            RuntimeWarning,
            stacklevel=2,
        )

    per_feature: list[dict[str, np.ndarray]] = []
    for f in feats:
        v = data[f].to_numpy(dtype=float)
        per_feature.append(
            {
                lbl: v[at][np.isfinite(v[at])]
                for lbl, at in zip(design["cell_label"], design["rows_of_cell"])
            }
        )

    centers = sa_group_centers(
        per_feature,
        feats,
        design["cell_label"],
        fc_mean,
        paired=False,
        input_scale=input_scale,
    )
    effect = sa_fact_effect(centers, feats, design["cell_label"], fc_mean)

    plan = sa_factorial_plan(design["factor_lv"], design["cells"], ss_type)
    label = sa_fact_anova_label(design["anova_type"], ss_type)

    # One fit per feature, both axes of it. The failures are held rather than
    # raised so that they become an NA row and one grouped warning for the
    # whole scan.
    fits: list[dict[str, Any] | None] = []
    errors: list[str | None] = []
    for i in range(len(feats)):
        try:
            fits.append(sa_factorial_anova(per_feature[i], plan))
            errors.append(None)
        except Exception as exc:  # noqa: BLE001 - re-raised inside sa_feature_table
            fits.append(None)
            errors.append(str(exc))

    def whole_row(i: int) -> pd.Series:
        if errors[i] is not None:
            raise ValueError(errors[i])
        return pd.Series(fits[i]["model"])

    tests = {
        "anova_test": sa_feature_table(
            feats, _WHOLE_COLUMNS, label, whole_row, p_adjust_method=p_adjust
        )
    }

    terms_tbl = sa_fact_term_table(
        feats, plan, fits, p_adjust, centers, design["cells"]
    )
    cells_tbl = sa_fact_cell_table(feats, per_feature, fits, design)

    posthoc_tables: dict[str, pd.DataFrame] = {}
    n_posthoc = 0
    if posthoc:
        stage = sa_fact_posthoc_stage(
            feats, fits, terms_tbl, design, posthoc_scope, posthoc_alpha, conf_level
        )
        posthoc_tables["anova_test"] = stage["table"]
        n_posthoc = stage["n_posthoc"]

    diagnostics = None
    if diagnose:
        from statassist.compare.diagnose_distribution import sa_diagnose_samples

        diagnostics = sa_diagnose_samples(
            per_feature, feats, design["cell_label"], paired=False
        )

    return sa_new_comparison(
        analysis="factorial_comparison",
        features=feats,
        design={
            "factor_lv": design["factor_lv"],
            "anova_type": design["anova_type"],
            "n_factors": len(design["factor_lv"]),
            "group_lv": design["cell_label"],
            "cell_n": design["cell_n"],
            "n_empty_cells": design["n_empty_cells"],
            "paired": False,
            "pairing": None,
            "within": [],
            "n_dropped": design["n_dropped"],
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
            # Every contrast is judged against the studentised range of its own
            # block, so its p-value is family-wise already.
            "posthoc_p_adjust": None,
            "posthoc_scope": posthoc_scope,
            "n_posthoc": n_posthoc,
        },
        effect=effect,
        tests=tests,
        terms=terms_tbl,
        cells=cells_tbl,
        posthoc=posthoc_tables or None,
        test_info={
            "anova_test": {
                "id": "factorial_anova",
                "label": label,
                "paired": False,
                "posthoc_id": "factorial_tukey",
                "posthoc_label": "Tukey HSD on marginal means and simple effects",
            }
        },
        diagnostics=diagnostics,
        subclass=sa_factorial,
    )
