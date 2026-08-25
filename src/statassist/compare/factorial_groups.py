"""Analyse a crossed-factor design as one model.

Port of ``R/compare_factorial_groups.R``. The factorial counterpart of
:func:`~statassist.compare_multiple_groups`, and the one place in the package
where a scenario function fits **one** model rather than running every applicable
test side by side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.contracts import posthoc_table_columns
from ..core.errors import SaValueError, notify, warn
from ..core.factorial import (
    FactLayout,
    fact_contrast_skeleton,
    fact_layout,
    fact_term_effect,
)
from ..core.padjust import p_adjust
from ..core.result import SaComparison, new_comparison
from ..core.tables import feature_table
from ..core.validate import (
    UNSET,
    check_flag,
    check_p_adjust,
    check_scalar_num,
    validate_wide_input,
)
from ..diagnose.distribution import diagnose_samples
from ..kernel.factorial import (
    SS_TYPES,
    FactorialFit,
    FactorialPlan,
    factorial_anova,
    factorial_plan,
    factorial_tukey,
)
from ..kernel.posthoc import posthoc_columns
from ..transform._foldchange import INPUT_SCALES, resolve_fc_mean
from ._multi import Centers, group_centers, multi_fold_change

__all__ = ["compare_factorial_groups"]

#: Views the pairwise stage of a factorial model can cover.
POSTHOC_SCOPES: tuple[str, ...] = ("both", "marginal", "simple")


def compare_factorial_groups(
    data: Any,
    feats: Any,
    factors: Any,
    factor_lv: Any = None,
    control_label: Any = None,
    within: Any = None,
    id: Any = None,  # noqa: A002 - matches the R argument name
    conf_level: float = 0.95,
    ss_type: str = "III",
    posthoc: bool = True,
    posthoc_alpha: float = 0.05,
    posthoc_scope: str = "both",
    fc_mean: Any = UNSET,
    input_scale: str = INPUT_SCALES[0],
    p_adjust: str = "BH",
    diagnose: bool = True,
) -> SaComparison:
    """Analyse a crossed-factor design as one model.

    Two crossed factors are a two-way ANOVA, three a three-way ANOVA and more than
    three a factorial ANOVA: three names for the same fully crossed linear model.
    Which name applies follows from ``len(factor_lv)`` and is reported in
    ``design["anova_type"]``.

    A single p-value cannot say which part of a crossed design a feature responds
    to, so the result carries ``terms`` (one row per feature and model term) and
    ``cells`` (one row per feature and cell) beside the whole-model
    ``tests["anova_test"]``. There is no ``pairwise`` slot: contrasts live only in
    ``posthoc``, keyed by factor and stratum.

    Args:
        data: Wide frame, one row per observation and one column per feature.
        feats: Names of the numeric columns to test.
        factors: Named mapping of the crossed factors, each entry either a column
            name of ``data`` or one value per row. At least two factors; the first
            is primary and fixes Type I term order.
        factor_lv: Levels per factor with the reference first, or ``None`` to take
            sorted unique values from the data.
        control_label: Per-factor reference level, as a named mapping.
        within: Within-subject factor names. Not implemented: a non-empty value
            is an error.
        id: Subject identifier. Ignored with a warning in a between-subject
            analysis.
        conf_level: Confidence level for post-hoc intervals.
        ss_type: ``"III"``, ``"II"`` or ``"I"`` sums of squares.
        posthoc: If ``False``, no pairwise stage runs and ``posthoc`` is omitted.
        posthoc_alpha: A contrast runs when its gating term's ``pval_adj`` is at
            or below this value.
        posthoc_scope: ``"marginal"``, ``"simple"`` or ``"both"``.
        fc_mean: ``"arith"`` or ``"geom"``. Defaults to ``"geom"`` when
            ``input_scale="log2"``.
        input_scale: ``"raw"`` or ``"log2"``. Affects ``effect`` only, never the
            tests.
        p_adjust: Multiplicity adjustment along the feature axis (within each
            term for ``terms``).
        diagnose: If ``True``, attach cell-wise assumption checks.

    Returns:
        A :class:`~statassist.core.result.SaComparison` with subclass
        ``sa_factorial``, analysis ``"factorial_comparison"``.

    Raises:
        SaValueError: If arguments are invalid or ``within`` is non-empty.

    Notes:
        ``terms["log2_effect"]`` is not a fold change between two levels. It is
        the largest ANOVA component of that term (a deviation from what the rest
        of the model already predicts), with its sign. For a two-level factor
        whose levels differ by one log2 unit the components are ``-0.5`` and
        ``+0.5``, so the reported effect is half the marginal fold change on the
        log2 scale. Omnibus ``effect["log2fc"]`` (extreme cell against the
        reference) is the quantity that answers an up/down fold-change question;
        the term column does not.

        Because a two-level term's components tie in absolute value by
        construction, values within ``FACT_TOL`` of the largest absolute
        component are treated as a tie and the earlier (reference) cell is kept.
        Absolute size, p-values and ``is_signif`` stay aligned across languages;
        CRAN R still uses a bare ``which.max`` and may pick the opposite sign of
        the same magnitude until the same near-tie rule lands there. Do not read
        that sign as a treatment-versus-control direction.
    """
    if input_scale not in INPUT_SCALES:
        raise SaValueError("`input_scale` must be one of: " + ", ".join(INPUT_SCALES) + ".")
    mean_type = resolve_fc_mean(fc_mean, input_scale)
    if ss_type not in SS_TYPES:
        raise SaValueError("`ss_type` must be one of: " + ", ".join(SS_TYPES) + ".")
    if posthoc_scope not in POSTHOC_SCOPES:
        raise SaValueError("`posthoc_scope` must be one of: " + ", ".join(POSTHOC_SCOPES) + ".")
    posthoc = check_flag(posthoc, "posthoc")
    diagnose = check_flag(diagnose, "diagnose")
    conf_level = check_scalar_num(conf_level, "conf_level", 0, 1, lower_open=True, upper_open=True)
    posthoc_alpha = check_scalar_num(posthoc_alpha, "posthoc_alpha", 0, 1, lower_open=True)
    p_adjust = check_p_adjust(p_adjust, "p_adjust")

    if within is not None:
        within_names = [within] if isinstance(within, str) else [str(name) for name in within]
        if within_names:
            raise SaValueError(
                "`within` names factor(s) measured within subjects, which this "
                "version cannot analyse: "
                + ", ".join(within_names)
                + ". A mixed factorial model needs a second error stratum, which is "
                "not implemented yet. Analyse the between-subject factors on one "
                "level of the within one, or use compare_multiple_groups(paired=True) "
                "on the repeated factor alone."
            )
    if id is not None:
        warn(
            "`id` is only used to match repeated measurements and is ignored "
            "by a between-subject factorial analysis."
        )

    validated = validate_wide_input(data, feats, group=None, group_lv=None)
    frame = validated.data
    names = list(validated.feats)

    layout = fact_layout(frame, factors, factor_lv, control_label)
    if layout.n_dropped > 0:
        notify(f"Dropped {layout.n_dropped} row(s) belonging to a level outside `factor_lv`.")
    if layout.n_empty_cells > 0:
        empty = [
            label
            for label, count in zip(layout.cell_label, layout.cell_n, strict=True)
            if count == 0
        ]
        notify(
            f"{layout.n_empty_cells} of {layout.n_cells} cell(s) hold no observation, "
            "so no crossed model can be fitted on them: " + ", ".join(empty) + "."
        )

    per_feature: dict[str, dict[str, np.ndarray]] = {}
    for feature in names:
        column = frame[feature].to_numpy(dtype=float)
        samples: dict[str, np.ndarray] = {}
        for label, rows in zip(layout.cell_label, layout.rows_of_cell, strict=True):
            held = column[rows]
            samples[label] = held[~np.isnan(held)]
        per_feature[feature] = samples

    centers = group_centers(
        per_feature, names, layout.cell_label, mean_type, paired=False, input_scale=input_scale
    )
    effect = _fact_effect(centers, names, layout.cell_label, mean_type)

    plan = factorial_plan(layout.factor_lv, layout.cells, ss_type)
    label = _fact_anova_label(layout.anova_type, ss_type)

    fits: list[FactorialFit | None] = []
    errors: list[str | None] = []
    for feature in names:
        try:
            fits.append(factorial_anova(per_feature[feature], plan))
            errors.append(None)
        except Exception as error:  # noqa: BLE001 - held for feature_table
            fits.append(None)
            errors.append(str(error))

    tests = {
        "anova_test": feature_table(
            names,
            [
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
            ],
            label,
            fun=lambda index: _model_row(fits[index], errors[index]),
            p_adjust_method=p_adjust,
        )
    }

    terms_tbl = _fact_term_table(names, plan, fits, p_adjust, centers, layout.cells)
    cells_tbl = _fact_cell_table(names, per_feature, fits, layout)

    posthoc_tables: dict[str, pd.DataFrame] = {}
    n_posthoc = 0
    if posthoc:
        stage = _fact_posthoc_stage(
            names, fits, terms_tbl, layout, posthoc_scope, posthoc_alpha, conf_level
        )
        posthoc_tables["anova_test"] = stage["table"]
        n_posthoc = int(stage["n_posthoc"])

    diagnostics = (
        diagnose_samples(per_feature, names, layout.cell_label, paired=False) if diagnose else None
    )

    return new_comparison(
        analysis="factorial_comparison",
        features=names,
        design={
            "factor_lv": layout.factor_lv,
            "anova_type": layout.anova_type,
            "n_factors": len(layout.factor_lv),
            "group_lv": layout.cell_label,
            "cell_n": layout.cell_n,
            "n_empty_cells": layout.n_empty_cells,
            "paired": False,
            "pairing": None,
            "within": [],
            "n_dropped": layout.n_dropped,
            "unmatched_ids": [],
        },
        parameters={
            "alternative": "two.sided",
            "conf_level": conf_level,
            "ss_type": ss_type,
            "fc_mean": mean_type,
            "input_scale": input_scale,
            "p_adjust": p_adjust,
            "posthoc": posthoc,
            "posthoc_alpha": posthoc_alpha,
            # Studentised-range p-values are already family-wise within a block.
            "posthoc_p_adjust": None,
            "posthoc_scope": posthoc_scope,
            "n_posthoc": n_posthoc,
        },
        effect=effect,
        tests=tests,
        terms=terms_tbl,
        cells=cells_tbl,
        posthoc=posthoc_tables,
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
        subclass="sa_factorial",
    )


def _model_row(fit: FactorialFit | None, error: str | None) -> dict[str, float]:
    """Whole-model row, or re-raise the held failure inside ``feature_table``."""
    if error is not None or fit is None:
        raise SaValueError(error or "factorial ANOVA failed.")
    return dict(fit.model)


def _fact_anova_label(anova_type: str, ss_type: str) -> str:
    """The readable name of the analysis that ran.

    Port of ``sa_fact_anova_label()``.
    """
    titles = {
        "two_way": "Two-way ANOVA",
        "three_way": "Three-way ANOVA",
        "factorial": "Factorial ANOVA",
    }
    return f"{titles[anova_type]} (Type {ss_type} sums of squares)"


def _fact_effect(
    centers: Centers,
    feats: Sequence[str],
    cell_label: Sequence[str],
    mean_type: str,
) -> pd.DataFrame:
    """Fold change of the most extreme cell against the reference cell.

    Port of ``sa_fact_effect()``: ``multi_fold_change`` with the cells as levels,
    then rename ``n_groups`` / ``extreme_level`` to the cell wording.
    """
    out = multi_fold_change(centers, feats, cell_label, mean_type)
    out = out.rename(columns={"n_groups": "n_cells", "extreme_level": "extreme_cell"})
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


def _fact_cell_table(
    feats: Sequence[str],
    per_feature: Mapping[str, Mapping[str, np.ndarray]],
    fits: Sequence[FactorialFit | None],
    layout: FactLayout,
) -> pd.DataFrame:
    """Stack the cell means into one feature-by-cell table.

    Port of ``sa_fact_cell_table()``.
    """
    n_cells = layout.n_cells
    n_feats = len(feats)
    rows: list[dict[str, Any]] = []

    for feature, fit in zip(feats, fits, strict=True):
        samples = per_feature[feature]
        for cell, label in enumerate(layout.cell_label):
            held = samples[label]
            n_used = int(held.size)
            mean = float(np.mean(held)) if n_used > 0 else float("nan")
            sd = float(np.std(held, ddof=1)) if n_used >= 2 else float("nan")
            if fit is None or n_used == 0:
                se = float("nan")
            else:
                se = float(np.sqrt(fit.ms_error / n_used))
            row: dict[str, Any] = {"features": feature}
            for name in layout.factor_lv:
                row[name] = layout.factor_lv[name][int(layout.cells[name].iloc[cell])]
            row.update({"cell": label, "n": n_used, "mean": mean, "sd": sd, "se": se})
            rows.append(row)

    out = pd.DataFrame(rows)
    # Contract columns first, then the per-factor level columns in declaration
    # order between features and cell, matching R's column layout.
    factor_cols = list(layout.factor_lv)
    ordered = ["features", *factor_cols, "cell", "n", "mean", "sd", "se"]
    assert len(out.index) == n_feats * n_cells
    return out[ordered]


def _fact_term_table(
    feats: Sequence[str],
    plan: FactorialPlan,
    fits: Sequence[FactorialFit | None],
    adjust: str,
    centers: Centers,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Stack the per-feature term results into one feature-by-term table.

    Port of ``sa_fact_term_table()``.
    """
    columns = [
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
    n_terms = len(plan.labels)
    blocks: list[np.ndarray] = []
    for fit in fits:
        if fit is None:
            blocks.append(np.full((n_terms, len(columns)), np.nan))
        else:
            blocks.append(fit.terms.loc[:, columns].to_numpy(dtype=float))

    out = pd.DataFrame(
        {
            "features": np.repeat(list(feats), n_terms),
            "terms": list(plan.labels) * len(feats),
            "term_order": list(plan.orders) * len(feats),
        }
    )
    stats = pd.DataFrame(np.vstack(blocks), columns=columns)
    out = pd.concat([out, stats], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_center = np.log2(centers.centers.to_numpy(dtype=float))
    effects = [
        fact_term_effect(log2_center[index], cells, plan.terms) for index in range(len(feats))
    ]
    out["log2_effect"] = np.concatenate(effects) if effects else np.array([], dtype=float)

    out["pval_adj"] = np.nan
    for label in plan.labels:
        at = out["terms"].to_numpy() == label
        out.loc[at, "pval_adj"] = p_adjust(out.loc[at, "pval"].to_numpy(dtype=float), adjust)

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


def _fact_posthoc_stage(
    feats: Sequence[str],
    fits: Sequence[FactorialFit | None],
    terms_tbl: pd.DataFrame,
    layout: FactLayout,
    scope: str,
    alpha: float,
    conf_level: float,
) -> dict[str, Any]:
    """Run the contrasts each feature's term tests earned it.

    Port of ``sa_fact_posthoc_stage()``.
    """
    factor_lv = layout.factor_lv
    skeleton = fact_contrast_skeleton(factor_lv, layout.cells)
    rows = skeleton.table
    contract = posthoc_table_columns()
    columns = ["features", "factor", "stratum", *[name for name in contract if name != "features"]]

    if scope == "both":
        wanted = np.ones(len(rows.index), dtype=bool)
    elif scope == "marginal":
        wanted = rows["stratum"].isna().to_numpy()
    else:
        wanted = rows["stratum"].notna().to_numpy()
    candidates = np.flatnonzero(wanted)

    gate: list[str] = []
    for index in candidates:
        factor = str(rows["factor"].iloc[index])
        if pd.isna(rows["stratum"].iloc[index]):
            used = {factor}
        else:
            used = set(factor_lv)
        gate.append(":".join(name for name in factor_lv if name in used))
    nmeans = [len(factor_lv[str(factor)]) for factor in rows["factor"]]

    blocks: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for feature, fit in zip(feats, fits, strict=True):
        if fit is None:
            continue
        padj = terms_tbl.loc[terms_tbl["features"] == feature].set_index("terms")["pval_adj"]
        earned = padj.reindex(gate).to_numpy(dtype=float)
        take = candidates[~np.isnan(earned) & (earned <= alpha)]
        if take.size == 0:
            continue
        try:
            stats_df = factorial_tukey(
                fit,
                skeleton.sel1,
                skeleton.sel2,
                nmeans,
                [int(row) for row in take],
                conf_level,
            )
        except Exception as error:  # noqa: BLE001 - NA rows + grouped warning
            failures[feature] = str(error)
            stats_df = pd.DataFrame(
                np.full((take.size, len(posthoc_columns())), np.nan),
                columns=posthoc_columns(),
            )
        stats_df = stats_df.copy()
        stats_df["pval_adj"] = stats_df["pval"]
        block = pd.DataFrame(
            {
                "features": [feature] * take.size,
                "factor": rows["factor"].iloc[take].to_numpy(),
                "stratum": rows["stratum"].iloc[take].to_numpy(),
                "contrast": rows["contrast"].iloc[take].to_numpy(),
                "group1": rows["group1"].iloc[take].to_numpy(),
                "group2": rows["group2"].iloc[take].to_numpy(),
            }
        )
        blocks.append(pd.concat([block, stats_df.reset_index(drop=True)], axis=1))

    if failures:
        detail = "\n".join(f"  {name}: {message}" for name, message in failures.items())
        warn(
            "Tukey HSD on marginal means and simple effects could not be "
            f"computed for {len(failures)} of {len(feats)} feature(s); those "
            f"rows are NA:\n{detail}"
        )

    if not blocks:
        empty = {
            name: pd.Series(
                dtype=object
                if name in {"features", "factor", "stratum", "contrast", "group1", "group2"}
                else float
            )
            for name in columns
        }
        return {"table": pd.DataFrame(empty)[columns], "n_posthoc": 0}

    out = pd.concat(blocks, ignore_index=True)
    return {"table": out[columns], "n_posthoc": len(blocks)}
