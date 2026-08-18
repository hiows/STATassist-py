"""Factorial design helpers (R utils_factorial.R + simulate_factorial_groups.R)."""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_control_first, sa_level_pairs

FACT_TOL = 1e-8


def sa_fact_grid(factor_lv: dict[str, list[str]]) -> pd.DataFrame:
    """Match R ``expand.grid(lapply(lv_list, seq_along))``: first factor varies fastest."""
    if not factor_lv:
        return pd.DataFrame(index=[0])
    names = list(factor_lv.keys())
    ranges = [range(1, len(factor_lv[n]) + 1) for n in names]
    rows = [tuple(reversed(r)) for r in product(*reversed(ranges))]
    return pd.DataFrame(rows, columns=names)


def sa_fact_cell_labels(factor_lv: dict[str, list[str]], cells: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in cells.iterrows():
        parts = [factor_lv[f][int(row[f]) - 1] for f in factor_lv]
        labels.append(".".join(parts))
    return labels


def sa_fact_cell_index(level_idx: np.ndarray, dims: list[int]) -> np.ndarray:
    if len(dims) == 0:
        return np.ones(level_idx.shape[0], dtype=int)
    strides = np.cumprod([1] + dims[:-1]).astype(int)
    return 1 + (level_idx - 1) @ strides


def sa_fact_terms(fac_names: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for m in range(1, len(fac_names) + 1):
        out.extend([list(c) for c in combinations(fac_names, m)])
    return out


def sa_fact_term_labels(terms: list[list[str]]) -> list[str]:
    return [":".join(t) for t in terms]


def sa_fact_subsets(term: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for m in range(1, len(term) + 1):
        out.extend([list(c) for c in combinations(term, m)])
    return out


def sa_r_mean(x: np.ndarray) -> float:
    """R ``mean.default()`` for a double vector.

    R accumulates in long double and then makes a second pass to correct the
    result, which lands on a different last bit than NumPy's pairwise summation.
    That bit decides which of two equal-and-opposite ANOVA components is seen as
    the larger one, so it is worth reproducing rather than rounding away.
    """
    n = len(x)
    if n == 0:
        return float("nan")
    s = math.fsum(x) / n
    if not math.isfinite(s):
        return float(s)
    t = math.fsum(float(v) - s for v in x)
    return float(s + t / n)


def sa_fact_collapse(eff: np.ndarray, cells: pd.DataFrame, keep: list[str]) -> np.ndarray:
    if len(keep) == 0:
        return np.full(len(eff), sa_r_mean(eff))
    keys = cells[keep].astype(str).agg(".".join, axis=1).to_numpy()
    out = np.empty(len(eff), dtype=float)
    for key in np.unique(keys):
        mask = keys == key
        out[mask] = sa_r_mean(eff[mask])
    return out


def sa_fact_component(eff: np.ndarray, cells: pd.DataFrame, term: list[str]) -> np.ndarray:
    total = np.zeros(len(eff))
    for sub in sa_fact_subsets(term):
        sign = (-1) ** (len(term) - len(sub))
        collapsed = sa_fact_collapse(eff, cells, sub)
        total = total + sign * collapsed
    total[np.abs(total) < FACT_TOL] = 0.0
    return total


def sa_fact_term_effect(
    eff: np.ndarray,
    cells: pd.DataFrame,
    terms: list[list[str]],
) -> np.ndarray:
    """The largest effect each term accounts for, with its sign.

    One number per term out of the whole component vector, so a term has an
    effect size that can be put on an axis beside its p-value. A component is a
    deviation from what the other terms already predict, not the difference
    between two levels: the components of a two-level factor whose levels differ
    by ``d`` are ``-d/2`` and ``+d/2``. That is the quantity
    ``simulate_factorial_groups()`` records in ``truth_term$max_abs_delta``, so
    the two tables hold the same number and a result can be scored term by term.
    """
    out = np.empty(len(terms), dtype=float)
    for i, term in enumerate(terms):
        comp = sa_fact_component(eff, cells, term)
        if np.all(np.isnan(comp)):
            out[i] = np.nan
            continue
        # A component vector is symmetric about zero, so the largest absolute
        # value is reached twice and which copy is seen first is otherwise
        # decided by the last bit of a group mean. Ties take the earlier cell,
        # which is the earlier level of the first factor.
        magnitude = np.abs(comp)
        largest = float(np.nanmax(magnitude))
        out[i] = comp[int(np.flatnonzero(magnitude >= largest - FACT_TOL)[0])]
    return out


def sa_fact_control_first(
    factor_lv: dict[str, list[str]],
    control_label: dict[str, str] | None,
    lv_source: str = "factor_lv",
) -> dict[str, list[str]]:
    """Point each named factor at the reference level it was told to hold.

    A crossed design has one reference per factor rather than one in total, so
    naming a level moves it to the front of its own factor and leaves the other
    factors in the order they arrived.
    """
    if control_label is None:
        return factor_lv
    if not isinstance(control_label, dict) or len(control_label) == 0:
        raise ValueError(
            "`control_label` must be a named list, one level name per factor it "
            "points, with the factor's name as the name. Naming a factor twice, "
            "or none, is not a direction."
        )
    unknown = [nm for nm in control_label if nm not in factor_lv]
    if unknown:
        raise ValueError(
            "`control_label` names factor(s) the design does not hold: "
            f"{', '.join(unknown)}. Present: {', '.join(factor_lv)}."
        )
    out = dict(factor_lv)
    for nm, ref in control_label.items():
        out[nm] = sa_control_first(
            out[nm],
            ref,
            arg=f"control_label${nm}",
            lv_arg=f"{lv_source}${nm}",
        )
    return out


def sa_fact_contrast_skeleton(design: dict[str, Any]) -> dict[str, Any]:
    cells = design["cells"]
    factor_lv = design["factor_lv"]
    fac: list[str] = []
    stratum: list[str | None] = []
    contrast: list[str] = []
    group1: list[str] = []
    group2: list[str] = []
    sel1: list[np.ndarray] = []
    sel2: list[np.ndarray] = []

    def add(f: str, label: str | None, at: list[np.ndarray]) -> None:
        pairs = sa_level_pairs(factor_lv[f])
        for _, prow in pairs.iterrows():
            fac.append(f)
            stratum.append(label)
            contrast.append(prow["contrast"])
            group1.append(prow["group1"])
            group2.append(prow["group2"])
            # `sa_level_pairs()` indexes levels from zero here, unlike its R
            # counterpart, so these are used as they arrive.
            sel1.append(at[int(prow["i"])])
            sel2.append(at[int(prow["j"])])

    for f in factor_lv:
        lv = factor_lv[f]
        others = [o for o in factor_lv if o != f]
        strata = sa_fact_grid({o: factor_lv[o] for o in others}) if others else pd.DataFrame([{}])
        at_marginal = [np.where(cells[f].to_numpy() == j + 1)[0] for j in range(len(lv))]
        add(f, None, at_marginal)
        if len(strata) == 0:
            continue
        for s_idx in range(len(strata)):
            held = np.ones(len(cells), dtype=bool)
            label_parts = []
            for o in others:
                lvl = int(strata.iloc[s_idx][o])
                held &= cells[o].to_numpy() == lvl
                label_parts.append(factor_lv[o][lvl - 1])
            label = ".".join(label_parts)
            at_simple = [
                np.where(held & (cells[f].to_numpy() == j + 1))[0] for j in range(len(lv))
            ]
            add(f, label, at_simple)

    table = pd.DataFrame(
        {
            "factor": fac,
            "stratum": stratum,
            "contrast": contrast,
            "group1": group1,
            "group2": group2,
        }
    )
    return {"table": table, "sel1": sel1, "sel2": sel2}


def sa_fact_shapes() -> list[str]:
    return ["main_only", "additive", "interaction", "crossover", "nuisance_only"]


def sa_fact_shuffle(x: list[str]) -> list[str]:
    from statassist.utils.rng_r import get_rng

    if len(x) <= 1:
        return x
    idx = get_rng().sample_int(len(x), len(x)) - 1
    return [x[i] for i in idx]


def sa_fact_tol() -> float:
    return FACT_TOL


def sa_fact_profile(d: float, spread: str, k: int) -> np.ndarray:
    from statassist.utils.simulate_utils import sa_sim_pattern_delta

    raw = np.concatenate([[0.0], sa_sim_pattern_delta(d, spread, k - 1)])
    return raw - raw.mean()


def sa_fact_flip(k: int) -> np.ndarray:
    s = np.resize(np.array([1.0, -1.0]), k)
    s = s - s.mean()
    return s / np.max(np.abs(s))


def sa_fact_partner(shape: str, fac_names: list[str]) -> str | None:
    from statassist.utils.rng_r import get_rng

    if shape == "main_only":
        return None
    others = fac_names[1:]
    return others[get_rng().sample_int(len(others), 1)[0] - 1]


def sa_fact_plant(
    d: float,
    shape: str,
    spread: str,
    mate: str | None,
    factor_lv: dict[str, list[str]],
    cells: pd.DataFrame,
    interaction_scale: float,
) -> np.ndarray:
    primary = list(factor_lv.keys())[0]

    def profile(f: str) -> np.ndarray:
        return sa_fact_profile(d, spread, len(factor_lv[f]))

    def main(f: str, v: np.ndarray) -> np.ndarray:
        return v[cells[f].to_numpy() - 1]

    def crossed(v: np.ndarray) -> np.ndarray:
        flip = sa_fact_flip(len(factor_lv[mate]))
        mat = np.outer(v, flip)
        pi = cells[primary].to_numpy() - 1
        mi = cells[mate].to_numpy() - 1
        return mat[pi, mi]

    if shape == "main_only":
        return main(primary, profile(primary))
    if shape == "additive":
        return main(primary, profile(primary)) + main(mate, profile(mate))
    if shape == "nuisance_only":
        return main(mate, profile(mate))
    if shape == "interaction":
        p = profile(primary)
        return main(primary, p) + interaction_scale * crossed(p)
    if shape == "crossover":
        return crossed(profile(primary))
    raise RuntimeError(f"internal error: unknown effect shape `{shape}`.")


def sa_fact_design(
    factor_lv: dict[str, list[str]],
    within: list[str] | None,
    n_per_cell: int | list[int],
) -> dict[str, Any]:
    from statassist.utils.validate import sa_check_count

    if (
        not isinstance(factor_lv, dict)
        or len(factor_lv) < 2
        or any(k is None or k == "" for k in factor_lv)
        or len(set(factor_lv)) != len(factor_lv)
    ):
        raise ValueError(
            "`factor_lv` must be a named list of at least two crossed factors, "
            "each entry the levels of one factor with the reference level first. "
            "Use simulate_multiple_groups() for a single factor."
        )
    reserved = set(factor_lv) & {
        "features",
        "is_ref",
        "delta",
        "center",
        "sd",
        "n",
    }
    if reserved:
        raise ValueError(
            "`factor_lv` names factor(s) that the answer tables already use as "
            f"columns: {', '.join(sorted(reserved))}."
        )
    for nm, lv in factor_lv.items():
        if (
            not isinstance(lv, (list, tuple))
            or len(lv) < 2
            or any(x is None or x == "" for x in lv)
            or len(set(lv)) != len(lv)
        ):
            raise ValueError(
                f"`factor_lv${nm}` must be at least two distinct non-empty "
                "level names, the first being the reference."
            )

    if within is None:
        within = []
    within = list(within)
    if any(w is None or w == "" for w in within) or len(set(within)) != len(within):
        raise ValueError(
            "`within` must be NULL, or the distinct names of the factors "
            "measured within subjects."
        )
    unknown = set(within) - set(factor_lv)
    if unknown:
        raise ValueError(
            f"`within` names factor(s) that `factor_lv` does not hold: "
            f"{', '.join(sorted(unknown))}. Known factors are: "
            f"{', '.join(factor_lv)}."
        )
    within = [f for f in factor_lv if f in within]
    between = [f for f in factor_lv if f not in within]

    dims = [len(factor_lv[f]) for f in factor_lv]
    cells = sa_fact_grid(factor_lv)
    n_cells = len(cells)
    cell_label = sa_fact_cell_labels(factor_lv, cells)

    between_cells = sa_fact_grid({f: factor_lv[f] for f in between})
    n_between = max(len(between_cells), 1)
    within_cells = sa_fact_grid({f: factor_lv[f] for f in within}) if within else pd.DataFrame([{}])
    n_within = max(len(within_cells), 1)

    if isinstance(n_per_cell, (int, float)):
        n_per_cell_list = [int(n_per_cell)]
    else:
        n_per_cell_list = list(n_per_cell)

    if len(n_per_cell_list) not in (1, n_between):
        raise ValueError(
            f"`n_per_cell` must be one size, or one size per combination of the "
            f"between-subject factors, of which this design has {n_between}. "
            "The within-subject factors are crossed with every subject, so "
            "they cannot hold sizes of their own."
        )
    if len(n_per_cell_list) == 1:
        sizes = [sa_check_count(n_per_cell_list[0], "n_per_cell", 2)] * n_between
    else:
        sizes = [
            sa_check_count(n_per_cell_list[k], f"n_per_cell[{k}]", 2)
            for k in range(len(n_per_cell_list))
        ]

    unit_between = np.repeat(np.arange(n_between), sizes)
    n_units = len(unit_between)
    n_rows = n_units * n_within
    subject_idx = np.repeat(np.arange(n_units), n_within)
    within_row = np.tile(np.arange(n_within), n_units)

    fac_names = list(factor_lv.keys())
    level_idx = np.zeros((n_rows, len(fac_names)), dtype=int)
    for f in between:
        col = between_cells[f].to_numpy()
        level_idx[:, fac_names.index(f)] = col[unit_between[subject_idx]]
    for f in within:
        if within:
            col = within_cells[f].to_numpy()
            level_idx[:, fac_names.index(f)] = col[within_row]

    cell_idx = sa_fact_cell_index(level_idx, dims) - 1
    between_dims = [len(factor_lv[f]) for f in between] if between else []
    cell_between = (
        sa_fact_cell_index(cells[between].to_numpy(), between_dims) - 1
        if between
        else np.zeros(n_cells, dtype=int)
    )

    factors = {
        f: np.array([factor_lv[f][level_idx[i, fac_names.index(f)] - 1] for i in range(n_rows)])
        for f in fac_names
    }

    return {
        "factor_lv": factor_lv,
        "within": within,
        "between": between,
        "cells": cells,
        "n_cells": n_cells,
        "cell_label": cell_label,
        "ref_cell": 0,
        "cell_n": np.array(sizes)[cell_between],
        "terms": sa_fact_terms(fac_names),
        "cell_idx": cell_idx,
        "factors": factors,
        "subject": (
            [f"subject_{i + 1}" for i in subject_idx] if within else None
        ),
        "subject_idx": subject_idx,
        "n_units": n_units,
        "n_rows": n_rows,
    }


def sa_fact_truth(
    feats: list[str],
    delta: np.ndarray,
    design: dict[str, Any],
    pattern: np.ndarray,
    spread: np.ndarray,
    direction: np.ndarray,
    partner: np.ndarray,
    baseline: np.ndarray,
    sd_subject: np.ndarray,
) -> pd.DataFrame:
    abs_delta = np.abs(delta)
    largest = abs_delta.max(axis=1)
    tied = (abs_delta == largest[:, None]).sum(axis=1) > 1
    which_max = abs_delta.argmax(axis=1)
    cell_label = design["cell_label"]
    extreme_cell = np.array([cell_label[i] for i in which_max], dtype=object)
    extreme_cell[largest == 0] = None
    log2fc = delta[np.arange(len(feats)), which_max]
    return pd.DataFrame(
        {
            "features": feats,
            "pattern": pattern,
            "spread": spread,
            "direction": direction,
            "partner": partner,
            "extreme_cell": extreme_cell,
            "extreme_tied": tied,
            "log2fc": log2fc,
            "baseline": baseline,
            "sd_subject": sd_subject,
        }
    )


def sa_fact_truth_term(
    feats: list[str],
    delta: np.ndarray,
    design: dict[str, Any],
    planted: np.ndarray,
) -> pd.DataFrame:
    terms = design["terms"]
    labels = sa_fact_term_labels(terms)
    orders = [len(t) for t in terms]
    is_within = [any(t in design["within"] for t in term) for term in terms]
    cells = design["cells"]
    comp = np.zeros((len(feats), len(terms)))
    for i in np.where(planted)[0]:
        for k, term in enumerate(terms):
            comp[i, k] = np.max(
                np.abs(sa_fact_component(delta[i], cells, term))
            )
    flat = comp.ravel(order="F")
    return pd.DataFrame(
        {
            "features": np.repeat(feats, len(terms)),
            "terms": np.tile(labels, len(feats)),
            "term_order": np.tile(orders, len(feats)),
            "is_within": np.tile(is_within, len(feats)),
            "max_abs_delta": flat,
            "is_effect": flat > 0,
        }
    )


def sa_fact_truth_cell(
    feats: list[str],
    delta: np.ndarray,
    center: np.ndarray,
    sd_mat: np.ndarray,
    design: dict[str, Any],
) -> pd.DataFrame:
    n_cells = design["n_cells"]
    n_feats = len(feats)
    rows: dict[str, Any] = {"features": np.repeat(feats, n_cells)}
    cells = design["cells"]
    for f in design["factor_lv"]:
        rows[f] = np.tile(
            [design["factor_lv"][f][cells[f].iloc[j] - 1] for j in range(n_cells)],
            n_feats,
        )
    rows["is_ref"] = np.tile(np.arange(n_cells) == design["ref_cell"], n_feats)
    rows["delta"] = delta.ravel(order="F")
    rows["center"] = center.ravel(order="F")
    rows["sd"] = sd_mat.ravel(order="F")
    rows["n"] = np.tile(design["cell_n"], n_feats)
    return pd.DataFrame(rows)


def sa_fact_truth_contrast(
    feats: list[str],
    delta: np.ndarray,
    design: dict[str, Any],
) -> pd.DataFrame:
    skel = sa_fact_contrast_skeleton(design)
    n_rows = len(skel["table"])
    mat = np.zeros((len(feats), n_rows))
    for k in range(n_rows):
        mat[:, k] = delta[:, skel["sel1"][k]].mean(axis=1) - delta[
            :, skel["sel2"][k]
        ].mean(axis=1)
    mat[np.abs(mat) < sa_fact_tol()] = 0
    flat = mat.ravel(order="F")
    out = skel["table"].loc[np.tile(np.arange(n_rows), len(feats))].reset_index(
        drop=True
    )
    out.insert(0, "features", np.repeat(feats, n_rows))
    out["delta"] = flat
    out["is_diff"] = flat != 0
    return out
