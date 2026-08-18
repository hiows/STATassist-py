"""Simulate a contingency table with known association (R simulate_categorical_groups.R)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.rng_r import get_rng, sa_r_seed, sample_prob_replace
from statassist.utils.validate import sa_check_count, sa_check_flag, sa_check_scalar_num, sa_check_num_vector


def sa_finite_or_na(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out if arr.ndim > 0 else float(out.flat[0])


def sa_sim_cat_check_given(given: dict[str, bool], paired: bool) -> None:
    reads = ["discordance"] if paired else ["margins", "assoc", "pattern"]
    ignored = [k for k, v in given.items() if v and k not in reads]
    if not ignored:
        return
    design = "matched" if paired else "cross-classified"
    other = "TRUE" if paired else "FALSE"
    warnings.warn(
        f"a {design} design reads {', '.join(reads)}, so the value(s) given "
        f"for {', '.join(ignored)} were ignored. Set `paired = {other}` for "
        "the design those belong to.",
        UserWarning,
        stacklevel=3,
    )


def sa_sim_cat_levels(category_lv: dict[str, list[str]]) -> dict[str, list[str]]:
    if (
        not isinstance(category_lv, dict)
        or len(category_lv) < 2
        or any(k is None or k == "" for k in category_lv)
        or len(set(category_lv)) != len(category_lv)
    ):
        raise ValueError(
            "`category_lv` must be a named list of at least two variables, each "
            "entry holding that variable's levels."
        )
    out: dict[str, list[str]] = {}
    for nm, lv in category_lv.items():
        lv = [str(x) for x in lv]
        if len(lv) < 2 or any(x is None or x == "nan" for x in lv) or len(set(lv)) != len(lv):
            raise ValueError(
                f"`category_lv${nm}` must hold at least two distinct "
                "non-missing levels."
            )
        out[nm] = lv
    return out


def sa_sim_cat_margins(
    margins: dict[str, list[float]] | None,
    category_lv: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    if margins is None:
        return {nm: np.full(len(lv), 1 / len(lv)) for nm, lv in category_lv.items()}
    if (
        not isinstance(margins, dict)
        or set(margins.keys()) != set(category_lv.keys())
    ):
        raise ValueError(
            "`margins` must be a named list holding one entry per variable of "
            f"`category_lv`: {', '.join(category_lv)}."
        )
    out: dict[str, np.ndarray] = {}
    for nm in category_lv:
        m = sa_check_num_vector(margins[nm], f"margins${nm}", 0)
        if len(m) != len(category_lv[nm]):
            raise ValueError(
                f"`margins${nm}` must hold one weight per level of `category_lv${nm}`: "
                f"got {len(m)} for {len(category_lv[nm])} level(s)."
            )
        if m.sum() <= 0 or np.any(m == 0):
            raise ValueError(
                f"`margins${nm}` must give every level a positive weight; a level "
                "with none is a level to leave out of `category_lv`."
            )
        out[nm] = m / m.sum()
    return out


def sa_sim_cat_perturb(
    independent: np.ndarray,
    assoc: float,
    pattern: str,
) -> np.ndarray:
    if assoc == 0:
        return independent.copy()
    r, c = independent.shape

    def zero_sum(k: int) -> np.ndarray:
        if pattern == "corner":
            v = np.zeros(k)
            v[0] = 1
            if k > 1:
                v[1] = -1
            return v
        if pattern == "single":
            v = np.full(k, -1 / (k - 1))
            v[0] = 1
            return v
        if pattern == "gradient":
            ramp = np.linspace(1, -1, k)
            return ramp - ramp.mean()
        raise ValueError(f"unknown pattern `{pattern}`")

    step = np.outer(zero_sum(r), zero_sum(c))
    losing = step < 0
    max_step = np.min(independent[losing] / -step[losing])
    out = independent + assoc * max_step * step
    out[out < 0] = 0
    return out


def sa_sim_cat_truth_cell(
    planted: np.ndarray,
    independent: np.ndarray,
    total: int,
    row_lv: list[str] | None = None,
    col_lv: list[str] | None = None,
) -> pd.DataFrame:
    r = planted.shape[0]
    c = planted.shape[1]
    if row_lv is None:
        row_lv = [str(i) for i in range(r)]
    if col_lv is None:
        col_lv = [str(j) for j in range(c)]
    rows = []
    for i in range(r):
        for j in range(c):
            p_ind = independent[i, j]
            p_pl = planted[i, j]
            rows.append(
                {
                    "row_level": row_lv[i],
                    "col_level": col_lv[j],
                    "p_independent": float(p_ind),
                    "p_planted": float(p_pl),
                    "lift": sa_finite_or_na(p_pl / p_ind if p_ind else np.nan),
                    "expected_n": total * float(p_pl),
                }
            )
    return pd.DataFrame(rows)


def sa_sim_cat_truth(
    planted: np.ndarray,
    independent: np.ndarray,
    n_samples: int,
    assoc: float,
    pattern: str,
) -> pd.DataFrame:
    phi_sq = np.sum((planted - independent) ** 2 / independent)
    min_df = min(planted.shape) - 1
    out: dict[str, Any] = {
        "n_samples": n_samples,
        "pattern": pattern,
        "assoc": assoc,
        "cramers_v": float(np.sqrt(phi_sq / min_df)),
    }
    if planted.shape == (2, 2):
        cross = planted[0, 0] * planted[1, 1] - planted[0, 1] * planted[1, 0]
        rs = planted.sum(axis=1)
        cs = planted.sum(axis=0)
        out["phi_coefficient"] = float(
            cross / np.sqrt(np.prod(np.concatenate([rs, cs])))
        )
        denom = planted[0, 1] * planted[1, 0]
        out["odds_ratio"] = sa_finite_or_na(
            planted[0, 0] * planted[1, 1] / denom if denom else np.nan
        )
    return pd.DataFrame([out])


def sa_sim_cat_check_drawn(
    data: pd.DataFrame,
    category_lv: dict[str, list[str]],
) -> None:
    absent = []
    for nm, lv in category_lv.items():
        missed = set(lv) - set(data[nm].astype(str))
        if missed:
            absent.append(f"{nm}: {', '.join(sorted(missed))}")
    if absent:
        warnings.warn(
            "no row was drawn at level(s) "
            + "; ".join(absent)
            + ", which leaves an empty row or column that no test of "
            "independence can be run on. Raise `n_samples`, or give the level "
            "more weight in `margins`.",
            UserWarning,
            stacklevel=3,
        )


def sa_sim_cat_crossed(
    n_samples: int,
    category_lv: dict[str, list[str]],
    margins: dict[str, list[float]] | None,
    assoc: float,
    pattern: str,
) -> dict[str, Any]:
    if len(category_lv) != 2:
        raise ValueError(
            "a cross-classified simulation plants an association between exactly "
            f"two variables, and `category_lv` names {len(category_lv)}. "
            "Set `paired = TRUE` for repeated measurements of one thing."
        )
    variables = list(category_lv.keys())
    row_lv = category_lv[variables[0]]
    col_lv = category_lv[variables[1]]

    probs = sa_sim_cat_margins(margins, category_lv)
    independent = np.outer(probs[variables[0]], probs[variables[1]])
    independent = pd.DataFrame(independent, index=row_lv, columns=col_lv).to_numpy()

    planted_df = sa_sim_cat_perturb(independent, assoc, pattern)
    planted_df = pd.DataFrame(planted_df, index=row_lv, columns=col_lv)

    flat_probs = np.asfortranarray(planted_df.to_numpy()).ravel(order="F")
    drawn = sample_prob_replace(flat_probs, n_samples) - 1
    row_idx = drawn % len(row_lv)
    col_idx = drawn // len(row_lv)

    data = pd.DataFrame(
        {
            variables[0]: [row_lv[i] for i in row_idx],
            variables[1]: [col_lv[i] for i in col_idx],
        }
    )
    sa_sim_cat_check_drawn(data, category_lv)

    planted = planted_df.to_numpy()
    return {
        "args": {"data": data, "category_lv": category_lv, "paired": False},
        "truth": sa_sim_cat_truth(planted, independent, n_samples, assoc, pattern),
        "truth_cell": sa_sim_cat_truth_cell(
            planted, independent, n_samples, row_lv, col_lv
        ),
    }


def sa_sim_cat_truth_matched(
    n_samples: int,
    k: int,
    discordance: np.ndarray,
    rates: np.ndarray,
) -> pd.DataFrame:
    move_up, move_down = discordance
    b = 0.5 * move_up
    c = 0.5 * move_down
    out: dict[str, Any] = {
        "n_samples": n_samples,
        "pattern": "transition",
        "n_conditions": k,
        "move_up": float(move_up),
        "move_down": float(move_down),
    }
    if k == 2:
        out["odds_ratio_paired"] = sa_finite_or_na(b / c if c else np.nan)
        out["risk_difference_paired"] = b - c
        out["cohens_g"] = sa_finite_or_na(b / (b + c) - 0.5 if (b + c) else np.nan)
    else:
        out["rate_first"] = float(rates[0])
        out["rate_last"] = float(rates[-1])
        out["rate_range"] = float(rates.max() - rates.min())
    return pd.DataFrame([out])


def sa_sim_cat_truth_cell_matched(
    n_samples: int,
    k: int,
    variables: list[str],
    levels: list[str],
    discordance: np.ndarray,
    rates: np.ndarray,
) -> pd.DataFrame:
    if k == 2:
        move_up, move_down = discordance
        planted = np.array(
            [
                [0.5 * (1 - move_up), 0.5 * move_down],
                [0.5 * move_up, 0.5 * (1 - move_down)],
            ]
        )
        independent = np.outer(planted.sum(axis=1), planted.sum(axis=0))
        out = sa_sim_cat_truth_cell(
            planted, independent, n_samples, levels, levels
        )
        symmetric = (planted + planted.T) / 2
        out["p_symmetric"] = symmetric.ravel()
        out["expected_symmetry_n"] = n_samples * symmetric.ravel()
        return out

    planted = np.column_stack([1 - rates, rates]) / k
    independent = np.outer(planted.sum(axis=1), planted.sum(axis=0))
    out = sa_sim_cat_truth_cell(
        planted, independent, n_samples * k, variables, levels
    )
    return out


def sa_sim_cat_matched(
    n_samples: int,
    category_lv: dict[str, list[str]],
    discordance: list[float],
) -> dict[str, Any]:
    discordance = sa_check_num_vector(discordance, "discordance", 0, 1)
    if len(discordance) != 2:
        raise ValueError(
            "`discordance` must be two transition probabilities, from the first "
            "level to the second and back."
        )

    levels = category_lv[list(category_lv.keys())[0]]
    if not all(list(lv) == list(levels) for lv in category_lv.values()):
        raise ValueError(
            "a matched simulation measures one thing repeatedly, so every entry "
            "of `category_lv` holds the same levels in the same order. They differ."
        )
    if len(levels) != 2:
        raise ValueError(
            f"a matched simulation needs binary conditions, and `category_lv` "
            f"holds {len(levels)} level(s). McNemar's test and Cochran's Q are "
            "both about a binary response."
        )

    variables = list(category_lv.keys())
    k = len(variables)
    move_up, move_down = discordance

    rng = get_rng()
    state = np.zeros((n_samples, k), dtype=bool)
    state[:, 0] = rng.runif(n_samples) < 0.5
    for j in range(k - 1):
        move_p = np.where(state[:, j], move_down, move_up)
        move = rng.runif(n_samples) < move_p
        state[:, j + 1] = np.logical_xor(state[:, j], move)

    rates = np.zeros(k)
    rates[0] = 0.5
    for j in range(k - 1):
        rates[j + 1] = rates[j] * (1 - move_down) + (1 - rates[j]) * move_up

    data = pd.DataFrame(
        {
            variables[j]: [levels[int(state[i, j])] for i in range(n_samples)]
            for j in range(k)
        }
    )

    return {
        "args": {"data": data, "category_lv": category_lv, "paired": True},
        "truth": sa_sim_cat_truth_matched(n_samples, k, discordance, rates),
        "truth_cell": sa_sim_cat_truth_cell_matched(
            n_samples, k, variables, levels, discordance, rates
        ),
    }


_MISSING = object()


def simulate_categorical_groups(
    n_samples: int = 200,
    category_lv: dict[str, list[str]] | None = None,
    margins: dict[str, list[float]] | None = None,
    assoc: float | object = _MISSING,
    pattern: str | object = _MISSING,
    paired: bool = False,
    discordance: list[float] | object = _MISSING,
    seed: float | None = None,
) -> dict[str, Any]:
    given = {
        "margins": margins is not None,
        "assoc": assoc is not _MISSING,
        "pattern": pattern is not _MISSING,
        "discordance": discordance is not _MISSING,
    }

    if assoc is _MISSING:
        assoc = 0.3
    if pattern is _MISSING:
        pattern = "corner"
    elif pattern not in ("corner", "single", "gradient"):
        raise ValueError("`pattern` must be one of: corner, single, gradient.")
    pattern = str(pattern)

    if discordance is _MISSING:
        discordance = [0.25, 0.10]

    sa_check_flag(paired, "paired")
    n_samples = sa_check_count(n_samples, "n_samples", 2)
    sa_check_scalar_num(float(assoc), "assoc", 0, 1)

    if category_lv is None:
        category_lv = (
            {"before": ["fail", "pass"], "after": ["fail", "pass"]}
            if paired
            else {"cat_1": ["y", "n"], "cat_2": ["high", "mid", "low"]}
        )
    category_lv = sa_sim_cat_levels(category_lv)
    sa_sim_cat_check_given(given, paired)

    with sa_r_seed(seed):
        if paired:
            return sa_sim_cat_matched(n_samples, category_lv, list(discordance))
        return sa_sim_cat_crossed(
            n_samples, category_lv, margins, float(assoc), pattern
        )
