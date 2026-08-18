"""Internal helpers shared by the supervised learning and group simulators.

Transcription of STATassist/R/utils_simulate.R and the helpers at the bottom of
STATassist/R/simulate_multiple_groups.R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cholesky
from scipy.optimize import brentq
from scipy.special import expit

from statassist.utils.rng_r import rnorm, runif, sample_int
from statassist.utils.validate import (
    sa_check_count,
    sa_check_range,
    sa_check_scalar_num,
    sa_level_pairs,
)


class _MISSING:
    """Sentinel for R ``missing()`` semantics."""


def _is_missing(x: Any) -> bool:
    return x is _MISSING


def sa_sim_recycle(x: Any, n: int, arg: str, lower: float = -np.inf) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size not in (1, n) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{arg}` must be a finite numeric vector of length 1 or {n}, "
            "the number of numeric predictors."
        )
    if np.any(arr < lower):
        raise ValueError(f"`{arg}` must not go below {lower}.")
    return np.resize(arr, n)


def sa_sim_chol(cor_mat: np.ndarray) -> np.ndarray | None:
    try:
        return cholesky(np.asarray(cor_mat, dtype=float), lower=False)
    except LinAlgError:
        return None


def sa_sim_cor_root(
    cor_mat: np.ndarray | None,
    n_pred: int,
    arg: str = "cor_mat",
) -> np.ndarray:
    if cor_mat is None:
        return np.eye(n_pred)
    mat = np.asarray(cor_mat, dtype=float)
    if mat.ndim != 2 or mat.shape != (n_pred, n_pred):
        raise ValueError(
            f"`{arg}` must be a numeric {n_pred} x {n_pred} matrix, "
            "one row and column per numeric predictor."
        )
    if not np.all(np.isfinite(mat)):
        raise ValueError(
            f"`{arg}` must not contain missing or non-finite values."
        )
    if not np.allclose(mat, mat.T):
        raise ValueError(f"`{arg}` must be symmetric.")
    if not np.all(np.diag(mat) == 1):
        raise ValueError(
            f"`{arg}` must have 1 on its diagonal, since a variable is "
            "perfectly correlated with itself."
        )
    if np.any(np.abs(mat) > 1):
        raise ValueError(f"`{arg}` holds correlation(s) outside [-1, 1].")

    root = sa_sim_chol(mat)
    if root is None:
        raise ValueError(
            f"`{arg}` is not positive definite, so no data has these "
            "correlations. Build it with make_block_cor(), which says which of "
            "the blocks cannot hold."
        )
    return root


def sa_sim_mvnorm(
    n: int,
    value_mean: np.ndarray,
    value_sd: np.ndarray,
    root: np.ndarray,
) -> np.ndarray:
    n_pred = len(value_mean)
    z = rnorm(n * n_pred).reshape(n, n_pred, order="F")
    out = z @ root
    return out * value_sd + value_mean


def sa_sim_pred_spec(
    n_pred: int | Any,
    beta: np.ndarray | None,
    n_pos: int,
    n_neg: int,
    beta_range: tuple[float, float] | np.ndarray,
    value_mean: Any,
    value_sd: Any,
    explicit: set[str],
) -> dict[str, Any]:
    if beta is None:
        n_pred = sa_check_count(n_pred, "n_pred", 1)
        n_pos = sa_check_count(n_pos, "n_pos")
        n_neg = sa_check_count(n_neg, "n_neg")
        if n_pos + n_neg > n_pred:
            raise ValueError(
                f"`n_pos` + `n_neg` is {n_pos + n_neg}, which is more "
                f"coefficients than the {n_pred} numeric predictor(s) that "
                "`n_pred` asks for."
            )
    else:
        clash = sorted(explicit & {"n_pos", "n_neg"})
        if clash:
            joined = " and ".join(f"`{c}`" for c in clash)
            raise ValueError(
                "`beta` states every coefficient, so there is nothing left for "
                f"{joined} to plant. Drop one of the two."
            )
        beta_arr = np.asarray(beta, dtype=float)
        if beta_arr.size == 0 or not np.all(np.isfinite(beta_arr)):
            raise ValueError(
                "`beta` must be a finite numeric vector, one coefficient per "
                "numeric predictor and no intercept among them."
            )
        if "n_pred" in explicit and beta_arr.size != n_pred:
            raise ValueError(
                f"`n_pred` asks for {n_pred} numeric predictor(s) but `beta` "
                f"gives {beta_arr.size} coefficient(s). The intercept is not one "
                "of them: it is `intercept` for a regression and `event_rate` for a "
                "classification."
            )
        n_pred = int(beta_arr.size)
        n_pos = 0
        n_neg = 0

    return {
        "n_pred": n_pred,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "beta": beta,
        "value_mean": sa_sim_recycle(value_mean, n_pred, "value_mean"),
        "value_sd": sa_sim_recycle(value_sd, n_pred, "value_sd", 0),
    }


def sa_sim_plant_beta(
    spec: dict[str, Any],
    beta_range: tuple[float, float] | np.ndarray,
) -> dict[str, Any]:
    if spec["beta"] is not None:
        coefs = np.asarray(spec["beta"], dtype=float)
        direction = np.where(
            coefs > 0,
            "up",
            np.where(coefs < 0, "down", "none"),
        )
        return {"beta": coefs, "direction": direction}

    n_pred = spec["n_pred"]
    coefs = np.zeros(n_pred, dtype=float)
    direction = np.array(["none"] * n_pred, dtype=object)
    if spec["n_pos"] + spec["n_neg"] > 0:
        picked = sample_int(n_pred, spec["n_pos"] + spec["n_neg"])
        pos_idx = picked[: spec["n_pos"]] - 1
        neg_idx = picked[spec["n_pos"] :] - 1
        direction[pos_idx] = "up"
        direction[neg_idx] = "down"
        lo, hi = float(beta_range[0]), float(beta_range[1])
        if spec["n_pos"] > 0:
            coefs[pos_idx] = runif(spec["n_pos"], lo, hi)
        if spec["n_neg"] > 0:
            coefs[neg_idx] = -runif(spec["n_neg"], lo, hi)
    return {"beta": coefs, "direction": direction}


def sa_sim_subject_sizes(
    n_samples: int | Any,
    n_per_subject: np.ndarray | list[float] | None,
    use_default_n: bool,
) -> dict[str, Any]:
    if n_per_subject is None:
        return {
            "sizes": None,
            "n_samples": sa_check_count(n_samples, "n_samples", 2),
        }

    per_arr = np.asarray(n_per_subject, dtype=float)
    if per_arr.size == 0:
        raise ValueError(
            "`n_per_subject` must be one or more row counts, one per subject, or "
            "NULL for one row per subject."
        )

    if per_arr.size == 1:
        n_samples = sa_check_count(n_samples, "n_samples", 2)
        per = sa_check_count(float(per_arr[0]), "n_per_subject", 1)
        if n_samples % per != 0:
            raise ValueError(
                f"`n_per_subject` = {per} does not divide the {n_samples} "
                "row(s) `n_samples` asks for. Pass a row count per subject, such "
                f"as `n_per_subject = rep({per}, {n_samples // per})`."
            )
        sizes = np.full(n_samples // per, per, dtype=int)
    else:
        sizes = np.array(
            [
                sa_check_count(float(per_arr[k]), f"n_per_subject[{k + 1}]", 1)
                for k in range(per_arr.size)
            ],
            dtype=int,
        )
        total = int(sizes.sum())
        if not use_default_n and sa_check_count(n_samples, "n_samples", 2) != total:
            raise ValueError(
                f"`n_per_subject` gives {len(sizes)} subject(s) holding "
                f"{total} row(s) in all, but `n_samples` asks for {n_samples}. "
                "Drop one of the two."
            )
        n_samples = total

    if len(sizes) < 2:
        raise ValueError(
            f"`n_per_subject` describes {len(sizes)} subject(s), and a "
            "split taken over subjects needs at least 2."
        )
    return {"sizes": sizes, "n_samples": int(n_samples)}


def sa_sim_balanced_levels(n: int, levels: list[str]) -> pd.Categorical:
    k = len(levels)
    pool = np.resize(np.arange(1, k + 1), n)
    perm = sample_int(n, n)
    picked = [levels[i - 1] for i in pool[perm - 1]]
    return pd.Categorical(picked, categories=levels, ordered=False)


def sa_sim_factor_offsets(
    factor_lv: list[str],
    beta_range: tuple[float, float] | np.ndarray,
) -> dict[str, float]:
    k = len(factor_lv)
    lo, hi = float(beta_range[0]), float(beta_range[1])
    signs = np.resize(np.array([1.0, -1.0]), k - 1)
    offsets = {lv: 0.0 for lv in factor_lv}
    if k > 1:
        drawn = runif(k - 1, lo, hi)
        for lv, sign, val in zip(factor_lv[1:], signs, drawn):
            offsets[lv] = float(sign * val)
    return offsets


def sa_sim_mask_missing(x: pd.DataFrame, p_missing: float) -> pd.DataFrame:
    out = x.copy()
    if p_missing == 0 or out.shape[1] == 0:
        return out
    n_row, n_col = out.shape
    n_na = int(round(p_missing * n_row * n_col))
    if n_na == 0:
        return out

    at = sample_int(n_row * n_col, n_na)
    rows = (at - 1) % n_row
    cols = (at - 1) // n_row
    for j in np.unique(cols):
        mask = cols == j
        out.iloc[rows[mask], int(j)] = np.nan
    return out


def sa_sim_solve_intercept(eta: np.ndarray, event_rate: float) -> float:
    def gap(a: float) -> float:
        return float(np.mean(expit(a + eta)) - event_rate)

    lower = -1.0
    upper = 1.0
    while gap(lower) > 0 and lower > -1e4:
        lower *= 2
    while gap(upper) < 0 and upper < 1e4:
        upper *= 2
    if gap(lower) > 0 or gap(upper) < 0:
        raise ValueError(
            f"no intercept gives an event rate of {event_rate} "
            "on these predictors. A rate this far from a half needs a smaller "
            "`beta_range` or a smaller `subject_sd`."
        )
    tol = np.finfo(float).eps ** 0.5
    return float(brentq(gap, lower, upper, xtol=tol, rtol=tol))


def sa_sim_supervised_design(
    n_samples: int | Any,
    n_pred: int | Any,
    beta: np.ndarray | None,
    n_pos: int,
    n_neg: int,
    beta_range: tuple[float, float] | np.ndarray,
    value_mean: Any,
    value_sd: Any,
    cor_mat: np.ndarray | None,
    n_factor_pred: int,
    factor_lv: list[str],
    n_constant_pred: int,
    p_missing: float,
    n_per_subject: np.ndarray | list[float] | None,
    subject_sd: float,
    subject_share: float,
    pred_prefix: str,
    explicit: set[str],
    use_default_n: bool,
) -> dict[str, Any]:
    spec = sa_sim_pred_spec(
        n_pred,
        beta,
        n_pos,
        n_neg,
        beta_range,
        value_mean,
        value_sd,
        explicit,
    )
    n_pred = spec["n_pred"]
    sa_check_range(beta_range, "beta_range", 0)
    root = sa_sim_cor_root(cor_mat, n_pred)

    n_factor_pred = sa_check_count(n_factor_pred, "n_factor_pred")
    n_constant_pred = sa_check_count(n_constant_pred, "n_constant_pred")
    sa_check_scalar_num(p_missing, "p_missing", 0, 1, upper_open=True)
    sa_check_scalar_num(subject_sd, "subject_sd", 0)
    sa_check_scalar_num(subject_share, "subject_share", 0, 1)
    if n_factor_pred > 0 and (
        not isinstance(factor_lv, (list, tuple))
        or len(factor_lv) < 2
        or any(pd.isna(lv) for lv in factor_lv)
        or len(set(factor_lv)) != len(factor_lv)
    ):
        raise ValueError(
            "`factor_lv` must be at least two distinct non-missing level names, "
            "the first being the reference."
        )
    if not isinstance(pred_prefix, str) or pd.isna(pred_prefix) or pred_prefix == "":
        raise ValueError("`pred_prefix` must be a single non-empty string.")

    factor_lv = list(factor_lv)
    subjects = sa_sim_subject_sizes(n_samples, n_per_subject, use_default_n)
    sizes = subjects["sizes"]
    n_samples = subjects["n_samples"]
    n_unit = n_samples if sizes is None else len(sizes)

    numeric_pred = [f"{pred_prefix}_{i}" for i in range(1, n_pred + 1)]
    factor_pred = (
        [f"{pred_prefix}_cat_{i}" for i in range(1, n_factor_pred + 1)]
        if n_factor_pred > 0
        else []
    )
    constant_pred = (
        [f"{pred_prefix}_const_{i}" for i in range(1, n_constant_pred + 1)]
        if n_constant_pred > 0
        else []
    )

    planted = sa_sim_plant_beta(spec, beta_range)
    if sizes is None:
        values = sa_sim_mvnorm(n_samples, spec["value_mean"], spec["value_sd"], root)
    else:
        between = sa_sim_mvnorm(
            len(sizes),
            spec["value_mean"],
            spec["value_sd"] * np.sqrt(subject_share),
            root,
        )
        within = sa_sim_mvnorm(
            n_samples,
            np.zeros(n_pred),
            spec["value_sd"] * np.sqrt(1 - subject_share),
            root,
        )
        rep_idx = np.repeat(np.arange(len(sizes)), sizes)
        values = between[rep_idx, :] + within

    x = pd.DataFrame(values, columns=numeric_pred)
    eta = values @ planted["beta"]

    offsets_dict: dict[str, dict[str, float]] = {}
    for k in range(n_factor_pred):
        nm = factor_pred[k]
        level = sa_sim_balanced_levels(n_unit, factor_lv)
        if sizes is not None:
            level = pd.Categorical(
                np.repeat(level.astype(str).to_numpy(), sizes),
                categories=factor_lv,
            )
        off = sa_sim_factor_offsets(factor_lv, beta_range)
        offsets_dict[nm] = off
        x[nm] = level
        eta = eta + np.array([off[str(lv)] for lv in level])

    for nm in constant_pred:
        x[nm] = 1.0

    subject: list[str] | None = None
    subject_offset = np.zeros(n_samples, dtype=float)
    if sizes is not None:
        subject = np.repeat(
            [f"subject_{i}" for i in range(1, len(sizes) + 1)],
            sizes,
        ).tolist()
        subject_offset = np.repeat(
            rnorm(len(sizes), 0.0, subject_sd),
            sizes,
        )
    eta = np.asarray(eta + subject_offset, dtype=float).ravel()

    x[numeric_pred] = sa_sim_mask_missing(x[numeric_pred], p_missing)
    x.index = pd.RangeIndex(len(x))

    return {
        "x": x,
        "predictors": numeric_pred + factor_pred + constant_pred,
        "numeric_pred": numeric_pred,
        "factor_pred": factor_pred,
        "constant_pred": constant_pred,
        "beta": planted["beta"],
        "direction": planted["direction"],
        "offsets": offsets_dict,
        "eta": eta,
        "subject": subject,
        "subject_offset": subject_offset,
        "n_samples": n_samples,
        "sizes": sizes,
        "truth": sa_sim_truth_pred(
            planted,
            spec,
            numeric_pred,
            factor_pred,
            constant_pred,
            cor_mat,
        ),
        "truth_term": sa_sim_truth_term(planted["beta"], numeric_pred, offsets_dict),
    }


def sa_sim_truth_pred(
    planted: dict[str, Any],
    spec: dict[str, Any],
    numeric_pred: list[str],
    factor_pred: list[str],
    constant_pred: list[str],
    cor_mat: np.ndarray | None,
) -> pd.DataFrame:
    n_pred = spec["n_pred"]
    signal = np.where(planted["beta"] != 0)[0]
    if cor_mat is None:
        cors = np.eye(n_pred)
    else:
        cors = np.abs(np.asarray(cor_mat, dtype=float))

    max_cor = np.zeros(n_pred, dtype=float)
    for i in range(n_pred):
        others = [j for j in signal if j != i]
        max_cor[i] = 0.0 if len(others) == 0 else float(np.max(cors[i, others]))

    n_other = len(factor_pred) + len(constant_pred)
    return pd.DataFrame(
        {
            "predictors": numeric_pred + factor_pred + constant_pred,
            "role": (
                ["null" if b == 0 else "signal" for b in planted["beta"]]
                + ["factor"] * len(factor_pred)
                + ["constant"] * len(constant_pred)
            ),
            "beta": (
                list(planted["beta"])
                + [np.nan] * len(factor_pred)
                + [0.0] * len(constant_pred)
            ),
            "direction": (
                list(planted["direction"])
                + [None] * len(factor_pred)
                + ["none"] * len(constant_pred)
            ),
            "value_mean": list(spec["value_mean"]) + [np.nan] * n_other,
            "value_sd": list(spec["value_sd"]) + [np.nan] * n_other,
            "max_cor_signal": list(max_cor) + [np.nan] * n_other,
        }
    )


def sa_sim_truth_term(
    beta: np.ndarray,
    numeric_pred: list[str],
    offsets: dict[str, dict[str, float]],
) -> pd.DataFrame:
    terms = list(numeric_pred)
    values = list(beta)
    predictors = list(numeric_pred)

    for nm, off in offsets.items():
        lv = list(off.keys())
        terms.extend(f"{nm}{level}" for level in lv[1:])
        values.extend([off[level] for level in lv[1:]])
        predictors.extend([nm] * (len(lv) - 1))

    return pd.DataFrame(
        {
            "terms": terms,
            "predictors": predictors,
            "beta": values,
        }
    )


def sa_sim_add_intercept(truth_term: pd.DataFrame, intercept: float) -> pd.DataFrame:
    head = pd.DataFrame(
        {
            "terms": ["(Intercept)"],
            "predictors": [None],
            "beta": [intercept],
        }
    )
    return pd.concat([head, truth_term], ignore_index=True)


def sa_sim_split_args(
    data: pd.DataFrame,
    design: dict[str, Any],
    stratify_outcome: bool,
) -> dict[str, Any]:
    if stratify_outcome or design["subject"] is None:
        stratified: str | None = "y"
    elif len(design["factor_pred"]) > 0:
        stratified = design["factor_pred"][0]
    else:
        stratified = None

    return {
        "data": data,
        "stratified": stratified,
        "id": None if design["subject"] is None else "subject",
    }


def sa_sim_design(
    n_control: int,
    n_treat: np.ndarray | list[float],
    group_lv: list[str] | None,
    use_default: bool,
    paired: bool,
) -> dict[str, Any]:
    treat_arr = np.asarray(n_treat, dtype=float)
    if treat_arr.size == 0:
        raise ValueError(
            "`n_treat` must be one or more group sizes, one per treatment group."
        )

    if group_lv is None:
        if treat_arr.size < 2:
            raise ValueError(
                "`n_treat` holds one size per treatment group, and there must be "
                "at least two of them for a comparison of three or more levels. "
                f"Pass a size per group, such as `n_treat = rep({treat_arr[0]}, 3)`, "
                "or use simulate_two_groups() for two groups in all."
            )
        group_lv = ["control"] + [f"treat_{i}" for i in range(1, len(treat_arr) + 1)]
    else:
        if (
            not isinstance(group_lv, (list, tuple))
            or len(group_lv) < 3
            or any(pd.isna(lv) for lv in group_lv)
            or len(set(group_lv)) != len(group_lv)
        ):
            raise ValueError(
                "`group_lv` must be at least three distinct non-missing group "
                "labels, the first being the control."
            )
        n_wanted = len(group_lv) - 1
        if use_default or treat_arr.size == 1:
            treat_arr = np.full(n_wanted, treat_arr[0])
        if treat_arr.size != n_wanted:
            raise ValueError(
                f"`group_lv` names {n_wanted} treatment group(s) after the "
                f"control, but `n_treat` gives {treat_arr.size} size(s)."
            )

    sizes = [sa_check_count(n_control, "n_control", 2)] + [
        sa_check_count(float(treat_arr[k]), f"n_treat[{k + 1}]", 2)
        for k in range(treat_arr.size)
    ]

    if paired and len(set(sizes)) > 1:
        raise ValueError(
            "`paired = TRUE` measures every condition on the same subjects, so "
            "every group holds the same number of them, but the sizes given are "
            f"{', '.join(str(s) for s in sizes)}."
        )

    return {"group_lv": list(group_lv), "sizes": sizes}


def sa_sim_pattern_mix(
    pattern_mix: dict[str, float] | pd.Series,
    known: tuple[str, ...] = ("all", "gradient", "single"),
    arg: str = "pattern_mix",
) -> dict[str, float]:
    if isinstance(pattern_mix, pd.Series):
        names = pattern_mix.index.tolist()
        values = pattern_mix.to_numpy(dtype=float)
    elif isinstance(pattern_mix, dict):
        names = list(pattern_mix.keys())
        values = np.asarray(list(pattern_mix.values()), dtype=float)
    else:
        names = []
        values = np.array([], dtype=float)

    if (
        values.size == 0
        or not names
        or any(n is None or (isinstance(n, float) and np.isnan(n)) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            f"`{arg}` must be a named numeric vector of weights with one "
            "entry per shape and no duplicates. Known shapes are: "
            f"{', '.join(known)}."
        )

    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(
            f"`{arg}` names unknown shape(s): {', '.join(unknown)}. "
            f"Known shapes are: {', '.join(known)}."
        )
    if np.any(values < 0):
        raise ValueError(f"`{arg}` weights must not be negative.")
    if float(np.sum(values)) <= 0:
        raise ValueError(
            f"`{arg}` needs at least one positive weight, otherwise there "
            "is no shape left to plant an effect in."
        )

    return {n: float(v) for n, v in zip(names, values) if v > 0}


def sa_sim_allocate(n: int, weights: dict[str, float]) -> dict[str, int]:
    names = list(weights.keys())
    w = np.array([weights[k] for k in names], dtype=float)
    out_arr = np.zeros(len(names), dtype=int)
    if n == 0:
        return dict(zip(names, out_arr.tolist()))

    share = n * w / w.sum()
    out_arr = np.floor(share).astype(int)
    short = n - int(out_arr.sum())
    if short > 0:
        remainder = share - out_arr
        order = np.argsort(-remainder, kind="stable")
        out_arr[order[:short]] += 1
    return dict(zip(names, out_arr.tolist()))


def sa_sim_pattern_delta(d: float, pattern: str, n_groups: int) -> np.ndarray:
    if pattern == "all":
        return np.full(n_groups, d, dtype=float)
    if pattern == "gradient":
        return d * np.arange(1, n_groups + 1, dtype=float) / n_groups
    if pattern == "single":
        out = np.zeros(n_groups, dtype=float)
        out[sample_int(n_groups, 1)[0] - 1] = d
        return out
    raise RuntimeError(f"internal error: unknown effect shape `{pattern}`.")


def sa_sim_truth(
    feats: list[str],
    delta: np.ndarray,
    group_lv: list[str],
    pattern: np.ndarray,
    direction: np.ndarray,
    baseline: np.ndarray,
    sd_subject: np.ndarray,
) -> pd.DataFrame:
    treat_delta = delta[:, 1:]
    abs_delta = np.abs(treat_delta)
    largest = np.max(abs_delta, axis=1)
    tied = np.sum(abs_delta == largest[:, None], axis=1) > 1
    which_max = np.argmax(abs_delta, axis=1)

    extreme_level = np.array([group_lv[1 + j] for j in which_max], dtype=object)
    extreme_level[largest == 0] = None

    log2fc = treat_delta[np.arange(len(feats)), which_max]

    return pd.DataFrame(
        {
            "features": feats,
            "pattern": pattern,
            "direction": direction,
            "extreme_level": extreme_level,
            "extreme_tied": tied,
            "log2fc": log2fc,
            "baseline": baseline,
            "sd_subject": sd_subject,
        }
    )


def sa_sim_truth_group(
    feats: list[str],
    delta: np.ndarray,
    center: np.ndarray,
    sd_mat: np.ndarray,
    group_lv: list[str],
    sizes: list[int],
) -> pd.DataFrame:
    n_lv = len(group_lv)
    n_feat = len(feats)
    return pd.DataFrame(
        {
            "features": np.repeat(feats, n_lv),
            "group": np.tile(group_lv, n_feat),
            "is_ref": np.tile(np.arange(n_lv) == 0, n_feat),
            "delta": delta.T.reshape(-1, order="F"),
            "center": center.T.reshape(-1, order="F"),
            "sd": sd_mat.T.reshape(-1, order="F"),
            "n": np.tile(sizes, n_feat),
        }
    )


def sa_sim_truth_contrast(
    feats: list[str],
    delta: np.ndarray,
    group_lv: list[str],
) -> pd.DataFrame:
    pairs = sa_level_pairs(group_lv)
    i_idx = pairs["i"].to_numpy(dtype=int)
    j_idx = pairs["j"].to_numpy(dtype=int)
    diffs = delta[:, i_idx] - delta[:, j_idx]
    flat = diffs.T.reshape(-1, order="F")

    n_pairs = len(pairs)
    n_feat = len(feats)
    return pd.DataFrame(
        {
            "features": np.repeat(feats, n_pairs),
            "contrast": np.tile(pairs["contrast"].to_numpy(), n_feat),
            "group1": np.tile(pairs["group1"].to_numpy(), n_feat),
            "group2": np.tile(pairs["group2"].to_numpy(), n_feat),
            "delta": flat,
            "is_diff": flat != 0,
        }
    )
