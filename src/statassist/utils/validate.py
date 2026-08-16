"""Input validation helpers shared by exported functions."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# R stats::p.adjust.methods
P_ADJUST_METHODS: tuple[str, ...] = (
    "holm",
    "hochberg",
    "hommel",
    "bonferroni",
    "BH",
    "BY",
    "fdr",
    "none",
)

_METHOD_TO_STATSMODELS: dict[str, str] = {
    "holm": "holm",
    "hochberg": "hochberg",
    "hommel": "hommel",
    "bonferroni": "bonferroni",
    "BH": "fdr_bh",
    "BY": "fdr_by",
    "fdr": "fdr_bh",
    "none": "none",
}


def sa_check_feat_names(feats: Sequence[str]) -> list[str]:
    if not isinstance(feats, (list, tuple, np.ndarray, pd.Index)):
        raise ValueError(
            "`feats` must be a non-empty character vector of feature names."
        )
    feats_list = [str(f) for f in feats]
    if len(feats_list) == 0:
        raise ValueError(
            "`feats` must be a non-empty character vector of feature names."
        )
    if any(f is None or (isinstance(f, float) and np.isnan(f)) for f in feats_list):
        raise ValueError("`feats` must not contain NA.")
    dup = sorted({f for f in feats_list if feats_list.count(f) > 1})
    if dup:
        raise ValueError(
            f"`feats` contains duplicated names: {', '.join(dup)}"
        )
    return feats_list


def sa_validate_wide_input(
    data: pd.DataFrame | np.ndarray,
    feats: Sequence[str],
    group: Sequence[Any] | None,
    group_lv: Sequence[str] | None,
    id: Sequence[Any] | None = None,
    *,
    n_levels: int | None = None,
    min_levels: int = 2,
) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data.frame or a matrix.")
    if len(data) == 0:
        raise ValueError("`data` has zero rows.")

    feats = sa_check_feat_names(feats)
    unknown = [f for f in feats if f not in data.columns]
    if unknown:
        raise ValueError(
            f"`feats` not found in `data`: {', '.join(unknown)}"
        )
    non_numeric = [f for f in feats if not pd.api.types.is_numeric_dtype(data[f])]
    if non_numeric:
        raise ValueError(
            "`feats` must refer to numeric columns. Not numeric: "
            f"{', '.join(non_numeric)}"
        )

    if group is None and group_lv is None:
        if id is not None and len(id) != len(data):
            raise ValueError(
                f"`id` must have one entry per row of `data`: got "
                f"{len(id)} for {len(data)} rows."
            )
        return {
            "data": data,
            "feats": feats,
            "group": None,
            "id": None if id is None else [str(x) for x in id],
            "n_dropped": 0,
        }

    if group is None or group_lv is None:
        raise ValueError(
            "`group` and `group_lv` must both be supplied or both be `NULL`."
        )

    if len(group) != len(data):
        raise ValueError(
            f"`group` must have one entry per row of `data`: got "
            f"{len(group)} for {len(data)} rows."
        )
    group_chr = [str(g) for g in group]

    if id is not None and len(id) != len(data):
        raise ValueError(
            f"`id` must have one entry per row of `data`: got "
            f"{len(id)} for {len(data)} rows."
        )

    if len(group_lv) == 0:
        raise ValueError("`group_lv` must be a non-empty vector of group levels.")
    group_lv = [str(lv) for lv in group_lv]
    if any(lv is None or lv == "nan" for lv in group_lv):
        raise ValueError("`group_lv` must not contain NA.")
    dup_lv = sorted({lv for lv in group_lv if group_lv.count(lv) > 1})
    if dup_lv:
        raise ValueError(
            f"`group_lv` contains duplicated levels: {', '.join(dup_lv)}"
        )
    if n_levels is not None and len(group_lv) != n_levels:
        raise ValueError(
            f"`group_lv` must contain exactly {n_levels} levels, but "
            f"{len(group_lv)} were given: {', '.join(group_lv)}."
        )
    if len(group_lv) < min_levels:
        raise ValueError(
            f"`group_lv` must contain at least {min_levels} levels."
        )
    absent = [lv for lv in group_lv if lv not in group_chr]
    if absent:
        raise ValueError(
            f"`group_lv` level(s) absent from `group`: {', '.join(absent)}"
        )

    keep = np.array([g in group_lv for g in group_chr])
    n_dropped = int((~keep).sum())
    data_f = data.loc[keep].reset_index(drop=True)
    group_f = pd.Categorical(
        [group_chr[i] for i in np.where(keep)[0]],
        categories=group_lv,
        ordered=True,
    )
    id_f = None if id is None else [str(x) for i, x in enumerate(id) if keep[i]]

    return {
        "data": data_f,
        "feats": feats,
        "group": group_f,
        "id": id_f,
        "n_dropped": n_dropped,
    }


def sa_control_first(
    group_lv: Sequence[str],
    control_label: str | None,
    *,
    arg: str = "control_label",
    lv_arg: str = "group_lv",
) -> list[str]:
    group_lv = list(group_lv)
    if control_label is None:
        return group_lv
    if not isinstance(control_label, str) or control_label != control_label:
        raise ValueError(
            f"`{arg}` must be a single level name, the one to hold as the reference."
        )
    control_label = str(control_label)
    if control_label not in group_lv:
        present = ", ".join(group_lv)
        raise ValueError(
            f"`{arg}` names a level `{lv_arg}` does not hold: "
            f"{control_label}. Present: {present}."
        )
    rest = [lv for lv in group_lv if lv != control_label]
    return [control_label] + rest


def sa_check_flag(x: bool, arg: str) -> bool:
    if not isinstance(x, (bool, np.bool_)) or x is None:
        raise ValueError(f"`{arg}` must be TRUE or FALSE.")
    return bool(x)


def sa_check_scalar_num(
    x: float,
    arg: str,
    lower: float = -np.inf,
    upper: float = np.inf,
    *,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if not np.isscalar(x) or not np.isfinite(x):
        raise ValueError(f"`{arg}` must be a single non-missing number.")
    too_low = x <= lower if lower_open else x < lower
    too_high = x >= upper if upper_open else x > upper
    if too_low or too_high:
        lo_br = "(" if lower_open else "["
        hi_br = ")" if upper_open else "]"
        raise ValueError(
            f"`{arg}` must be in {lo_br}{lower}, {upper}{hi_br}, but is {x}."
        )
    return float(x)


def sa_check_count(x: float, arg: str, lower: int = 0) -> int:
    val = sa_check_scalar_num(x, arg, lower)
    if not np.isfinite(val) or val != int(val):
        raise ValueError(
            f"`{arg}` must be a finite whole number, but is {val}."
        )
    return int(val)


def sa_check_range(
    x: Sequence[float],
    arg: str,
    lower: float = -np.inf,
) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"`{arg}` must be a finite numeric vector of length 2.")
    if arr[0] > arr[1]:
        raise ValueError(
            f"`{arg}` must be increasing, but is c({arr[0]}, {arr[1]})."
        )
    if arr[0] < lower:
        raise ValueError(
            f"`{arg}` must not go below {lower}, but starts at {arr[0]}."
        )
    return float(arr[0]), float(arr[1])


def sa_check_lim(
    x: Sequence[float] | None,
    arg: str,
) -> tuple[float, float] | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{arg}` must be NULL or a finite numeric vector of length 2."
        )
    return float(arr[0]), float(arr[1])


def sa_check_p_adjust(x: str, arg: str) -> str:
    if not isinstance(x, str) or x not in P_ADJUST_METHODS:
        raise ValueError(
            f"`{arg}` must be one of: {', '.join(P_ADJUST_METHODS)}."
        )
    return x


def sa_pair_by_order(
    group: pd.Categorical | Sequence[Any],
    group_lv: Sequence[str],
) -> dict[str, Any]:
    group_arr = np.asarray(group)
    idx_x = np.where(group_arr == group_lv[0])[0]
    idx_y = np.where(group_arr == group_lv[1])[0]
    if len(idx_x) != len(idx_y):
        raise ValueError(
            "`paired = TRUE` without `id` pairs observations by row order, which "
            "requires the same number of rows per group. Got "
            f"{group_lv[0]} = {len(idx_x)}, {group_lv[1]} = {len(idx_y)}. "
            "Supply `id` to match on a pairing key instead."
        )
    return {
        "idx_x": idx_x,
        "idx_y": idx_y,
        "unmatched": [],
    }


def sa_pair_by_id(
    id: Sequence[str],
    group: pd.Categorical | Sequence[Any],
    group_lv: Sequence[str],
) -> dict[str, Any]:
    id = [str(x) for x in id]
    if any(x is None or x == "nan" for x in id):
        raise ValueError(
            "`id` must not contain NA when it is used to form pairs."
        )
    group_arr = np.asarray(group)
    idx_x = np.where(group_arr == group_lv[0])[0]
    idx_y = np.where(group_arr == group_lv[1])[0]
    id_x = [id[i] for i in idx_x]
    id_y = [id[i] for i in idx_y]

    repeated = sorted(
        set(x for x in id_x if id_x.count(x) > 1)
        | set(x for x in id_y if id_y.count(x) > 1)
    )
    if repeated:
        raise ValueError(
            "`id` must be unique within each group, otherwise the pairing is "
            f"ambiguous. Repeated id(s): {', '.join(repeated)}."
        )

    common = [x for x in id_x if x in set(id_y)]
    if len(common) < 2:
        raise ValueError(
            f"only {len(common)} id(s) appear in both `{group_lv[0]}` and "
            f"`{group_lv[1]}`; at least 2 pairs are needed."
        )

    id_x_map = {v: i for i, v in zip(idx_x, id_x)}
    id_y_map = {v: i for i, v in zip(idx_y, id_y)}
    return {
        "idx_x": np.array([id_x_map[c] for c in common]),
        "idx_y": np.array([id_y_map[c] for c in common]),
        "unmatched": sorted(set(id_x) | set(id_y) - set(common)),
    }


def sa_na_row(nms: Sequence[str]) -> pd.Series:
    return pd.Series({n: np.nan for n in nms})


def sa_row(**kwargs: float) -> pd.Series:
    return pd.Series({k: float(np.asarray(v).ravel()[0]) for k, v in kwargs.items()})


def sa_add_padj(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    out["pval_adj"] = p_adjust(out["pval"].to_numpy(), method)
    cols = [c for c in out.columns if c != "pval_adj"]
    pval_idx = cols.index("pval")
    ordered = cols[: pval_idx + 1] + ["pval_adj"] + cols[pval_idx + 1 :]
    return out[ordered]


def p_adjust(pvalues: np.ndarray, method: str) -> np.ndarray:
    """Apply multiplicity adjustment matching R stats::p.adjust."""
    sa_check_p_adjust(method, "method")
    p = np.asarray(pvalues, dtype=float)
    if method == "none":
        return p.copy()
    sm_method = _METHOD_TO_STATSMODELS[method]
    _, adj, _, _ = multipletests(p, method=sm_method)
    return adj


def sa_feature_table(
    feats: Sequence[str],
    columns: Sequence[str],
    label: str,
    fun: Callable[[int], pd.Series],
    p_adjust_method: str | None = "none",
) -> pd.DataFrame:
    failures: dict[str, str] = {}
    notes: dict[str, str] = {}
    rows: list[pd.Series] = []

    for i, feat in enumerate(feats):
        caught_notes: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                row = fun(i)
                for w in caught:
                    caught_notes.append(str(w.message))
        except Exception as exc:
            failures[feat] = str(exc)
            row = sa_na_row(columns)

        if caught_notes:
            notes[feat] = "; ".join(dict.fromkeys(caught_notes))

        absent = [c for c in columns if c not in row.index]
        if absent:
            raise RuntimeError(
                f"internal error: {label} row for `{feat}` is missing column(s): "
                f"{', '.join(absent)}"
            )
        rows.append(row[list(columns)])

    out = pd.DataFrame(rows)
    out.insert(0, "features", list(feats))
    out = out.reset_index(drop=True)

    if p_adjust_method is not None:
        out = sa_add_padj(out, p_adjust_method)

    if notes:
        grouped: dict[str, int] = {}
        for msg in notes.values():
            grouped[msg] = grouped.get(msg, 0) + 1
        lines = "\n".join(
            f"  [{n} feature(s)] {msg}" for msg, n in grouped.items()
        )
        print(
            f"{label}: engine note(s) for {len(notes)} of "
            f"{len(feats)} feature(s):\n{lines}"
        )

    if failures:
        lines = "\n".join(
            f"  {feat}: {msg}" for feat, msg in failures.items()
        )
        warnings.warn(
            f"{label} could not be computed for {len(failures)} of "
            f"{len(feats)} feature(s); those rows are NA:\n{lines}",
            RuntimeWarning,
            stacklevel=2,
        )

    return out


def sa_resolve_row_vector(
    x: Any,
    arg: str,
    data: pd.DataFrame,
    *,
    allow_na: bool = False,
) -> dict[str, Any]:
    if x is None:
        return {"value": None, "label": None}

    label = "<vector>"
    if isinstance(x, str) and x in data.columns:
        label = x
        x = data[x]

    if len(x) != len(data):
        raise ValueError(
            f"`{arg}` must name a column of `data` or hold one entry per row "
            f"of it: got {len(x)} for {len(data)} row(s)."
        )
    if not allow_na and pd.isna(x).any():
        raise ValueError(
            f"`{arg}` must not contain NA: a row it does not describe cannot "
            "be assigned to a side of the split."
        )
    return {"value": x, "label": label}


def sa_check_count(x: float, arg: str, lower: float = 0) -> int:
    sa_check_scalar_num(x, arg, lower)
    if not np.isfinite(x) or x != int(x):
        raise ValueError(f"`{arg}` must be a finite whole number, but is {x}.")
    return int(x)


def sa_check_num_vector(
    x: Sequence[float],
    arg: str,
    lower: float = -np.inf,
    upper: float = np.inf,
) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{arg}` must be a non-empty numeric vector of finite values."
        )
    bad = np.unique(arr[(arr < lower) | (arr > upper)])
    if bad.size > 0:
        raise ValueError(
            f"`{arg}` must be in [{lower}, {upper}], but holds "
            f"{', '.join(str(v) for v in bad)}."
        )
    return arr


def sa_check_range(
    x: Sequence[float],
    arg: str,
    lower: float = -np.inf,
) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"`{arg}` must be a finite numeric vector of length 2.")
    if arr[0] > arr[1]:
        raise ValueError(
            f"`{arg}` must be increasing, but is c({arr[0]}, {arr[1]})."
        )
    if arr[0] < lower:
        raise ValueError(
            f"`{arg}` must not go below {lower}, but starts at {arr[0]}."
        )
    return arr


@contextmanager
def sa_preserve_seed(seed: float | None) -> Generator[None, None, None]:
    import random

    if seed is None:
        yield
        return
    sa_check_scalar_num(seed, "seed")
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def sa_check_margin(x: Sequence[float], arg: str = "margin") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size != 4 or np.any(np.isnan(arr)) or np.any(arr < 0):
        raise ValueError(
            f"`{arg}` must be a numeric vector of 4 non-negative values."
        )
    return arr


def sa_check_lim(x: Sequence[float] | None, arg: str) -> Sequence[float] | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{arg}` must be NULL or a finite numeric vector of length 2."
        )
    return arr


def sa_check_pvalues(pvalue: np.ndarray, arg: str = "pvalue") -> np.ndarray:
    p = np.asarray(pvalue, dtype=float)
    bad = np.where(
        ~np.isnan(p) & (~np.isfinite(p) | (p < 0) | (p > 1))
    )[0]
    if bad.size > 0:
        shown = bad[:5]
        suffix = ", ..." if bad.size > 5 else ""
        raise ValueError(
            f"`{arg}` must lie in [0, 1]. Offending position(s): "
            f"{', '.join(str(i + 1) for i in shown)}{suffix}."
        )
    return p


def sa_align_by_subject(
    id: Sequence[str],
    group: pd.Categorical | Sequence[Any],
    group_lv: Sequence[str],
) -> dict[str, Any]:
    id = [str(x) for x in id]
    if any(x is None or x == "nan" for x in id):
        raise ValueError(
            "`id` must not contain NA when it is used to align conditions."
        )
    group_arr = np.asarray(group)
    per_level: dict[str, dict[str, int]] = {}
    for lv in group_lv:
        rows = np.where(group_arr == lv)[0]
        ids = [id[i] for i in rows]
        repeated = sorted({x for x in ids if ids.count(x) > 1})
        if repeated:
            raise ValueError(
                "`id` must be unique within each condition, otherwise the design "
                f"is ambiguous. Repeated id(s) in `{lv}`: {', '.join(repeated)}."
            )
        per_level[lv] = {v: int(r) for v, r in zip(ids, rows)}

    all_ids = list(dict.fromkeys(x for lv in group_lv for x in per_level[lv]))
    complete = set(per_level[group_lv[0]])
    for lv in group_lv[1:]:
        complete &= set(per_level[lv])
    complete = [x for x in all_ids if x in complete]

    if len(complete) < 2:
        raise ValueError(
            f"only {len(complete)} subject(s) have all {len(group_lv)} "
            "condition(s); at least 2 complete subjects are needed."
        )

    idx = np.column_stack(
        [[per_level[lv][s] for s in complete] for lv in group_lv]
    )
    return {
        "idx": idx,
        "subjects": complete,
        "unmatched": sorted(set(all_ids) - set(complete)),
    }


def sa_posthoc_stat_columns() -> list[str]:
    return [
        c
        for c in sa_posthoc_table_columns()
        if c not in ("features", "contrast", "group1", "group2")
    ]


def sa_posthoc_table(
    feats: list[str],
    group_lv: list[str],
    columns: list[str],
    label: str,
    fun: Callable[[str], np.ndarray],
    p_adjust_method: str = "holm",
) -> pd.DataFrame:
    from statassist.contracts.comparison import sa_posthoc_table_columns
    from statassist.kernels.posthoc import sa_posthoc_columns

    pairs = sa_level_pairs(group_lv)
    n_pairs = len(pairs)
    empty_cols = sa_posthoc_table_columns()
    if not feats:
        empty = pd.DataFrame(columns=empty_cols)
        return empty

    failures: dict[str, str] = {}
    blocks: list[pd.DataFrame] = []
    posthoc_cols = sa_posthoc_columns()

    for f in feats:
        try:
            mat = fun(f)
            if mat.shape[0] != n_pairs:
                raise RuntimeError(
                    f"internal error: {label} returned {mat.shape[0]} row(s) for "
                    f"`{f}`, expected {n_pairs}."
                )
            row_dict = {c: mat[:, i] for i, c in enumerate(posthoc_cols)}
        except Exception as exc:
            failures[f] = str(exc)
            row_dict = {c: np.full(n_pairs, np.nan) for c in posthoc_cols}

        block = pd.DataFrame(
            {
                "features": [f] * n_pairs,
                "contrast": pairs["contrast"].tolist(),
                "group1": pairs["group1"].tolist(),
                "group2": pairs["group2"].tolist(),
                **row_dict,
            }
        )
        block["pval_adj"] = p_adjust(block["pval"].to_numpy(), p_adjust_method)
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    if failures:
        lines = "\n".join(f"  {feat}: {msg}" for feat, msg in failures.items())
        warnings.warn(
            f"{label} could not be computed for {len(failures)} of "
            f"{len(feats)} feature(s); those rows are NA:\n{lines}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out[sa_posthoc_table_columns()]


def sa_split_for_screening(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Sequence[Any] | None,
    group_lv: Sequence[str] | None,
) -> dict[str, Any]:
    if group is None:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a data.frame or a matrix.")
        if len(data) == 0:
            raise ValueError("`data` has zero rows.")
        feats = sa_check_feat_names(feats)
        unknown = [f for f in feats if f not in data.columns]
        if unknown:
            raise ValueError(f"`feats` not found in `data`: {', '.join(unknown)}")
        non_numeric = [f for f in feats if not pd.api.types.is_numeric_dtype(data[f])]
        if non_numeric:
            raise ValueError(
                "`feats` must refer to numeric columns. Not numeric: "
                f"{', '.join(non_numeric)}"
            )
        return {
            "data": data,
            "rows": {"all": list(range(len(data)))},
            "row_id": np.arange(1, len(data) + 1),
            "grouped": False,
        }

    if group_lv is None:
        group_lv = sorted({str(g) for g in group})
    inp = sa_validate_wide_input(
        data, feats, group, group_lv, min_levels=1
    )
    if inp["n_dropped"] > 0:
        print(
            f"Dropped {inp['n_dropped']} row(s) belonging to a level outside "
            "`group_lv`."
        )
    rows = {
        lv: np.where(np.asarray(inp["group"]) == lv)[0].tolist()
        for lv in inp["group"].categories
    }
    group_chr = [str(g) for g in group]
    row_id = np.where(np.array([g in group_lv for g in group_chr]))[0] + 1
    return {
        "data": inp["data"],
        "rows": rows,
        "row_id": row_id,
        "grouped": True,
    }


def sa_level_pairs(group_lv: Sequence[str]) -> pd.DataFrame:
    from itertools import combinations

    pairs = list(combinations(range(len(group_lv)), 2))
    rows = []
    for i, j in pairs:
        rows.append(
            {
                "i": j,
                "j": i,
                "group1": group_lv[j],
                "group2": group_lv[i],
                "contrast": f"{group_lv[j]} - {group_lv[i]}",
            }
        )
    return pd.DataFrame(rows)
