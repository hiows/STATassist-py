"""Fold change between the two group levels of a comparison."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.validate import sa_feature_table, sa_level_pairs


def sa_resolve_fc_mean(
    fc_mean: str | None,
    input_scale: str,
    use_default: bool,
) -> str:
    if use_default:
        return "geom" if input_scale == "log2" else "arith"
    if fc_mean not in ("arith", "geom"):
        raise ValueError("`fc_mean` must be one of 'arith' or 'geom'.")
    return fc_mean


def sa_fc_center(
    v: np.ndarray | Sequence[float],
    side: str,
    mean_type: str,
    input_scale: str = "raw",
) -> float:
    arr = np.asarray(v, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"no usable observation left in the {side} group.")

    if input_scale == "log2":
        arr = np.power(2.0, arr)
        n_over = int(np.sum(~np.isfinite(arr)))
        if n_over > 0:
            raise ValueError(
                f"2^x overflows to infinity for {n_over} value(s) in the {side} "
                'group, so these observations are not on the log2 scale; use '
                '`input_scale = "raw"` instead.'
            )

    if mean_type == "arith":
        return float(np.mean(arr))

    n_nonpos = int(np.sum(arr <= 0))
    if n_nonpos > 0:
        raise ValueError(
            f"the geometric mean is undefined for the {n_nonpos} value(s) at or "
            f"below zero in the {side} group; use `fc_mean = \"arith\"` instead."
        )
    return float(np.exp(np.mean(np.log(arr))))


def sa_fold_change(
    samples: Sequence[dict[str, np.ndarray]],
    feats: Sequence[str],
    group_lv: Sequence[str],
    mean_type: str,
    input_scale: str = "raw",
) -> Any:
    import pandas as pd

    label = (
        "Arithmetic mean fold change"
        if mean_type == "arith"
        else "Geometric mean fold change"
    )

    def _row(i: int) -> pd.Series:
        s = samples[i]
        x_center = sa_fc_center(s["x"], group_lv[0], mean_type, input_scale)
        y_center = sa_fc_center(s["y"], group_lv[1], mean_type, input_scale)
        fold_change = x_center / y_center
        with np.errstate(invalid="ignore", divide="ignore"):
            log2fc = np.log2(fold_change)
        return pd.Series(
            {
                "x_center": x_center,
                "y_center": y_center,
                "fold_change": fold_change,
                "log2fc": log2fc,
            }
        )

    out = sa_feature_table(
        feats,
        ["x_center", "y_center", "fold_change", "log2fc"],
        label,
        _row,
        p_adjust_method=None,
    )

    def _report(mask: np.ndarray, what: str) -> None:
        hit = out.loc[mask.fillna(False), "features"].tolist()
        if hit:
            print(
                f"Fold change: {what} for {len(hit)} feature(s): "
                f"{', '.join(hit)}."
            )

    _report(
        out["y_center"] == 0,
        f"the {group_lv[1]} centre is zero, so `fold_change` is infinite",
    )
    _report(
        (out["x_center"] == 0) & (out["y_center"] != 0),
        f"the {group_lv[0]} centre is zero, so `log2fc` is -Inf and clears any cutoff",
    )
    _report(
        (out["x_center"] * out["y_center"]) < 0,
        "the two centres have opposite signs, so `log2fc` is NaN",
    )

    return out


def sa_feature_samples(
    x: Any,
    group_lv: list[str],
    paired: bool,
) -> dict[str, np.ndarray]:
    if not paired:
        return x
    mat = np.asarray(x) if not isinstance(x, pd.DataFrame) else x.to_numpy()
    return {group_lv[j]: mat[:, j] for j in range(len(group_lv))}


def sa_group_centers(
    per_feature: list[Any],
    feats: list[str],
    group_lv: list[str],
    mean_type: str,
    paired: bool,
    input_scale: str = "raw",
) -> dict[str, Any]:
    n_lv = len(group_lv)
    centers = np.full((len(feats), n_lv), np.nan)
    errors = [None] * len(feats)
    n_used = np.full(len(feats), np.nan)

    for i, f in enumerate(feats):
        samples = sa_feature_samples(per_feature[i], group_lv, paired)
        try:
            row = np.array(
                [
                    sa_fc_center(samples[lv], lv, mean_type, input_scale)
                    for lv in group_lv
                ]
            )
            centers[i, :] = row
            if paired:
                n_used[i] = per_feature[i].shape[0]
            else:
                n_used[i] = sum(samples[lv].size for lv in group_lv)
        except Exception as exc:
            errors[i] = str(exc)

    return {
        "centers": pd.DataFrame(centers, index=feats, columns=group_lv),
        "errors": errors,
        "n_used": n_used,
    }


def sa_multi_fold_change(
    centers: dict[str, Any],
    feats: list[str],
    group_lv: list[str],
    mean_type: str,
) -> Any:
    label = (
        "Arithmetic mean fold change"
        if mean_type == "arith"
        else "Geometric mean fold change"
    )

    def _row(i: int) -> pd.Series:
        if centers["errors"][i]:
            raise ValueError(centers["errors"][i])
        row = centers["centers"].iloc[i].to_numpy(dtype=float)
        ratios = row / row[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ratios = np.log2(ratios)
        rankable = np.isfinite(log_ratios)
        rankable[0] = False
        if np.any(rankable):
            idx = np.where(rankable)[0][np.argmax(np.abs(log_ratios[rankable]))]
        else:
            idx = 1
        return pd.Series(
            {
                "n_used": centers["n_used"][i],
                "n_groups": len(row),
                "ref_center": row[0],
                "extreme_index": idx + 1,
                "extreme_center": row[idx],
                "fold_change": ratios[idx],
                "log2fc": log_ratios[idx],
            }
        )

    out = sa_feature_table(
        feats,
        [
            "n_used",
            "n_groups",
            "ref_center",
            "extreme_index",
            "extreme_center",
            "fold_change",
            "log2fc",
        ],
        label,
        _row,
        p_adjust_method=None,
    )
    out["extreme_level"] = [group_lv[int(i) - 1] for i in out["extreme_index"]]
    return out[
        [
            "features",
            "n_used",
            "n_groups",
            "ref_center",
            "extreme_level",
            "extreme_center",
            "fold_change",
            "log2fc",
        ]
    ]


def sa_pairwise_tables(
    posthoc_tbl: pd.DataFrame,
    centers: pd.DataFrame,
    feats: list[str],
    group_lv: list[str],
) -> dict[str, pd.DataFrame]:
    from statassist.contracts.comparison import sa_posthoc_stat_columns

    pairs = sa_level_pairs(group_lv)
    stat_cols = sa_posthoc_stat_columns()
    out: dict[str, pd.DataFrame] = {}

    for _, pr in pairs.iterrows():
        g1, g2 = pr["group1"], pr["group2"]
        ratios = (centers[g1] / centers[g2]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            log2fc = np.log2(ratios)
        tbl = pd.DataFrame(
            {
                "features": feats,
                "contrast": pr["contrast"],
                "group1": g1,
                "group2": g2,
                "fold_change": ratios,
                "log2fc": log2fc,
            }
        )
        rows = posthoc_tbl.loc[posthoc_tbl["contrast"] == pr["contrast"]]
        for col in stat_cols:
            mapped = rows.set_index("features")[col].reindex(feats)
            tbl[col] = mapped.to_numpy()
        out[pr["contrast"]] = tbl.reset_index(drop=True)
    return out

