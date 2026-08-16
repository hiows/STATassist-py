"""Flag candidate outliers without removing them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statassist.utils.validate import (
    sa_check_scalar_num,
    sa_split_for_screening,
)


def _sa_flag_outliers(
    v: np.ndarray,
    criterion: str,
    iqr_multiplier: float,
    z_threshold: float,
    alpha: float,
) -> dict[str, np.ndarray]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    flag = np.zeros(n, dtype=bool)
    score = np.full(n, np.nan)

    if n == 0:
        return {"flag": flag, "score": score}

    if criterion == "iqr":
        q1, q3 = np.quantile(v, [0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        below = v < lower
        above = v > upper
        flag = below | above
        score = np.where(
            below,
            (lower - v) / iqr if iqr > 0 else 0,
            np.where(above, (v - upper) / iqr if iqr > 0 else 0, 0),
        )
    elif criterion == "robust_z":
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad <= 0:
            return {"flag": flag, "score": score}
        rz = 0.6745 * (v - med) / mad
        score = np.abs(rz)
        flag = score > z_threshold
    elif criterion == "grubbs":
        if n < 3:
            return {"flag": flag, "score": score}
        mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        if sd <= 0:
            return {"flag": flag, "score": score}
        g = np.abs(v - mean) / sd
        g_max = float(np.max(g))
        t = g_max * np.sqrt((n - 2) / (n - 1 - g_max**2))
        pval = 2 * stats.t.sf(t, n - 2)
        idx = int(np.argmax(g))
        score[idx] = g_max
        if pval < alpha:
            flag[idx] = True
    else:
        raise ValueError('`criterion` must be one of "iqr", "robust_z", or "grubbs".')

    return {"flag": flag, "score": score}


def screen_outliers(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: Any | None = None,
    group_lv: list[str] | None = None,
    *,
    criterion: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if criterion not in ("iqr", "robust_z", "grubbs"):
        raise ValueError('`criterion` must be one of "iqr", "robust_z", or "grubbs".')
    sa_check_scalar_num(iqr_multiplier, "iqr_multiplier", 0)
    sa_check_scalar_num(z_threshold, "z_threshold", 0, lower_open=True)
    sa_check_scalar_num(alpha, "alpha", 0, 1, lower_open=True)

    split = sa_split_for_screening(data, feats, group, group_lv)
    blocks: list[pd.DataFrame] = []

    for f in feats:
        for lv, rows in split["rows"].items():
            v = split["data"][f].to_numpy()[rows]
            res = _sa_flag_outliers(
                v, criterion, iqr_multiplier, z_threshold, alpha
            )
            hit = np.where(res["flag"])[0]
            if hit.size == 0:
                continue
            blocks.append(
                pd.DataFrame(
                    {
                        "features": [f] * hit.size,
                        "group": [lv if split["grouped"] else np.nan] * hit.size,
                        "row": split["row_id"][np.array(rows)[hit]],
                        "value": v[hit],
                        "score": res["score"][hit],
                    }
                )
            )

    if blocks:
        out = pd.concat(blocks, ignore_index=True)
    else:
        out = pd.DataFrame(
            columns=["features", "group", "row", "value", "score"]
        )
    out.attrs.update(
        {
            "criterion": criterion,
            "iqr_multiplier": iqr_multiplier,
            "z_threshold": z_threshold,
            "alpha": alpha,
        }
    )
    return out
